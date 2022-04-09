import os
import os.path as osp
import warnings
from glob import glob
from time import time

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.preprocessing import RobustScaler
import shogun as sgn

from evaluation_classifier import Evaluator
from feature_selector import FeatureSelector

ANALYSIS_DIR = '/data/pzhutovsky/fMRI_data/PTSD_veterans/analysis'
PARENT_DIR_DR = osp.join(ANALYSIS_DIR, 'ptsd_dual_regression')
PARENT_DIR_MELODIC = osp.join(ANALYSIS_DIR, 'controls_ICA_aroma.gica')
SIGNAL_COMPONENTS = osp.join(PARENT_DIR_MELODIC, 'dim70', 'meta_melodic_dim70',
                             'classification.csv')
LABEL_FILE = osp.join(PARENT_DIR_DR, 'labels.csv')
DR_FOLDER = osp.join(PARENT_DIR_DR, 'dual_regression_dim70.dr')
IC_COMPONENTS_PATH = np.array(sorted(glob(osp.join(DR_FOLDER, 'dr_stage2_ic*.nii.gz'))))
CV = 'repeated10fold'
N_JOBS = 5
N_PERM = 1
ACC_PERM_CRIT = 0.65            # only networks with at least 65% accuracy will be investigated
SAVE_FOLDER = osp.join(PARENT_DIR_DR, 'ml_analysis')
SAVE_NAME = 'evaluations_{}.npz'.format(CV)


def extract_ic_id(ic_signal):
    # step 1: remove .nii.gz
    # step 2: partition on '_' and take the last part == ic-label from the path
    return np.char.rpartition(np.char.rpartition(ic_signal, '.nii.gz')[:, 0], '_')[:, -1]


def get_zthresh():
    return np.arange(2.5, 4.1, 0.1)


def get_labels(label_file=LABEL_FILE, transform_0_to_neg1=True):
    y = pd.read_csv(label_file).reduction_30.values
    if transform_0_to_neg1:
        y[y == 0] = -1
    return y


def load_data(data_file, dtype=np.float64):
    return np.array(nib.load(data_file).get_data(), dtype=dtype)


def get_signal_ids(signal_components=SIGNAL_COMPONENTS):
    return pd.read_csv(signal_components).SIGNAL.values


def classification(X, y, cv_type='repeated10fold', n_jobs=1, print_output=True, seed=None):
    evaluator = Evaluator()
    eval_labels = evaluator.evaluate_labels()

    if not seed:
        seed = int(time())

    cv, n_splits, n_repeats = get_cv_and_folds(X, y, cv_type, seed, print_output=False)
    z_thresh = get_zthresh()
    # (hard prediction and probability)
    predictions = np.zeros((n_repeats, y.size, 2))
    evaluations = np.zeros((n_splits, len(eval_labels)))
    chosen_z_vals = np.zeros(n_splits)
    picked_ftrs = np.zeros((n_splits, X.shape[1]), dtype=np.int)

    t1 = time()

    eval_runs = Parallel(n_jobs=n_jobs, verbose=1)(delayed(one_cv_run)(X, y, train_id, test_id, evaluator, z_thresh,
                                                                       seed + i_cv + 1)
                                                   for i_cv, (train_id, test_id) in enumerate(cv.split(X, y)))

    for (i_cv, (_, test_id)) in enumerate(cv.split(X, y)):
        # integer divison (Python 3 approved)
        i_repeat = i_cv // n_repeats
        evaluations[i_cv] = eval_runs[i_cv][0]
        predictions[i_repeat, test_id] = eval_runs[i_cv][1]
        chosen_z_vals[i_cv] = eval_runs[i_cv][2]
        picked_ftrs[i_cv] = eval_runs[i_cv][3]

    if print_output:
        print
        print eval_labels
        print evaluations.mean(axis=0)
        print
    t2 = time()
    print 'Time taken: {}'.format(round((t2 -t1)/60., 2))
    return evaluations, predictions.squeeze(), seed, chosen_z_vals, picked_ftrs


def gpc_sgn(Xtrain, ytrain, Xtest):
    y_trn_sgn = sgn.BinaryLabels(ytrain)
    # shogun expects the data to be transposed from the sklearn format
    X_trn_sgn = sgn.RealFeatures(Xtrain.T)
    X_tst_sgn = sgn.RealFeatures(Xtest.T)

    mean_function = sgn.ZeroMean()
    kernel = sgn.LinearKernel()
    kernel.set_normalizer(sgn.SqrtDiagKernelNormalizer())
    likelihood = sgn.ProbitLikelihood()
    inference_method = sgn.EPInferenceMethod(kernel, X_trn_sgn, mean_function, y_trn_sgn, likelihood)
    gp_classifier = sgn.GaussianProcessClassification(inference_method)
    gp_classifier.train()
    y_score = gp_classifier.get_probabilities(X_tst_sgn)
    y_pred = gp_classifier.apply_binary(X_tst_sgn)
    return y_pred.get_labels(), y_score


def one_cv_run(X, y, train_id, test_id, evaluator, z_thresh, seed):
    Xtrain, ytrain = X[train_id], y[train_id]
    Xtest, ytest = X[test_id], y[test_id]

    cv_inner, n_splits, _ = get_cv_and_folds(Xtrain, ytrain, 'stratified5fold', seed=seed, print_output=False)
    grid_score = np.zeros((n_splits, z_thresh.size))

    for i_inner, (train_inner, test_inner) in enumerate(cv_inner.split(Xtrain, ytrain)):
        for i_z, z in enumerate(z_thresh):
            Xtrain_inner, Xtest_inner = Xtrain[train_inner], Xtrain[test_inner]
            ytrain_inner, ytest_inner = ytrain[train_inner], ytrain[test_inner]

            feat_sel = FeatureSelector(z_thresh=z)
            Xtrain_inner = feat_sel.fit_transform(Xtrain_inner, ytrain_inner)
            Xtest_inner = feat_sel.transform(Xtest_inner)

            scl = RobustScaler()
            Xtrain_inner = scl.fit_transform(Xtrain_inner)
            Xtest_inner = scl.transform(Xtest_inner)

            y_pred, _ = gpc_sgn(Xtrain_inner, ytrain_inner, Xtest_inner)
            grid_score[i_inner, i_z] = balanced_accuracy_score(ytest_inner, y_pred)

    id_z_best = grid_score.mean(axis=0).argmax()
    z_best = z_thresh[id_z_best]

    feat_sel = FeatureSelector(z_thresh=z_best)
    Xtrain = feat_sel.fit_transform(Xtrain, ytrain)
    Xtest = feat_sel.transform(Xtest)

    scl = RobustScaler()
    Xtrain = scl.fit_transform(Xtrain)
    Xtest = scl.transform(Xtest)

    ypred, yscore = gpc_sgn(Xtrain, ytrain, Xtest)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        eval_run = evaluator.evaluate_prediction(y_true=ytest, y_pred=ypred, y_score=yscore)
    return eval_run, np.column_stack((ypred, yscore)), z_best, feat_sel.chosen_ftrs


def get_cv_and_folds(X, y, cv_type, seed=42, print_output=True):
    unq, counts = np.unique(y, return_counts=True)

    if print_output:
        print 'Classes: {}; Counts: {}'.format(unq, counts)

    if cv_type == 'LOGO':
        assert np.all(np.diff(counts)) == 0, 'Classes have different distribution. LOGO cannot be used!'
        # leave-one-per-group-out
        cv = StratifiedKFold(n_splits=y.size / 2, random_state=seed)
        n_splits, n_repeats = y.size / 2, 1
    elif cv_type == 'shufflesplit':
        cv = StratifiedShuffleSplit(n_splits=200, test_size=0.2, random_state=seed)
        n_splits, n_repeats = 200, 1
    elif cv_type == 'repeated10fold':
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=seed)
        n_splits, n_repeats = 100, 10
    else:
        # 5-fold cv
        cv = StratifiedKFold(n_splits=5, random_state=seed)
        n_splits, n_repeats = 5, 1
    return cv, n_splits, n_repeats


def load_all_components(components, mask, num_subj):
    num_voxel = mask.sum()
    num_comp = components.size
    all_components = np.zeros((num_comp, num_subj, num_voxel), dtype=np.float64)

    print 'Loading all components'
    for i_comp, comp in enumerate(components):
        all_components[i_comp] = load_data(comp)[mask, :].T
    return all_components


def run(ic_components_path=IC_COMPONENTS_PATH, cv_type=CV, n_jobs=N_JOBS, n_perm=N_PERM, save_folder=SAVE_FOLDER,
        signal_components=SIGNAL_COMPONENTS, label_file=LABEL_FILE, save_name=SAVE_NAME, acc_crit=ACC_PERM_CRIT):
    if not osp.exists(save_folder):
        os.makedirs(save_folder)

    mask = get_mask()

    signal_comp = get_signal_ids(signal_components=signal_components)

    print 'Total #ICs: {}; Signal Components: {}'.format(ic_components_path.size, np.sum(signal_comp))

    y = get_labels(label_file=label_file, transform_0_to_neg1=True)

    n_subject = y.size
    rsn_signal_path = ic_components_path[signal_comp]
    n_rsn = rsn_signal_path.size
    rsn_signal = load_all_components(rsn_signal_path, mask, n_subject)
    _, n_splits, n_repeats = get_cv_and_folds(np.zeros_like(y), y, cv_type)
    eval_labels = np.array(Evaluator().evaluate_labels())
    n_evals = eval_labels.size

    evaluation_RSNs = np.zeros((n_rsn, n_splits, n_evals))
    prediction_RSNs = np.zeros((n_rsn, n_repeats, n_subject, 2))
    seeds_cv = np.zeros(n_rsn, dtype=np.int)
    picked_z_vals_RSNs = np.zeros((n_rsn, n_splits))
    chosen_ftrs_RSNs = np.zeros((n_rsn, n_splits, mask.sum()), dtype=np.int)

    t1 = time()
    print 'Start classification'
    for i_rsn in xrange(n_rsn):
        print 'IC: {}/{}'.format(i_rsn + 1, n_rsn)

        X = rsn_signal[i_rsn]
        evaluation_RSNs[i_rsn], prediction_RSNs[i_rsn], seeds_cv[i_rsn], picked_z_vals_RSNs[i_rsn], \
        chosen_ftrs_RSNs[i_rsn] = classification(X, y, cv_type=cv_type, n_jobs=n_jobs)
    t2 = time()
    print 'Time taken overall: {}'.format(round((t2 - t1)/60., 2))
    np.savez_compressed(osp.join(save_folder, save_name), evaluations=evaluation_RSNs, seeds=seeds_cv,
                        evaluations_labels=eval_labels, predictions=prediction_RSNs, picked_z_vals=picked_z_vals_RSNs,
                        chosen_ftrs=chosen_ftrs_RSNs, rsn_path=rsn_signal_path)

    if n_perm > 1:
        print 'Start permutation tests'
        permutation_tests(cv_type, eval_labels, evaluation_RSNs, rsn_signal, rsn_signal_path, n_jobs, n_perm, n_splits,
                          y, acc_crit=acc_crit, save_folder=save_folder)


def permutation_tests(cv_type, eval_labels, evaluation_RSNs, rsn_signal, rsn_signal_path, n_jobs, n_perm, n_splits,
                      y, acc_crit=ACC_PERM_CRIT, save_folder=SAVE_FOLDER):

    assert rsn_signal.shape[0] == rsn_signal_path.size, 'Signal RSN data and signal RSN path has different n_network!'
    assert evaluation_RSNs.shape[0] == rsn_signal.shape[0], 'Missmatch between RSN evaluations and RSN data'

    n_evals = eval_labels.size
    id_acc = eval_labels == 'balanced_accuracy'
    eval_RSN_acc = evaluation_RSNs[:, :, id_acc].squeeze()
    RSN_mean_acc = eval_RSN_acc.mean(axis=1)
    id_RSN_to_permute = RSN_mean_acc >= acc_crit
    RSN_to_permute = rsn_signal[id_RSN_to_permute]
    RSN_to_permute_path = rsn_signal_path[id_RSN_to_permute]
    RSN_id = extract_ic_id(RSN_to_permute_path)
    n_to_permute = RSN_to_permute_path.size
    from_sec_to_h = 60.*60.  # to "speed up" the calculation

    print '#ICs with > {} accuracy: {}'.format(acc_crit, n_to_permute)
    t1_total = time()
    for i_rsn in xrange(n_to_permute):
        print RSN_to_permute_path[i_rsn]
        print 'IC: {}/{}'.format(i_rsn + 1, RSN_to_permute.shape[0])
        save_file = '{}_permutations{}'.format(RSN_id[i_rsn], n_perm)
        print 'save-name: {}'.format(save_file)
        eval_perm = np.zeros((n_perm, n_splits, n_evals))
        y_perm = np.copy(y)
        t1_perm = time()
        for i_perm in xrange(n_perm):
            print '#perm: {}/{}'.format(i_perm + 1, n_perm)
            np.random.shuffle(y_perm)
            X = RSN_to_permute[i_rsn]

            eval_perm[i_perm], _, _, _, _ = classification(X, y_perm, cv_type=cv_type, n_jobs=n_jobs, print_output=False)
        print 'Saving permutations'
        np.savez_compressed(osp.join(save_folder, save_file),
                            evaluations=eval_perm,
                            evaluations_labels=eval_labels,
                            rsn_name=np.array(RSN_to_permute_path[i_rsn]))
        t2_perm = time()
        print 'Time permutations: {:0.2f}h'.format((t2_perm - t1_perm)/from_sec_to_h)
    t2_total = time()
    print 'Total time: {:0.2f}days'.format((t2_total - t1_total)/(from_sec_to_h * 24))


def get_mask():
    return load_data(osp.join(DR_FOLDER, 'mask.nii.gz'), dtype=np.bool)


if __name__ == '__main__':
    print CV
    run()
