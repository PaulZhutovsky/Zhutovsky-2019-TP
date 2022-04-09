import os
import numpy as np
import pandas as pd
import os.path as osp
from glob import glob
from rsnClf_sgn import extract_ic_id


PARENT_DIR = '/data/pzhutovsky/fMRI_data/PTSD_veterans/analysis/ptsd_dual_regression/'
# RESULT_DIR = osp.join(PARENT_DIR, 'ml_analysis')
RESULT_DIR = osp.join(PARENT_DIR, 'ml_analysis')
EVAL_FILE = osp.join(RESULT_DIR, 'evaluations_repeated10fold.npz')
PERM_FILES = sorted(glob(osp.join(RESULT_DIR, 'ic*_permutations1000.npz')))
METRICS = ['balanced_accuracy', 'sensitivity', 'specificity', 'AUC', 'positive_predictive_value',
           'negative_predictive_value']
RENAMED_METRICS = {'balanced_accuracy': 'acc', 'sensitivity': 'sens', 'specificity': 'spec', 'AUC': 'AUC',
                   'positive_predictive_value': 'PPV', 'negative_predictive_value': 'NPV'}


def get_signal_comp_ids(ic_signal_path):
    return extract_ic_id(ic_signal_path)


def load_results(file_path):
    with np.load(file_path) as tmp:
        return tmp['evaluations'], tmp['evaluations_labels'], tmp['rsn_path']


def load_permutations(file_paths):
    if not file_paths:
        return None, []
    n_perm = len(file_paths)
    evals_perm = []
    components = []

    for i_perm in xrange(n_perm):
        tmp = np.load(file_paths[i_perm])
        evals_perm.append(tmp['evaluations'])
        components.append(np.atleast_1d(tmp['ic_name']))
    return evals_perm, components


def calculate_p_value(neutral_permutation, permutations):
    n_perm = permutations.size
    return (np.sum(permutations >= neutral_permutation, dtype=np.float) + 1.)/(n_perm + 1)


def run(eval_file=EVAL_FILE, metrics=METRICS, renamed_metrics=RENAMED_METRICS, perm_files=PERM_FILES,
        save_dir=RESULT_DIR):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    evaluation_metrics, evaluation_metric_labels, evalution_rsn_files = load_results(eval_file)
    ic_labels = get_signal_comp_ids(evalution_rsn_files)
    evaluations_permutations, permuted_components = load_permutations(perm_files)
    assert ic_labels.size == evaluation_metrics.shape[0], "The networks results we stored for don't match the " \
                                                          "amount of signal networks!"
    n_cv = evaluation_metrics.shape[1]
    mean_cv_evals = evaluation_metrics.mean(axis=1)
    std_cv_evals = evaluation_metrics.std(axis=1)
    se_cv_evals = std_cv_evals/np.sqrt(n_cv)
    results = []
    full_labels = []

    for metric in metrics:
        p_value_dummy = np.ones(ic_labels.size) * np.nan
        renamed_metric = renamed_metrics[metric]
        id_metric = evaluation_metric_labels == metric

        full_labels = full_labels + [renamed_metric + ': mean',
                                     renamed_metric + ': SD',
                                     renamed_metric + ': SE',
                                     renamed_metric + ': p']
        for i, ic_perm in enumerate(permuted_components):
            ic_perm_id = extract_ic_id(ic_perm)
            id_ic = ic_labels == ic_perm_id
            avg_cv_perm = evaluations_permutations[i].mean(axis=1)
            avg_cv_perm_metric = avg_cv_perm[:, id_metric].squeeze()
            true_value = mean_cv_evals[id_ic, id_metric]
            assert true_value.size == 1, 'More than one true value for {}'.format(ic_perm_id)
            true_value = true_value[0]
            p_value_dummy[id_ic] = calculate_p_value(true_value, avg_cv_perm_metric)

        results.append(np.column_stack((mean_cv_evals[:, id_metric].squeeze(),
                                        std_cv_evals[:, id_metric].squeeze(),
                                        se_cv_evals[:, id_metric].squeeze(),
                                        p_value_dummy)))
    df_results = pd.DataFrame(data=np.column_stack(results), columns=full_labels, index=ic_labels)
    df_results.to_csv(osp.join(save_dir, 'results_table.csv'), na_rep='NA')


if __name__ == '__main__':
    run()
