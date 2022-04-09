import numpy as np
from sklearn.metrics import balanced_accuracy_score
import nibabel as nib
import warnings


def calculate_calibaration_curve(y_true, y_score, save_path):
    """
    :param y_score: nRepeatsCV, nSubjects
    :return:
    """

    n_repeats = y_score.shape[0]
    thresholds = np.arange(0., 0.5, 0.01)
    n_thresh = thresholds.size
    n_subj = y_score.shape[1]
    # for future convenience we will compute the absolute and relative amount of excluded subjects
    excluded = np.zeros((n_repeats, n_thresh, 2))
    accuracy = np.ones((n_repeats, n_thresh)) * np.nan
    accuracy_mean_sd = np.zeros((n_thresh, 2))
    for i_repeat in xrange(n_repeats):
        y_score_run = y_score[i_repeat]

        diff_score = np.abs(0.5 - y_score_run)

        for i_thresh in xrange(n_thresh):
            t = thresholds[i_thresh]
            included = diff_score >= t
            excluded[i_repeat, i_thresh, 0] = n_subj - included.sum()
            excluded[i_repeat, i_thresh, 1] = (n_subj - included.sum())/float(n_subj)

            y_included = y_score_run[included]
            # we can always use 0.5 as a cut-off because it either will not exclude anyone (i.e. 0.5 is the correct
            # cutoff) or it will exclude some people (i.e. threshold will be > 0.5 anyway)
            y_pred = (y_included >= 0.5).astype(np.float64)
            y_true_included = y_true[included]

            if included.sum() > 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    acc = balanced_accuracy_score(y_true=y_true_included, y_pred=y_pred)
                accuracy[i_repeat, i_thresh] = acc
    accuracy_mean_sd[:, 0] = np.nanmean(accuracy, axis=0)
    accuracy_mean_sd[:, 1] = np.nanstd(accuracy, axis=0)

    np.savez_compressed(save_path, excluded=excluded, accuracy=accuracy, accuracy_mean_sd=accuracy_mean_sd)

    return thresholds * 100, accuracy_mean_sd


def create_selected_features_img(mask_path, selected_features, store_path):
    mask = nib.load(mask_path)
    affine = mask.affine
    mask = np.array(mask.get_data(), dtype=np.bool)
    n_runs = selected_features.shape[0]
    selected_ftrs_perc = selected_features.sum(axis=0)/float(n_runs) * 100
    selected_ftrs_3d = np.zeros(mask.shape, dtype=np.float)
    selected_ftrs_3d[mask] = selected_ftrs_perc
    img = nib.Nifti1Image(selected_ftrs_3d, affine=affine)
    nib.save(img, store_path)
