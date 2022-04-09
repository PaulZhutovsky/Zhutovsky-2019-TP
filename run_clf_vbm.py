from rsnClf_sgn import get_labels, classification, PARENT_DIR_DR, SAVE_FOLDER
from evaluation_classifier import Evaluator
from create_final_clinical_file import extract_subjid_from_path
from glob import glob
import nibabel as nib
import os.path as osp
import os
import numpy as np


PROJECT_PATH = '/data/pzhutovsky/fMRI_data/PTSD_veterans/'
VBM_PATH = osp.join(PROJECT_PATH, 'analysis/ptsd_vbm/vbm_data_all.nii.gz')
RSN_DR_FILE = np.loadtxt(osp.join(PARENT_DIR_DR, 'dual_regression_ptsd_filtered.txt'), dtype=str)


def get_data():
    if glob(VBM_PATH):
        return np.array(nib.load(VBM_PATH).get_data(), dtype=np.float)

    # we will ensure that the order is the same as for the resting-state classification
    subj_ids = extract_subjid_from_path(RSN_DR_FILE)
    vbm_files = []
    for subj_id in subj_ids:
        vbm_file = glob(osp.join(PROJECT_PATH, 'derivatives', '*' + subj_id, 'ses-T0', 'anat', 'vbm', 'smwc1*.nii'))
        assert len(vbm_file) == 1, 'More than one file for {}'.format(subj_id)
        vbm_files.append(vbm_file[0])
    os.system('fslmerge -t {} {}'.format(VBM_PATH, ' '.join(vbm_files)))
    return np.array(nib.load(VBM_PATH).get_data(), dtype=np.float)


def mask_data(vbm_arr, indv_thresh=0.2, across_thresh=0.5):
    masks = np.zeros_like(vbm_arr, dtype=bool)

    for i_subj in xrange(vbm_arr.shape[-1]):
        masks[..., i_subj] = vbm_arr[..., i_subj] >= indv_thresh

    mask_mean = masks.mean(axis=-1)
    mask = mask_mean > across_thresh

    return vbm_arr[mask, :].T


def run():
    vbm_arr = get_data()
    X = mask_data(vbm_arr)
    y = get_labels()
    print X.shape, y.size
    eval_labels = Evaluator().evaluate_labels()

    evaluations, predictions, seed, chosen_z_vals, picked_ftrs = classification(X, y, n_jobs=5)
    np.savez_compressed(osp.join(SAVE_FOLDER, 'evaluations_VBM_repeated10fold.npz'),
                        evaluations=evaluations, seeds=seed, evaluations_labels=eval_labels,
                        predictions=predictions, picked_z_vals=chosen_z_vals,
                        chosen_ftrs=picked_ftrs, vbm_path=VBM_PATH)


if __name__ == '__main__':
    run()