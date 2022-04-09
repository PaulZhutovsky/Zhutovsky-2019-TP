import pandas as pd
import numpy as np
import os.path as osp


PROJECT_PATH = '/data/pzhutovsky/fMRI_data/PTSD_veterans'
ORIG_CLIN_FILE = osp.join(PROJECT_PATH, 'sourcedata', 'BETER_Geuze_AMS_ML_20170118.csv')
RESPONSE_NONRESPONSE_FILE = osp.join(PROJECT_PATH, 'analysis', 'ptsd_dual_regression', 'labels.csv')
CONTROLS_USED = osp.join(PROJECT_PATH, 'analysis', 'controls_ICA_aroma.gica',
                         'controls_preprocessed_aroma_final_without21_rejected_Sattleworth.txt')
PTSD_USED = osp.join(PROJECT_PATH, 'analysis', 'ptsd_dual_regression', 'dual_regression_ptsd_filtered.txt')
CONTROLS_FD = osp.join(PROJECT_PATH, 'analysis', 'controls_ICA_aroma.gica', 'motion_metrics.csv')
PTSD_FD = osp.join(PROJECT_PATH, 'analysis', 'ptsd_dual_regression', 'motion_metrics.csv')
PTSD_TIV = osp.join(PROJECT_PATH, 'analysis', 'ptsd_vbm', 'ptsd_TIV.csv')


def extract_subjid_from_path(array_path_files):
    """
    1. Partition on 'sub-' (the next word will be ptsd or control);
    2. Partition on '_' (the part before is ptsd/control + number, part after is subject id)
    3. Partition on '/' (the part before is subject id)
    :param array_path_files:
    :return:
    """
    return np.char.partition(np.char.partition(np.char.partition(array_path_files, 'sub-')[:, -1], '_')[:, -1],
                             '/')[:, 0]


def load_TIV():
    return pd.read_csv(PTSD_TIV)


def load_fd():
    df_control = pd.read_csv(CONTROLS_FD)
    df_control['subj_id'] = extract_subjid_from_path(df_control.subj_id.values.astype(str))

    df_ptsd = pd.read_csv(PTSD_FD)
    df_ptsd['subj_id'] = extract_subjid_from_path(df_ptsd.subj_id.values.astype(str))
    return pd.concat((df_control[['subj_id', 'FD_mean']], df_ptsd[['subj_id', 'FD_mean']]), axis=0, ignore_index=True)


def load_labels_file():
    return pd.read_csv(RESPONSE_NONRESPONSE_FILE)[['orig_id', 'reduction_30']]


def load_subj_used():
    return np.loadtxt(CONTROLS_USED, dtype=str), np.loadtxt(PTSD_USED, dtype=str)


def load_clinical_file():
    return pd.read_csv(ORIG_CLIN_FILE)


def add_response_status(df_data, df_response):
    ids_data = df_data.Name.values.astype(str)
    ids_response = df_response.orig_id.values.astype(str)
    df_data['response'] = np.nan
    n_controls = (df_data.Group == "Control").sum()

    for i, id_response in enumerate(ids_response):
        id_subj = ids_data == id_response
        assert id_subj.sum() == 1, 'More than one subject with id {}'.format(ids_response[id_response])
        assert (id_subj.argmax() - n_controls) == i, 'Data not ordered'
        df_data.loc[id_subj, 'response'] = df_response.reduction_30[df_response.orig_id == id_response].values

    assert np.all((df_data.Group == 'Control') == (df_data.response.isnull())), 'Something failed at adding response'
    df_data.response = df_data.response.replace({0: 'Non-Responder', 1: 'Responder'})
    df_data.loc[df_data.response == 'Responder', 'Group'] = 'PTSD-Responder'
    df_data.loc[df_data.response == 'Non-Responder', 'Group'] = 'PTSD-Non-Responder'
    return df_data


def add_FD(df_data, df_fd):
    df_data['FD'] = np.nan
    ids_data = df_data.Name.values.astype(str)
    ids_fd = df_fd.subj_id.values.astype(str)

    for id_data in ids_data:
        id_subj = id_data == ids_fd
        assert id_subj.sum() == 1, 'More than one subject with id {}'.format(ids_data[id_data])
        df_data.loc[ids_data == id_data, 'FD'] = df_fd.FD_mean.loc[id_subj].values

    assert np.all(df_data.FD.notnull()), 'Missing FD value'
    return df_data


def add_TIV(df_data, df_TIV):
    df_data['TIV'] = np.nan
    ids_data = df_data.Name.values.astype(str)
    ids_tiv = df_TIV.subj_id.values.astype(str)

    for id_tiv in ids_tiv:
        id_subj = ids_data == id_tiv
        assert id_subj.sum() == 1, 'More than one subject with id {}'.format(ids_tiv[id_tiv])
        df_data.loc[id_subj, 'TIV'] = df_TIV.loc[ids_tiv == id_tiv, 'TIV'].values

    return df_data


def extract_used_subj(df_all, ids_to_extract):
    ids_all = df_all.Name.values.astype(str)
    ids_subj = []
    for id_to_extract in ids_to_extract:
        id_subj = ids_all == id_to_extract

        assert id_subj.sum() == 1, 'More than one subject with id {}'.format(id_to_extract)
        # quick way to extract the id of a boolean array
        ids_subj.append(np.argmax(id_subj))

    return df_all.iloc[ids_subj].reset_index(drop=True)


def calculate_age(df):
    return (pd.to_datetime(df.T0_KA_datum) - pd.to_datetime(df.Geboorte_datum)).astype('timedelta64[Y]')


def clean_data(df_clin_used):
    df_clin_used['age_at_T0'] = calculate_age(df_clin_used)
    df_clin_used['perc_decr'] = np.nan
    id_ptsd = df_clin_used.Group != 'Control'
    df_clin_used['perc_decr'] = ((df_clin_used.loc[id_ptsd, 'T0_CAPS_TOTAAL'].astype(float) -
                                  df_clin_used.loc[id_ptsd, 'T6_CAPS_TOTAAL'].astype(float))
                                 / df_clin_used.loc[id_ptsd, 'T0_CAPS_TOTAAL'].astype(float)) * 100

    df_clin_used['years_since_last_depl'] = (pd.to_datetime(df_clin_used.T0_KA_datum) -
                                             pd.to_datetime(df_clin_used.Datum_laatste_uitzending,
                                                            errors='coerce')).astype('timedelta64[Y]')
    df_clin_used.rename(columns={'Rechts_Links': 'Handedness',
                                 'Opleiding_corrected': 'Education_self',
                                 'Opleiding_moeder': 'Education_mother',
                                 'Opleiding_vader': 'Education_father',
                                 'Aant_uitzendingen': 'Number_of_times_deployed',
                                 'Depressie_huidig': 'Depression_T0',
                                 'Angststoornis_huidig': 'Anxiety_T0',
                                 'Somatoform_huidig': 'Somatoform_T0',
                                 'Depressie_huidig_T6': 'Depression_T6',
                                 'Angststoornis_huidig_T6': 'Anxiety_T6',
                                 'Somatoform_huidig_T6': 'Somatoform_T6',
                                 'Alcoholafhankelijkheid': 'Alcoholism'}, inplace=True)
    education_renaming = {'Geen': 0,
                          'Lager onderwijs, basisschool': 1,
                          'Lager beroepsonderwijs': 2,
                          '(M)ULO, mavo': 3,
                          'HAVO': 4,
                          'VWO': 5,
                          'MBO': 6,
                          'HBO': 7,
                          'Universiteit': 8,
                          'Anders': 9,
                          '9999': np.nan}
    df_clin_used.Education_mother = df_clin_used.Education_mother.map(education_renaming)
    df_clin_used.Education_father = df_clin_used.Education_father.map(education_renaming)
    df_clin_used.Depression_T0 = df_clin_used.Depression_T0.map({'geen huidige depressie': 'No',
                                                                 'huidige depressie': 'Yes'})
    df_clin_used.Anxiety_T0 = df_clin_used.Anxiety_T0.map({'geen comorbide angstoornis': 'No',
                                                           'comorbide angststoornis': 'Yes'})
    df_clin_used.Somatoform_T0 = df_clin_used.Somatoform_T0.map({'geen huidige somatoforme stoornis': 'No',
                                                                 'huidige somatoforme stoornis': 'Yes'})
    df_clin_used = df_clin_used.replace(['9999', 9999, ' '], [np.nan, np.nan, np.nan])

    ssri_T0 = df_clin_used.SSRI_T0.values.astype(np.float)
    sari_T0 = df_clin_used.SARI_T0.values.astype(np.float)
    ssri_nan_T0 = np.isnan(ssri_T0)
    sari_nan_T0 = np.isnan(sari_T0)
    sri_T0 = ((ssri_T0 == 1) | (sari_T0 == 1)).astype(np.float)
    sri_T0[ssri_nan_T0 | sari_nan_T0] = np.nan

    ssri_T6 = df_clin_used.SSRI_T6.values.astype(np.float)
    sari_T6 = df_clin_used.SARI_T6.values.astype(np.float)
    ssri_nan_T6 = np.isnan(ssri_T6)
    sari_nan_T6 = np.isnan(sari_T6)
    sri_T6 = ((ssri_T6 == 1) | (sari_T6 == 1)).astype(np.float)
    sri_T6[ssri_nan_T6 | sari_nan_T6] = np.nan

    df_clin_used['SRI_T0'] = sri_T0
    df_clin_used['SRI_T6'] = sri_T6

    # Confirmed from Elbert that be3i (BET083) received 8-10 EMDR sessions (will set it to 9);
    # -be5h (BET075) received EMDR
    # -be6h (BET076) received CBT + EMDR
    # -be8h (BET078) received CBT + EMDR
    # -be3i (BET081) received CBT + structured treatment sessions
    # but we don't know how _many_ sessions they received... so I set it to NaN
    df_clin_used.loc[df_clin_used.PPNnr == 'BET083', 'EMDR_between_T0_T6'] = 9
    df_clin_used.loc[df_clin_used.PPNnr == 'BET083', 'CBT_between_T0_T6'] = 0
    df_clin_used.loc[df_clin_used.PPNnr == 'BET083', 'ECL_between_T0_T6'] = 0
    df_clin_used.loc[df_clin_used.PPNnr == 'BET083', 'Structs_between_T0_T6'] = 0

    id_bet075 =  df_clin_used.PPNnr == 'BET075'
    id_bet076 = df_clin_used.PPNnr == 'BET076'
    id_bet078 = df_clin_used.PPNnr == 'BET078'
    id_bet081 = df_clin_used.PPNnr == 'BET081'
    id_missing = id_bet075 | id_bet076 | id_bet078 | id_bet081
    all_therapies = ['EMDR_between_T0_T6', 'CBT_between_T0_T6', 'ECL_between_T0_T6', 'Structs_between_T0_T6']
    df_clin_used.loc[id_missing, all_therapies] = np.nan

    # some controls (which DID NOT receive therapy) have NaN values somehow. I will manually replace those with 0
    # just for consistency
    df_clin_used.loc[df_clin_used.Group == 'Control', all_therapies] = 0.

    all_sessions = df_clin_used.loc[:, all_therapies].astype(np.float)
    df_clin_used['total_number_sessions'] = all_sessions.sum(axis=1, skipna=False)

    df_clin_used[all_therapies] = df_clin_used[all_therapies].astype(np.float)

    # we also want to just know about therapy types (independent of how many sessions)
    # for the 4 missing subjects (where we don't know how many sessions they had), we know WHICH therapies they had
    therapies_used = np.array(['EMDR_used', 'CBT_used', 'ECL_used', 'Structs_used'])
    df_clin_used[therapies_used] = (df_clin_used[all_therapies] > 0).astype(np.int)
    df_clin_used.loc[id_bet075, therapies_used] = [1, 0, 0, 0]
    df_clin_used.loc[id_bet076, therapies_used] = [1, 1, 0, 0]
    df_clin_used.loc[id_bet078, therapies_used] = [1, 1, 0, 0]
    df_clin_used.loc[id_bet081, therapies_used] = [0, 1, 0, 1]

    df_clin_used['therapies_used_categorical'] = np.nan
    used_therapies_ohot = df_clin_used[therapies_used].values
    
    for i in xrange(df_clin_used.shape[0]):
        subj_therapies = used_therapies_ohot[i].astype(bool)

        if subj_therapies.any():
            therapies_names = therapies_used[subj_therapies]
            if therapies_names.size > 1:
                df_clin_used.loc[i, 'therapies_used_categorical'] = ' + '.join(therapies_names)
            else:
                df_clin_used.loc[i, 'therapies_used_categorical'] = therapies_names[0]
    return df_clin_used


def run():
    df_clin_file = load_clinical_file()
    df_fd = load_fd()
    df_tiv = load_TIV()
    df_labels_file = load_labels_file()
    controls_file, ptsd_file = load_subj_used()
    controls_subjid = extract_subjid_from_path(controls_file)
    ptsd_subjid = extract_subjid_from_path(ptsd_file)

    df_clin_used = extract_used_subj(df_clin_file, np.concatenate((controls_subjid, ptsd_subjid)))
    df_clin_used = add_response_status(df_clin_used, df_labels_file)
    df_clin_used = add_FD(df_clin_used, df_fd)
    df_clin_used = add_TIV(df_clin_used, df_tiv)
    df_clin_used_clean = clean_data(df_clin_used)
    df_clin_used_clean.to_csv(osp.join(PROJECT_PATH, 'analysis', 'clean_clinical_data.csv'), index=False,
                              na_rep='NA')


if __name__ == '__main__':
    run()
