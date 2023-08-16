# Import necessary packages:
import pandas as pd
import os
import numpy as np
import math
import pickle

print('|---STEP 1: PREPROCESSING BASELINE TADPOLE DATA ---|')







#--------------- PART 1: Loading data ---------------#
print('|1/4| Loading raw data...')

# Import raw TADPOLE data:
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
#tadpole_path = parent_directory + '/data/raw'
tadpole_path = 'data/raw'
tadpole_data = {}
tadpole_data['1: load'] = pd.read_csv(tadpole_path + '/TADPOLE_D1_D2.csv', low_memory=False)








#--------------- PART 2: Seperate data based on diagnosis ---------------#
print('|2/4| Seperating data based on diagnosis...')

# Seperate by baseline (BL) diagnosis (DX) of either MCI (EMCI or LMCI) or CN
temp = tadpole_data['1: load']
tadpole_data['2: filter_dx'] = {'mci_bl': temp[temp['DX_bl'].isin(['EMCI', 'LMCI'])],
                                         'cn_bl': temp[temp['DX_bl'].isin(['CN'])]}

# Filter set of patients with baseline (BL) CN DX
cn_filt = tadpole_data['2: filter_dx']['cn_bl']

# Filter DX column to check for any DX related to progression to MCI
cn_mci_filt = cn_filt[(cn_filt['DX']=='NL to MCI') | (cn_filt['DX']=='MCI')]

# Get 'viscode' which contains what month interval after BL the data 
# was collected for the patients
cn_mci_fixviscode = cn_mci_filt.copy()
for x in range(len(cn_mci_fixviscode['VISCODE'])):
    cn_mci_fixviscode['VISCODE'].iloc[x] = int(cn_mci_fixviscode['VISCODE'].iloc[x].strip('m'))

# Create empty dataframe to hold all patient data of BL DX of CN and future DX of MCI
cn_mci_final = pd.DataFrame()

# Loop through all patients
for rid in np.unique(cn_mci_filt['RID']):

    # Loop through each patient's viscode to find earliest visit that they were diagnosed with MCI, 
    # store earliest visit data in 'temp_row'
    for i, vcode in enumerate(cn_mci_fixviscode[cn_mci_fixviscode['RID']==rid]['VISCODE']):
        if i == 0:
            temp_vcode = vcode
            temp_row = cn_mci_fixviscode[(cn_mci_fixviscode['RID']==rid) & (cn_mci_fixviscode['VISCODE']==vcode)]
        if (i > 0) & (vcode < temp_vcode):
            temp_vcode = vcode
            temp_row = cn_mci_fixviscode[(cn_mci_fixviscode['RID']==rid) & (cn_mci_fixviscode['VISCODE']==vcode)]
    
    # Loop through each viscode to find the most recent DX
    for i, vcode in enumerate(cn_filt[cn_filt['RID']==rid]['VISCODE']):
        dx_state = list(cn_filt[(cn_filt['RID']==rid) & (cn_filt['VISCODE']==vcode)]['DX'])[0]
        if i == 0 and ~pd.isnull(dx_state) and (dx_state!='nan') and (type(dx_state)!=float):
            last_visit = vcode
            last_dx = dx_state
        if i > 0 and vcode > last_visit and ~pd.isnull(dx_state) and (dx_state!='nan') and (type(dx_state)!=float):
            last_visit = vcode
            last_dx = dx_state
    
    # Fill dataframe containing column of first MCI DX visit data and most recent DX
    temp_row.insert(loc = 0, column = 0, value = last_dx)
    cn_mci_final = pd.concat([cn_mci_final, temp_row], axis=0, ignore_index=True)

cn_mci_final.rename(columns={0: 'last_dx'}, inplace=True)

# Store in tadpole_data
tadpole_data['2: filter_dx']['cn_bl_to_mci'] = cn_mci_final









#--------------- PART 3: Collect only baseline data ---------------#
print('|3/4| Collecting baseline data...')

tadpole_data['3: baseline_data_and_last_dx'] = {}

# Get only data from baseline, and store the last diagnosis in the first column (this was already done
# for CN patients who progressed to MCI, here is ):
for dx in ['cn_bl', 'mci_bl']:
    all_visits = tadpole_data['2: filter_dx'][dx]
    baseline = pd.DataFrame()
    temp_dx = pd.DataFrame()

    # Loop through patient identifiers (RID)
    for rid in all_visits['RID'].unique():

        # Get all patient visits
        patient_visits = all_visits.loc[all_visits['RID'] == rid]

        # Check that the patient has had more than 1 visit
        if patient_visits.shape[0] > 1:
            
            # Get data from baseline and collect last diagnosis
            patient_baseline = patient_visits.loc[patient_visits['VISCODE'] == 'bl']
            last_dx = patient_visits['DX']
            last_dx = last_dx.loc[[last_dx.last_valid_index()]]

            # Make sure dx is not null
            if last_dx.iloc[0] != '' and not (pd.isnull(last_dx.iloc[0])):
                temp_dx = pd.concat([temp_dx, last_dx])
                baseline = pd.concat([baseline, patient_baseline], ignore_index=True)
                
    baseline.insert(0, 'last_dx', list(temp_dx[0]))
    tadpole_data['3: baseline_data_and_last_dx'][dx] = baseline
tadpole_data['3: baseline_data_and_last_dx']['cn_bl_to_mci'] = tadpole_data['2: filter_dx']['cn_bl_to_mci']








#--------------- PART 4: Declare modalities and save ---------------#
print('|4/4| Declaring modalities and saving...')

# Declare indices of features within each modality
feature_idx_by_modality = {'main_cognitive_tests': [22, 43],
                           'mri_roi': [487, 833],
                           'mri_vols': [48, 55],
                           'fdg_pet_roi': [839, 1173],
                           'av45_pet_roi': [1175, 1413],
                           'av1451_pet_roi': [1417, 1657],
                           'dti_roi': [1668, 1896],
                           'csf_bio': [1903, 1906],
                           'other': [12, 22]}

# Create final tadpole data dictionary to store features by modality
tadpole_data['4: filter_by_mode'] = {}

# By MCI, CN, or CN BL to MCI
for dx in tadpole_data['3: baseline_data_and_last_dx']:

    tadpole_data['4: filter_by_mode'][dx] = {}

    for mode in feature_idx_by_modality:

        idx0 = feature_idx_by_modality[mode][0]
        idx1 = feature_idx_by_modality[mode][1]
        mode_features = tadpole_data['3: baseline_data_and_last_dx'][dx].iloc[:, idx0:idx1]

        # Replace null characters with actual nans. This is slightly different for 'other' and 
        # csf_bio modalities
        if mode == 'other':
            mode_features_nan_fill = mode_features.replace('/', 
                                    '', regex=True).replace(' ',
                                    '', regex=True).replace('',
                                    np.nan, regex=True).replace(-4,
                                    np.nan, regex=True)
            # Store baseline diagnosis to 'other'
            baseline_dx = tadpole_data['3: baseline_data_and_last_dx'][dx]['DX_bl']
            mode_features_nan_fill = pd.concat([baseline_dx, mode_features_nan_fill], axis=1)

        elif mode == 'csf_bio':
            mode_features_nan_fill = mode_features.replace(r'^\s*$',
                                    np.nan, regex=True).replace('<',
                                    '', regex=True).replace('>', 
                                    '', regex=True).replace(-4,
                                    np.nan, regex=True)
            
        else:
            mode_features_nan_fill = mode_features.replace('', 
                                    np.nan, regex=True).replace(' ',
                                    np.nan, regex=True).replace(-4,
                                    np.nan, regex=True)
            
        tadpole_data['4: filter_by_mode'][dx][mode] = mode_features_nan_fill

# --- Part 4.5: Map MRI names to dictionary names --- #
tadpole_dict = pd.read_csv(tadpole_path + '/TADPOLE_D1_D2_Dict.csv')
for dx in tadpole_data['4: filter_by_mode']:
    for feat in tadpole_data['4: filter_by_mode'][dx]['mri_roi']:
        dict_row = tadpole_dict[tadpole_dict['FLDNAME'] == feat]
        text_name = list(dict_row['TEXT'])[0]
        tadpole_data['4: filter_by_mode'][dx]['mri_roi'].rename(columns={feat: text_name}, inplace=True)

# Get labels (last dx) for each patient
tadpole_labels = {}
for dx in tadpole_data['3: baseline_data_and_last_dx']:
    tadpole_labels[dx] = tadpole_data['3: baseline_data_and_last_dx'][dx]['last_dx']

# Save processed TADPOLE data!
processed_data_path = 'data/processed'
with open(f'{processed_data_path}/tadpole_data_baseline.pickle', 'wb') as file:
    pickle.dump(tadpole_data['4: filter_by_mode'], file)

with open(f'{processed_data_path}/tadpole_labels_baseline.pickle', 'wb') as file:
    pickle.dump(tadpole_labels, file)

print('Done!')
