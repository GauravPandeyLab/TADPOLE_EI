# Import necessary packages:
import pandas as pd
import os
import numpy as np
import math
import pickle
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
print('|---STEP 2: PROCESSING TRAIN/TEST SPLIT FOR TADPOLE DATA ---|')








#--------------- PART 1: Loading preprocessed data ---------------#
print('|1/5| Loading preprocessed data...')

# Load preprocessed TADPOLE data
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
processed_path = 'data/processed'
data = {}
labels = {}
with open(processed_path + '/tadpole_data_baseline.pickle', 'rb') as file:
    data['1: load'] = pickle.load(file)

with open(processed_path + '/tadpole_labels_baseline.pickle', 'rb') as file:
    labels['1: load'] = pickle.load(file)

# Label mapping (switching various diagnoses to either stable (CN or MCI) or further 
# demented (DEM)):
cn_bl_label_map = {'CN': ['NL', 'MCI to NL'],
                   'DEM': ['NL to MCI', 'MCI', 
                           'Dementia', 'MCI to Dementia',
                           'Dementia to MCI', 'NL to Dementia']}
mci_bl_label_map = {'MCI': ['MCI', 'MCI to NL', 'NL',
                            'NL to MCI', 'Dementia to MCI'],
                    'DEM': ['Dementia', 'MCI to Dementia']}

def map_items(item, mapping):
    for key, values in mapping.items():
        if item in values:
            return key
    return 'Unknown'

labels['1.5: map'] = {}
labels['1.5: map']['cn_bl'] = labels['1: load']['cn_bl'].apply(lambda x: map_items(x, cn_bl_label_map))
labels['1.5: map']['mci_bl'] = labels['1: load']['mci_bl'].apply(lambda x: map_items(x, mci_bl_label_map))
labels['1.5: map']['cn_bl_to_mci'] = labels['1: load']['cn_bl_to_mci'].apply(lambda x: map_items(x, mci_bl_label_map))








#--------------- PART 2: Splitting into training and testing sets ---------------#
print('|2/5| Splitting into training and testing sets...')

data['2: train test split'] = {}
labels['2: train test split'] = {}
splits = ['full', 'train', 'test']

for dx in ['cn_bl', 'mci_bl']:

    data['2: train test split'][dx] = {}
    labels['2: train test split'][dx] = {}

    for split in splits:
        
        data['2: train test split'][dx][split] = {}
        labels['2: train test split'][dx][split] = {}

    data['2: train test split'][dx]['full'] = data['1: load'][dx]
    labels['2: train test split'][dx]['full'] = labels['1.5: map'][dx]

    for mode in data['1: load'][dx]:

        (data['2: train test split'][dx]['train'][mode], 
        data['2: train test split'][dx]['test'][mode], 
        labels['2: train test split'][dx]['train'], 
        labels['2: train test split'][dx]['test']) = train_test_split(
        data['1: load'][dx][mode], labels['1.5: map'][dx], test_size=0.2, 
        random_state=47, stratify = labels['1.5: map'][dx])

labels['2: train test split']['cn_bl_to_mci'] = {}
labels['2: train test split']['cn_bl_to_mci']['full'] = labels['1.5: map']['cn_bl_to_mci']







#--------------- PART 3: Filtering out and plotting missing data ---------------#
print('|3/5| Filtering out and plotting missing data...')

missing_thresh = 0.34

# Dict to store data after missing data filtration
data['3: missing data filtration'] = {}

# Dict to plot missingness of each feature
missing_data_plot_dict = {}

# Filter missing based on training sets.
# This wont be done for CN patients who later have an MCI diagnosis
# as that is considered a test set for MCI patients - features will
# be filtered for that set depending on their presence in the MCI
# cohort
for dx in ['mci_bl', 'cn_bl']:
    data['3: missing data filtration'][dx] = {}
    missing_data_plot_dict[dx] = {}

    for split in ['train', 'full']:
        
        data['3: missing data filtration'][dx][split] = {}
        
        missing_data_plot_dict[dx][split] = {}

        sample_size = data['2: train test split'][dx][split]['other'].shape[0]
        rel_missing_thresh = sample_size * missing_thresh

        for mode in data['2: train test split'][dx][split]:
            
            missing_data_plot_dict[dx][split][mode] = {}
            temp_df = copy.deepcopy(data['2: train test split'][dx][split][mode])

            for feat in data['2: train test split'][dx][split][mode]:
                
                feat_missingness = data['2: train test split'][dx][split][mode][feat].isna().sum()
                missing_data_plot_dict[dx][split][mode][feat] = feat_missingness

                if feat_missingness > rel_missing_thresh:
                    temp_df.drop(feat, axis=1, inplace=True)

            data['3: missing data filtration'][dx][split][mode] = temp_df

# Select filtered columns from the training set to filter the respective testing sets
for dx in ['mci_bl', 'cn_bl']:

    data['3: missing data filtration'][dx]['test'] = {}

    for mode in data['3: missing data filtration'][dx]['train']:
            
        remaining_cols = list(data['3: missing data filtration'][dx]['train'][mode].keys())
        data['3: missing data filtration'][dx]['test'][mode] = data['2: train test split'][dx]['test'][mode][remaining_cols]

# Repeat for CN to MCI patients, using MCI training set
data['3: missing data filtration']['cn_bl_to_mci'] = {}
data['3: missing data filtration']['cn_bl_to_mci']['full'] = {}

for mode in data['3: missing data filtration']['mci_bl']['train']:
    remaining_cols = list(data['3: missing data filtration']['mci_bl']['train'][mode].keys())
    data['3: missing data filtration']['cn_bl_to_mci']['full'][mode] = data['1: load']['cn_bl_to_mci'][mode][remaining_cols]

# Plotting missing data for each modality
fig_save_path = 'output/figures'
for dx in ['cn_bl', 'mci_bl']:
    split = 'train'
    fig, axes = plt.subplots(1, 9, figsize=(25,3), dpi=600, sharey=True)
    for i, ax in enumerate(axes):
        keys = list(data['3: missing data filtration'][dx][split].keys())
        mode = keys[i]
        feats = missing_data_plot_dict[dx][split][mode]
        cats = list(feats.keys())
        vals = list(feats.values())

        sns.barplot(x = cats, y = vals, ax=ax, color='indianred')

        sample_size = data['3: missing data filtration'][dx][split][mode].shape[0]
        missing_line = sample_size*missing_thresh
        ax.hlines(y = missing_line, xmin=-1, xmax=len(cats), linestyle='--', color = 'firebrick')

        ax.set_xticks([])
        ax.set_title(mode.replace('_', ' '))
        ax.set_ylim([0, sample_size])

    plt.tight_layout()
    print(f'Saving missing data figure of {dx} to {fig_save_path}')
    plt.savefig(fig_save_path + f'/missing_feat_{dx}_{split}')








#--------------- PART 4: Filtering out empty modes ---------------#
print('|4/5| Filtering out empty modes...')
data['4: missing mode filtration'] = copy.deepcopy(data['3: missing data filtration'])
for dx in data['3: missing data filtration']:
    for split in data['3: missing data filtration'][dx]:
        for mode in data['3: missing data filtration'][dx][split]:
            current_mode = data['3: missing data filtration'][dx][split][mode]
            if current_mode.shape[1] == 0:
                del data['4: missing mode filtration'][dx][split][mode]








#--------------- PART 5: Final declaration of modes and saving ---------------#
print('|5/5| Final declaration of modes and saving...')

data['5: finalizing modes'] = copy.deepcopy(data['4: missing mode filtration'])

mri_mode_list = ['Volume (WM Parcellation)', 'Volume (Cortical Parcellation)',
                'Surface Area', 'Cortical Thickness Average', 
                'Cortical Thickness Standard Deviation']

for dx in data['5: finalizing modes']:
    for split in data['5: finalizing modes'][dx]:
        for new_mode in mri_mode_list:
            data['5: finalizing modes'][dx][split][f'MRI ROI: {new_mode}'] = data['5: finalizing modes'][dx][split]['mri_roi'].filter(like=new_mode)
        del data['5: finalizing modes'][dx][split]['mri_roi']

with open(f'{processed_path}/tadpole_data_train_test.pickle', 'wb') as file:
    pickle.dump(data['5: finalizing modes'], file)

with open(f'{processed_path}/tadpole_labels_train_test.pickle', 'wb') as file:
    pickle.dump(labels['2: train test split'], file)

print('Done!')