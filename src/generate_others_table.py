import pickle 
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

data_file_path = 'data/processed/tadpole_data_train_test.pickle'
with open(data_file_path, 'rb') as file:
    tadpole_data = pickle.load(file)

label_file_path = 'data/processed/tadpole_labels_train_test.pickle'
with open(label_file_path, 'rb') as file:
    tadpole_labels = pickle.load(file)

for dx_set in ['mci_bl', 'cn_bl_to_mci']:
    table1_dict = {}
    other_dict = {}
    for div in tadpole_data[dx_set]:
        
        other_dict[div] = pd.concat([tadpole_labels[dx_set][div], tadpole_data[dx_set][div]['other']], axis=1)
        other_dict[div]['last_dx'] = other_dict[div]['last_dx'].map({'MCI': 'MCI', 'Dementia': 'Dementia',
                                                                        'MCI to Dementia': 'Dementia',
                                                                        'MCI to NL': 'MCI',
                                                                        'NL': 'MCI',
                                                                        'NL to MCI': 'MCI',
                                                                        'Dementia to MCI': 'MCI',
                                                                        'DEM': 'Dementia'})
    other_dict_by_dx = {}
    for div in other_dict:
        other_dict_by_dx[div] = {}
        other_dict_by_dx[div]['MCI'] = other_dict[div][other_dict[div]['last_dx'] == 'MCI']
        other_dict_by_dx[div]['Dementia'] = other_dict[div][other_dict[div]['last_dx'] == 'Dementia']
        other_dict_by_dx[div]['All'] = other_dict[div]

    cont_vars = ['AGE', 'PTEDUCAT', 'FDG']
    cat_vars = ['DX_bl', 'PTGENDER', 'PTETHCAT', 'PTRACCAT','PTMARRY', 'APOE4']

    var_name_map = {'AGE': 'Age at baseline', 'PTEDUCAT': 'Years of education', 'FDG': 'Average FDG-PET of angular, temporal, and posterior cingulate',
                    'DX_bl': 'Diagnosis at baseline', 'PTGENDER': "Sex", 'PTETHCAT': 'Ethnicity', 'PTRACCAT': 'Race',
                    'PTMARRY': 'Marital Status', 'APOE4': 'APOE4 Allele'}

    # Do for full first, then train and test
    for div in tadpole_data[dx_set]:
        table1_dict[div] = pd.DataFrame()
        temp_dict = {}

        data = other_dict_by_dx[div]
        temp_dict['All patients'] = pd.DataFrame({'All patients': [data['All'].shape[0], data['MCI'].shape[0], 
                                                                data['Dementia'].shape[0]]}).T
        temp_dict['All patients'].rename(columns = {0: 'Overall', 1: 'Stayed at MCI', 2: 'Progressed to dementia'}, inplace=True)

        for cov in cont_vars:
            cols = {}
            for dx in ['All', 'MCI', 'Dementia']:
                data = other_dict_by_dx[div][dx][cov].dropna()
                median = round(data.median(), 2)
                q1 = round(np.percentile(data, 25), 2)
                q3 = round(np.percentile(data, 75), 2)
                full_str = f'{median} ({q1}, {q3})'
                cols[dx] = full_str

            temp_dict[cov] = pd.DataFrame({var_name_map[cov]: [cols['All'], cols['MCI'], cols['Dementia']]}).T
            temp_dict[cov].rename(columns = {0: 'Overall', 1: 'Stayed at MCI', 2: 'Progressed to dementia'}, inplace=True)

        for cav in cat_vars:    
            cols = {}
            for cav_unq in other_dict_by_dx[div]['All'][cav].unique():
                for dx in ['All', 'MCI', 'Dementia']:
                    data = other_dict_by_dx[div][dx][cav]
                    data_unq = data[data == cav_unq]
                    n = data_unq.shape[0]
                    perc = round(100 * n/data.shape[0], 1)
                    
                    full_str = f'{n} ({perc}%)'
                    cols[dx] = full_str

                temp_dict[cav_unq] = pd.DataFrame({cav_unq: [cols['All'], cols['MCI'], cols['Dementia']]}).T
                temp_dict[cav_unq].rename(columns = {0: 'Overall', 1: 'Stayed at MCI', 2: 'Progressed to dementia'}, inplace=True)

        for mode in temp_dict:
            table1_dict[div] = pd.concat([table1_dict[div], temp_dict[mode]], axis=0)

        row_to_move = table1_dict[div].iloc[3]
        table1_dict[div].drop(table1_dict[div].index[3], inplace=True)
        table1_dict[div] = table1_dict[div].append(row_to_move)
        table1_dict[div].rename_axis('Variables')

    for div in table1_dict:
        table1_dict[div].to_csv(f'output/tables/others_table_{dx_set}_{div}.csv')