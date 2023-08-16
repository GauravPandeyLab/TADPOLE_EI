import pickle 
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
import copy
from src_utils import encode_exclude_nan, other_process, impute_per_mode, normalize_per_col

print('|---STEP 3: ENCODING / IMPUTING / NORMALIZING TADPOLE DATA---|')

data_file_path = 'data/processed/tadpole_data_train_test.pickle'
with open(data_file_path, 'rb') as file:
    tadpole_data = pickle.load(file)

label_file_path = 'data/processed/tadpole_labels_train_test.pickle'
with open(label_file_path, 'rb') as file:
    tadpole_labels = pickle.load(file)

process_tadpole_dict = copy.deepcopy(tadpole_data)

print('|1/2| Encoding categorical features')

# Encode / Onehot categorical features (only found in Others)
for dx in tadpole_data:
    for div in tadpole_data[dx]:
        for mode in tadpole_data[dx][div]:
            if mode == 'other':
                process_tadpole_dict[dx][div][mode] = other_process(process_tadpole_dict[dx][div][mode])

print(process_tadpole_dict.keys())
print('|2/2| Imputing and normalizing all features')

# Impute and normalize
for dx in tadpole_data:
    for div in tadpole_data[dx]:
        if div in ['full']:
                process_tadpole_dict[dx][div] = impute_per_mode(process_tadpole_dict[dx][div])
                process_tadpole_dict[dx][div] = normalize_per_col(process_tadpole_dict[dx][div])
        if dx != 'cn_bl_to_mci':
            if div in ['train']:
                    process_tadpole_dict[dx]['train'], process_tadpole_dict[dx]['test'] = impute_per_mode(process_tadpole_dict[dx]['train'],
                                                                                                        process_tadpole_dict[dx]['test'])
                    process_tadpole_dict[dx]['train'], process_tadpole_dict[dx]['test']= normalize_per_col(process_tadpole_dict[dx]['train'],
                                                                                                        process_tadpole_dict[dx]['test'])
                    
process_tadpole_dict['cn_bl_to_mci']['full']['other'][:,0] = 0
                
processed_data_path = 'data/processed'
with open(f'{processed_data_path}/tadpole_data_imptn_norm.pickle', 'wb') as file:
    pickle.dump(process_tadpole_dict, file)

with open(f'{processed_data_path}/tadpole_labels_imptn_norm.pickle', 'wb') as file:
    pickle.dump(tadpole_labels, file)

print('Done!')