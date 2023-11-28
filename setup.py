import os
import shutil
import subprocess

# Setup file structure for data and some outputs
root_dir = os.getcwd()
data_dir = os.path.join(root_dir, 'data')
processed_dir = os.path.join(data_dir, 'processed')
raw_dir = os.path.join(data_dir, 'raw')
output_dir = os.path.join(root_dir, 'output')
figures_dir = os.path.join(output_dir, 'figures')
tables_dir = os.path.join(output_dir, 'tables')

folders_to_create = [data_dir, processed_dir, raw_dir, output_dir, figures_dir, tables_dir]

for folder in folders_to_create:
    if not os.path.exists(folder):
        os.makedirs(folder)




# Run the python files from src to process the TADPOLE data
python_files = ['tadpole_process_baseline.py', 'tadpole_process_train_test.py', 'tadpole_process_imptn_norm.py']

for python_file in python_files:
    file_path = os.path.join(os.path.join(root_dir, 'src'), python_file)
    if os.path.exists(file_path):
        print(f"Running {python_file}...")
        subprocess.run(['python', python_file])

print("Processing complete")