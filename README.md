# Using Ensemble Integration for TADPOLE data
Ensemble Integration (EI) is a customizable pipeline for generating diverse ensembles of heterogeneous classifiers. The original python implementation of EI can be found in the following repository: https://github.com/GauravPandeyLab/eip. All EI-derived models were adapted from this implementation, using a fork of EI already included in this repository.

## Data and processing
The original data provided by TADPOLE can be read about here: https://tadpole.grand-challenge.org/Data/#Data. To access TADPOLE data, you will need to register with ADNI and apply for access to these data. Once access is granted, follow these steps to process these data as we have in this project:

### Step 1:
Download this repository and the TADPOLE data. 

### Step 2:
In addition to the files included in this repository, you will need to initialize the following file structure for the code to function: In the root directory of this project, create an empty folder called 'data'. Within, create two more folders called 'processed' and 'raw'. Place the TADPOLE data files 'TADPOLE_D1_D2.csv' and 'TADPOLE_D1_D2_Dict.csv' in the 'raw' folder. Create another folder in the root directory called 'output'. Within this folder, create four more folders called 'figures', 'interpretation', 'models' and 'tables'. 

### Step 3:
From the root directory, run the following python files (located in /src) in order: 

#### tadpole_process_baseline.py
Collects baseline data from each patient with an MCI baseline diagnosis.

#### tadpole_process_train_test.py
Splits all data into stratified training and test sets.

#### tadpole_process_imptn_norm.py
Imputes and normalizes all data

## Running EI on TADPOLE data
All details and code necessary to run EI on TADPOLE data are located in the EI_for_TADPOLE jupyter notebook.


