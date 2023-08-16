import pandas as pd
import numpy as np
import sys
sys.path.append('src/ei_python/')

from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from ei import EnsembleIntegration
from sklearn.metrics import roc_auc_score, precision_recall_curve, \
    matthews_corrcoef, precision_recall_fscore_support, make_scorer
import sklearn



def normalize_per_col(train_dict, test_dict=None):
    if type(test_dict) == type(None):
        mode_dict_norm= {}
        for m in train_dict:
            mode_dict_norm[m] = pd.DataFrame()
            for col in range(train_dict[m].shape[1]):
                #print(mode_dict_feature_filtered_coldrop_train[m])
                dfmax = train_dict[m][:, col].max()
                dfmin = train_dict[m][:, col].min()
                norm_col_train = pd.DataFrame((train_dict[m][:, col] - 
                                                    dfmin) * 1.0 / (dfmax - dfmin))
                mode_dict_norm[m] = pd.concat([mode_dict_norm[m], pd.DataFrame(norm_col_train)], axis=1)
            mode_dict_norm[m] = np.array(mode_dict_norm[m])
        return(mode_dict_norm)
    elif type(test_dict) != type(None):
        mode_dict_norm_train = {}
        mode_dict_norm_test = {}
        for m in train_dict:
            mode_dict_norm_train[m] = pd.DataFrame()
            mode_dict_norm_test[m] = pd.DataFrame()
            for col in range(train_dict[m].shape[1]):
                dfmax = train_dict[m][:, col].max()
                dfmin = train_dict[m][:, col].min()
                norm_col_train = pd.DataFrame((train_dict[m][:, col] - 
                                                    dfmin) * 1.0 / (dfmax+0.000000000000001 - dfmin))
                norm_col_test = pd.DataFrame((test_dict[m][:, col] - 
                                                    dfmin) * 1.0 / (dfmax+0.000000000000001 - dfmin))
                mode_dict_norm_train[m] = pd.concat([mode_dict_norm_train[m], pd.DataFrame(norm_col_train)], axis=1)
                mode_dict_norm_test[m] = pd.concat([mode_dict_norm_test[m], pd.DataFrame(norm_col_test)], axis=1)
            mode_dict_norm_test[m][mode_dict_norm_test[m] > 1] = 1
            mode_dict_norm_test[m][mode_dict_norm_test[m] < 0] = 1
            mode_dict_norm_train[m] = np.array(mode_dict_norm_train[m])
            mode_dict_norm_test[m] = np.array(mode_dict_norm_test[m])
        return(mode_dict_norm_train, mode_dict_norm_test)
    
def impute_per_mode(train_dict, test_dict=None):
    if type(test_dict) == type(None):
        imputed_dict = {}
        for mode in train_dict:
            imputer = KNNImputer(missing_values=np.nan)
            #print(original_dict[mode])
            imputer = imputer.fit(train_dict[mode])
            imputed_dict[mode] = imputer.transform(train_dict[mode])
        return imputed_dict
    elif type(test_dict) != type(None):
        imputed_train_dict = {}
        imputed_test_dict = {}
        
        for mode in train_dict:
            #print(mode)
            imputer = KNNImputer(missing_values=np.nan)
            #print(original_dict[mode])
            imputer = imputer.fit(train_dict[mode])
            imputed_train_dict[mode] = imputer.transform(train_dict[mode])
            imputed_test_dict[mode] = imputer.transform(test_dict[mode])
        return imputed_train_dict, imputed_test_dict
    
def encode_exclude_nan(mode):
 
    le = preprocessing.LabelEncoder()
    nanmap = mode.isnull()
    for c in mode.keys():
        le.fit(mode[c])
        mode[c] = le.transform(mode[c])
    mode[nanmap == True] = np.nan

    return mode

def other_process(other_mode):
    if 'APOE4' in other_mode.keys():
        one_hot = pd.get_dummies(other_mode, columns=['APOE4'])
    else:
        one_hot = other_mode
    encoded = encode_exclude_nan(one_hot)
    return encoded

def_base_predictors = {
        'NB': GaussianNB(),
        'LR': make_pipeline(StandardScaler(), LogisticRegression()),
        "SVM": make_pipeline(Normalizer(), SVC(kernel='poly', degree=1, probability=True)),
        "Perceptron": Perceptron(),
        'AdaBoost': AdaBoostClassifier(n_estimators=10),
        "DT": DecisionTreeClassifier(),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=10),
        "RF": RandomForestClassifier(n_estimators=100),
        "XGB": XGBClassifier(n_esimators=100), 
        'KNN': KNeighborsClassifier(n_neighbors=1),
    }

def_meta_models = {
                "RF": RandomForestClassifier(),
                "SVM": SVC(kernel='linear', probability=True, max_iter=1e7),
                "NB": GaussianNB(),
                "LR": LogisticRegression(),
                "ADA": AdaBoostClassifier(),
                "DT": DecisionTreeClassifier(),
                "GB": GradientBoostingClassifier(),
                "KNN": KNeighborsClassifier(),
                "XGB": XGBClassifier()
}

def EI_model_train_and_save(project_name, base_predictors=def_base_predictors, meta_models=def_meta_models, 
        mode_dict = None, y = None, train = False, single_mode=False, 
        mode_name='unimode', random_state=42, path=None, model_building=False, 
        k_outer=5,
        k_inner=5, 
        n_samples=1,  
        sampling_strategy="hybrid",
        sampling_aggregation = 'mean',
        meta_training = True):

    EI = EnsembleIntegration(base_predictors=base_predictors, 
                            meta_models=meta_models,  
                            k_outer=k_outer,
                            k_inner=k_inner, 
                            n_samples=n_samples,  
                            sampling_strategy=sampling_strategy,
                            sampling_aggregation=sampling_aggregation,
                            n_jobs=-1,
                            random_state=random_state,
                            parallel_backend="loky",
                            project_name=project_name,
                            #additional_ensemble_methods=["Mean", "CES"]
                            model_building = model_building
                            ) 

    if train == True:
        if single_mode == False:
            #print('Whole EI')
            for name, modality in mode_dict.items():
                EI.train_base(modality, y, base_predictors, modality=name)
                

        elif single_mode == True:
            EI.train_base(mode_dict, y, base_predictors, modality=mode_name)
                #print('Single mode')
                #print(project_name)
        #EI.save() 

        #EI = EnsembleIntegration().load(f"EI.{project_name}")  # load models from disk

        #EI.train_meta(meta_models=meta_models)  # train meta classifiers
    
        meta_models = {
            "AdaBoost": AdaBoostClassifier(),
            "DT": DecisionTreeClassifier(max_depth=5),
            "GradientBoosting": GradientBoostingClassifier(),
            "KNN": KNeighborsClassifier(n_neighbors=21),
            "LR": LogisticRegression(),
            "NB": GaussianNB(),
            "MLP": MLPClassifier(),
            "RF": RandomForestClassifier(),
            "SVM": LinearSVC(tol=1e-2, max_iter=10000),
            "XGB": XGBClassifier(use_label_encoder=False, eval_metric='error')
        }
        #EI.save() 
        #EI = EnsembleIntegration().load(f"EI.{project_name}")  # load models from disk
        if meta_training == True:
            EI.train_meta(meta_models=meta_models)  # train meta classifiers

        EI.save(path=path) 

    else:
        EI = EnsembleIntegration().load(f"EI.{project_name}")  # load models from disk
        
    return EI

def fmeasure_score(labels, predictions, thres=None, 
                    beta = 1.0, pos_label = 1, thres_same_cls = False):
    
    """
        Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
        Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In Introduction to Information Retrieval. Cambridge University Press.
    """
    np.seterr(divide='ignore', invalid='ignore')
    if pos_label == 0:
        labels = 1-np.array(labels)
        predictions = 1-np.array(predictions)
        # if not(thres is None):
        #     thres = 1-thres
    # else:


    if thres is None:  # calculate fmax here
        np.seterr(divide='ignore', invalid='ignore')
        precision, recall, threshold = sklearn.metrics.precision_recall_curve(labels, predictions,
                                                                            #   pos_label=pos_label
                                                                              )

        fs = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
        fmax_point = np.where(fs==np.nanmax(fs))[0]
        p_maxes = precision[fmax_point]
        r_maxes = recall[fmax_point]
        pr_diff = np.abs(p_maxes - r_maxes)
        balance_fmax_point = np.where(pr_diff == min(pr_diff))[0]
        p_max = p_maxes[balance_fmax_point[0]]
        r_max = r_maxes[balance_fmax_point[0]]
        opt_threshold = threshold[fmax_point][balance_fmax_point[0]]

        return {'F':np.nanmax(fs), 'thres':opt_threshold, 'P':p_max, 'R':r_max, 'PR-curve': [precision, recall]}

    else:  # calculate fmeasure for specific threshold
        binary_predictions = np.array(predictions)
        if thres_same_cls:
            binary_predictions[binary_predictions >= thres] = 1.0
            binary_predictions[binary_predictions < thres] = 0.0
        else:
            binary_predictions[binary_predictions > thres] = 1.0
            binary_predictions[binary_predictions <= thres] = 0.0
        precision, recall, fmeasure, _ = precision_recall_fscore_support(labels,
                                                                        binary_predictions, 
                                                                        average='binary',
                                                                        # pos_label=pos_label
                                                                        )
        return {'P':precision, 'R':recall, 'F':fmeasure}  