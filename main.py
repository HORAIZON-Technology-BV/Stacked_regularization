#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import os
import os.path as op
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, roc_curve
from utils.utils import *
from load_data import *
from dependent_stacking_updated.stacked_w_MM import StackedModel
from joblib import Parallel, delayed
import multiprocessing
import time

def stratified_shuffle_cv(X, y, preds, DataSettings, BaySettings, ModelSettings, 
                          rand_seed=0):
    """
    Stratified Shuffle Split with ExtraTreesClassifier and permutation feature importance.
    
    Parameters:
    - X: Feature matrix
    - y: Labels
    - preds: Dictionary to store predictions and scores
    - DataSettings: Data-related settings for the model
    - BaySettings: Settings for Bayesian optimization
    - ModelSettings: Model configuration settings
    - rand_seed: Random seed for reproducibility
    """
    
    # Extract model settings
    manifold_alignment = ModelSettings.get('manifold_alignment')
    man_ali_k = ModelSettings.get('man_ali_k')
    Perm_imp = ModelSettings.get('Perm_imp')
    n_jobs = ModelSettings.get('n_jobs')

    # Extract data settings
    data_positions = DataSettings.get('data_positions')
    feat_names_by_group = DataSettings.get('feat_names_by_group')
    datasets_names = DataSettings.get('datasets_names')
    stacked_architecture = DataSettings.get('stacked_architecture')
    top_k_features = DataSettings.get('top_k_features')

    # Initialize lists to store feature importances and other results
    top_feats_array = []
    y_preds = []
    layer_0_performances = []

    # Define Stratified Shuffle Split with 50 splits and test size of 0.2
    splitter = StratifiedShuffleSplit(n_splits=50, test_size=0.2, random_state=rand_seed)
    shuffle_n = 0
    start = time.time()
    time_Save = []

    # Select top features for each omics dataset using univariate selection
    selected_feats = []
    relative_data_positions_ = []
    select_k = [30, 60, 90, "all"]  # Specify top features for each dataset
    for i in range(len(data_positions)):
        b = SelectKBest(f_classif, k=select_k[i])
        b.fit_transform(X.iloc[:, data_positions[i]], y)
        relative_data_positions_.append(b.get_support(indices=True))
        selected_feats.append(b.get_support(indices=True))

    # Run Stratified Shuffle Split for each train-test split
    for train, test in splitter.split(X, y):
        shuffle_n += 1
        end = time.time()
        time_Save.append(end - start)
        it_left = 50 - shuffle_n
        print(preds['score_list']['auc'])
        print(f"{shuffle_n}/50 time to go: {(np.mean(time_Save) * it_left):.2f}s")
        start = time.time()

        # Prepare training and testing sets
        X_train, X_test = X.iloc[train, :].values, X.iloc[test, :].values
        y_train, y_test = y.iloc[train].values, y.iloc[test].values
        
        # Initialize and train stacked model
        stacked = StackedModel(stack_architecture=stacked_architecture, cross=False,
                               top_k_features=top_k_features, positions=relative_data_positions_,
                               group_names=datasets_names, feat_names=feat_names_by_group,
                               manifold_alignment=manifold_alignment, man_ali_k=man_ali_k)
        
        # Perform Bayesian optimization
        stacked.param_bayesian_search(X_train, y_train, BaySettings['bds'],
                                      max_iterations=BaySettings['max_iterations'],
                                      num_cores=BaySettings['num_cores'], 
                                      batch_size=BaySettings['batch_size'], verbose=True,
                                      acquisition_type=BaySettings['acquisition_type'],
                                      exact_feval=False, n_splits_bo=3, rand_seed=rand_seed)
        
        # Fit the model and make predictions
        stacked.fit(X_train, y_train)
        y_preds.append(stacked.predict(X_test)[0])

        # Store layer-0 predictions
        layer0_per = [model.predict(X_test[:, relative_data_positions_[ind]]) 
                      for ind, model in enumerate(stacked.layer_0)]
        layer_0_performances.append(layer0_per)

    # Calculate AUC score and ROC curve for predictions
    mean_fpr = np.linspace(0, 1, 100)
    score = roc_auc_score(y, y_preds)
    fpr, tpr, _ = roc_curve(y, y_preds, pos_label=1)
    preds['score_list']['list_tprs'].append(np.interp(mean_fpr, fpr, tpr))
    preds['score_list']['list_tprs'][-1][0] = 0.0
    preds['score_list']['auc'].append(score)

    # Store prediction results
    preds['preds'].append(y_preds)
    preds['layer0_preds'].append(layer_0_performances)
    preds['y_test'].append(y.values)

    return preds

def main(): 
    """
    Main function to load data, define model architecture, and execute cross-validation.
    """
    AnalyzeName = 'Stacked_model_without_mani'
    CPU = -1  # Use all CPUs
    
    # Load dataset and model configuration data
    dataset, y, datasets_names, data_positions, feat_names_by_group = load_multi_omics() 
    path = os.getcwd()

    # Define model structure for each layer
    layers = [1]
    for i in range(len(data_positions)):
        for layer in layers:
            exec(f'rf{i}_layer{layer} = ExtraTreesClassifier(n_estimators=100, random_state=0, n_jobs={CPU})')
    
    top_model = [ExtraTreesClassifier(n_estimators=50, random_state=0, n_jobs=-1)]
    layer1 = [eval(f'rf{i}_layer1') for i in range(len(data_positions))]
    stacked_architecture = [layer1, top_model]
    top_k_features = [15] * len(data_positions)

    # Set Bayesian optimization parameters
    bds = [{'name': 'n_estimators', 'type': 'discrete', 'domain': [50, 100, 200]},
           {'name': 'max_depth', 'type': 'discrete', 'domain': [2, 5]},
           {'name': 'min_samples_split', 'type': 'discrete', 'domain': [2, 5]},
           {'name': 'max_features', 'type': 'categorical', 'domain': [0], 'categories': ["auto"]}]
    
    bds1 = [bds for _ in range(len(data_positions))]
    bds_top_model = [{'name': 'n_estimators', 'type': 'discrete', 'domain': [10000]},
                     {'name': 'max_depth', 'type': 'discrete', 'domain': [2, 5]},
                     {'name': 'min_samples_split', 'type': 'discrete', 'domain': [3, 5]},
                     {'name': 'max_features', 'type': 'categorical', 'domain': [0], 'categories': [None]}]
    bds = [bds1, [bds_top_model]]

    num_cores = multiprocessing.cpu_count()
    batch_size = 1
    max_iterations = 5
    acquisition_type = 'EI_MCMC'
    
    BaySettings = {'num_cores': num_cores, 'batch_size': batch_size,
                   'max_iterations': max_iterations, 'acquisition_type': acquisition_type, 'bds': bds}

    DataSettings = {'data_positions': data_positions, 'feat_names_by_group': feat_names_by_group, 
                    'datasets_names': datasets_names, 'stacked_architecture': stacked_architecture,
                    'top_k_features': top_k_features}
     
    ModelSettings = {'manifold_alignment': False, 'man_ali_k': 5, 'Perm_imp': True, 'n_jobs': CPU}
    
    score_list_dict = {'auc': [], 'list_tprs': []}
    results_dict = {'score_list': deepcopy(score_list_dict), 'preds': [], 'layer0_preds': [],
                    'feat_imps_normal': [], 'feat_imps_cor': [], 'feat_imps_noncor': [],
                    'y_test': [], 'perm_test': []}
    
    # Define path for results
    now = datetime.now()
    results_path = op.join(path, 'output', now.strftime("%d_%m_%Y"), AnalyzeName)
    os.makedirs(results_path, exist_ok=True)
    
    # Perform stratified shuffle split cross-validation
    stratified_shuffle_cv(dataset, y, results_dict, DataSettings, BaySettings, ModelSettings, 
                          rand_seed=0)
    
if __name__ == "__main__":
    main()
