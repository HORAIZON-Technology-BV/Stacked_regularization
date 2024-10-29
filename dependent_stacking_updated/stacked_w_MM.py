import sys
import numpy as np
import pandas as pd
from functools import reduce

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, r2_score, f1_score
import GPy

from GPyOpt.methods import BayesianOptimization

from sklearn.base import BaseEstimator

from scipy.stats import spearmanr, pearsonr
from .manifold_alignment import ManifoldAlignment


class StackedModel(BaseEstimator):

    def __init__(self, stack_architecture=None, positions=None, cross=False,
                 top_k_features=None, corr='spearman', significance_value=0.05,
                 significance_factor=0.4, group_names=None, feat_names=None, corr_n_p_runs=None,
                 score_type=None, intermediate_verbose=False,
                 manifold_alignment=False, manifold_alignment_metric='braycurtis',
                 man_ali_weights=None, manifold_bds=None, man_partition=True,
                 man_ali_k=5, all_layers_loss=False):  # TODO: pass the gse with opt_c_nu_when_fit = False

        """
        Builds a stack with two base models and a third on top
                          X[pos1] X[pos2] ... X[pos_n]
                           |        |           |
                           v        v           v
        layer1           model_1  model_2     model_n
                           |       |            |
                           |       |            |
                           v       v            v
        layer2           model_1 model_2     model_m
                           |       |            |
                           v       v            v
        other layers              ...
                           |    |
                           |    |
                           v    v
                           Output

        :param stack_architecture: list of models for each layer in format: [[layer_1_model_1, layer_1_model_2],
        [layer2_model_1, layer_2_model_2,...], ...]
        :param positions: If you want to pass a single matrix and want to select subsets, pass here the column positions
         of the subsets. If you have different size datasets DO NOT pass anything here
        list with a length equal to the number of models in the first layer
        :param cross: If you want to feed the original data plus all the predictions in the first layer to the next,
        pass True. There must be at least two non-blender layers (the goal is to break independence of the
        models in the first layer)
        :param significance_value: When computing the inter group connections what significance value to use
        :param significance_factor: when weighting correlation by the p-value, how much of the original correlation
        value to consider when the p-value sits exactly at the significance value. That is, for p=significance_value:
        C = C*significance_factor
        :param group_names: list of group names
        :param feat_names: list of lists of len=to each group, with each feat name.
        ex: [[group1_feat1, group1_feat2,...], [group2_feat1, group2_feat2,...], ... ]
        :param corr_n_p_runs: number of runs to perform during calculation of correlation values' p-value
        :param manifold_alignment: Whether to use manifold alignment or not
        :param all_layers_loss: if True, then the loss of all the models contribute to the loss optimization
        """

        self.stack_size = len(stack_architecture)
        self.layer_size = []

        """Assign each layer to the overall model"""

        self.scores = []

        for layer_ind, layer in enumerate(stack_architecture):
            setattr(self, 'layer_{}'.format(layer_ind), layer)
            self.layer_size.append(len(stack_architecture[layer_ind]))
            self.scores.append([[] for _ in range(self.layer_size[layer_ind])])

        self.score_type = score_type

        self.n_inputs = len(stack_architecture[0])

        self.positions = positions

        self.cross = cross

        self.top_k_features = top_k_features

        self.corr = corr

        self.corr_n_p_runs = corr_n_p_runs

        self.significance_value = significance_value

        self.significance_factor = significance_factor

        self.k = np.log(significance_factor) / np.log(1 - significance_value)  # for p-value integrated correlation

        self.conns = None  # Placeholder for inter group feature connections

        if self.top_k_features is not None:
            self.top_features = []  # create placeholder for list of top features for each data subset
            self.feats_importances = []

        self.feat_names = feat_names

        self.group_names = group_names

        self.intermediate_verbose = intermediate_verbose

        self.splitters = {}

        self.all_layers_loss = all_layers_loss

        self.manifold_alignment = manifold_alignment

        self.manifold_alignment_metric = manifold_alignment_metric

        self.man_ali_weights = man_ali_weights

        self.manifold_bds = manifold_bds

        self.man_ali_k = man_ali_k

        if manifold_alignment:

            self.man_ali = ManifoldAlignment(k=man_ali_k, metric=self.manifold_alignment_metric,
                                             domain_weights=self.man_ali_weights,
                                             data_positions=self.positions,
                                             partition=man_partition)

        else:

            self.man_ali = None

    def restart_scores(self):

        for layer_ind, layer in enumerate(self.layer_size):
            self.scores.append([[] for _ in range(self.layer_size[layer_ind])])

    def assign_score_type(self, y):

        if isinstance(y, pd.DataFrame):

            y_true = y.values

        else:

            y_true = y

        if self.score_type is None:

            unique_y = np.unique(y_true)

            if len(np.unique(np.mod(unique_y, 1))) == 1:  # integer classes -> classification

                if len(unique_y) > 2:  # multi-class classification

                    self.score_type = 'f1_score'

                else:  # binary classification

                    self.score_type = 'roc_auc_score'

            else:  # regression

                self.score_type = 'r2_score'

    def score(self, y_true, y_pred):

        """
        Computes the score of the prediction, with user specification or default metrics according to
        the task otherwise
        """
        if self.score_type is None:
            self.assign_score_type(y_true)

        score_ = eval('{}(y_true, y_pred)'.format(self.score_type))

        return score_

    def select_data(self, X, input_number, sample_index=None):

        """Select subset of X required for this model"""

        '''Check if X is list of datasets or one single matrix and select columns'''

        if isinstance(X, list):

            X_ = X[input_number]

        else:

            if self.positions is not None:

                X_ = X[:, self.positions[input_number]]

            else:

                X_ = X.copy()

        '''If datasets have different size select the samples subset'''

        if sample_index is not None:  # for the different input size case

            X_ = self.select_rows_columns(X_, rows=sample_index)

        return X_

    @staticmethod
    def select_rows_columns(X, rows=None, cols=None):

        """Method for selecting rows and columns of a matrix X, abstracting whether its numpy or pandas dataframe,
        and if rows and cols are in the form of strings or indexes"""

        X_ = X.copy()

        if isinstance(X_, pd.DataFrame):
            if rows is not None:
                if isinstance(rows, pd.Index):
                    X_ = X_.loc[rows, :]
                else:
                    X_ = X_.iloc[rows, :]
            if cols is not None:
                if isinstance(rows, pd.Index):
                    X_ = X_.loc[:, cols]
                else:
                    X_ = X_.iloc[:, cols]
        else:
            if rows is not None:
                X_ = X_[rows, :]
            if cols is not None:
                X_ = X_[:, cols]

        return X_

    @staticmethod
    def check_input(X):

        """Checks input data type and if it has subsets of different sizes"""

        sample_index = None

        diff_size = False

        smaller_sample_index = None

        if isinstance(X, list):

            X_sizes = list(map(len, X))
            min_sample_size = np.min(X_sizes)
            smaller_sample_index = np.argmin(X_sizes)

            if len(np.unique(X_sizes)) > 1:
                diff_size = True
                min_X = X[smaller_sample_index]
                if isinstance(min_X, pd.DataFrame):
                    sample_index = min_X.index
                else:
                    sample_index = np.arange(min_sample_size)

        return sample_index, diff_size, smaller_sample_index

    def fit(self, X, y, compute_inter_conns=False):

        """
        Fit stacked model
        :param X: Training data
        :param y: Target train values
        :param compute_inter_conns: If you want to compute inter feature group connections(correlations) pass True here
        """

        preds = []

        self.feats_importances = []  # feature importances list with a list of importances per group
        self.top_features = []  # positions of top features. list with list of top positions per group
        self.restart_scores()

        '''Check input for datasets of different data size'''

        sample_index, diff_size, smaller_sample_index = self.check_input(X)

        ##############################################################################################################
        # Manifold Alignment
        ##############################################################################################################

        if self.manifold_alignment:

            if not isinstance(X, list):

                if self.positions is None:

                    Exception('If passing unified dataset, must pass the positions of each domain')

                else:

                    X = self.man_ali.fit(X.T)
                    # X = np.hstack(X)

            else:

                X = [xi.T for xi in X]

                X = self.man_ali.fit(X)

        #######################################################
        """First Layer Fit"""
        #######################################################

        for input_number in range(self.n_inputs):

            X_ = self.select_data(X, input_number)  # select subset of data for this model

            model = eval('self.layer_0[{}]'.format(input_number))

            if diff_size:

                y_ = y[input_number]

            else:

                y_ = y

            model.fit(X_, y_)

            """"Pass to the next layer only same length predictions"""

            if diff_size:

                model_prediction = model.predict(self.select_rows_columns(X_, rows=sample_index))

            else:

                model_prediction = model.predict(X_)

            preds.append(model_prediction)

            """Get first layer importances"""

            if self.top_k_features is not None:

                top_k = None

                if hasattr(model, 'feature_importances_'):

                    if isinstance(self.top_k_features, int):

                        top_k = self.top_k_features

                    elif isinstance(self.top_k_features, list):

                        top_k = self.top_k_features[input_number]

                    top_feats_pos = np.argsort(model.feature_importances_)[-top_k:]

                    self.top_features.append(top_feats_pos)  # top features positions

                    self.feats_importances.append(model.feature_importances_)  # feature importances

                else:

                    top_k = X_.shape[1]

                    top_feats_pos = np.arange(top_k)

                    self.top_features.append(top_feats_pos)

                    # TODO: For models w/out importances, consider doing RFE

            else:

                top_feats_pos = np.arange(X_.shape[1])

                self.top_features.append(top_feats_pos)  # top features positions

                self.feats_importances.append(model.feature_importances_)

        preds_prev = np.c_[preds].T

        '''From the first layer on, work only with the smallest y'''

        if diff_size:
            y_ = y[smaller_sample_index]
        else:
            y_ = y

        ##############################################################
        """Rest of stack fit"""
        ##############################################################

        layer_1_preds = []

        for layer_number in range(1, self.stack_size):

            layer = eval('self.layer_{}'.format(layer_number))

            preds = []

            for model_number, model in enumerate(layer):

                if np.logical_and(self.cross, layer_number is not self.stack_size - 1):

                    """If feeding previous layer prediction plus the original data to the next layer"""

                    X_ = self.select_data(X, model_number, sample_index=sample_index)

                    """Select only top features in Layer 1"""

                    if np.logical_and(hasattr(self, 'top_features'), layer_number == 1):
                        X_ = self.select_rows_columns(X_, cols=self.top_features[model_number])

                    preds_model = np.c_[X_, preds_prev]

                else:

                    preds_model = preds_prev

                model.fit(preds_model, y_)
                if layer_number is not self.stack_size - 1:
                    new_layer_model_pred = model.predict(preds_model)
                    preds.append(new_layer_model_pred)

                    '''If computing inter group connections, save layer 1 predictions'''

                    if compute_inter_conns and layer_number == 1:
                        layer_1_preds.append(new_layer_model_pred)

            if layer_number is not self.stack_size - 1:
                preds_prev = np.c_[preds].T

        ##############################################################
        '''Compute the inter-group connections'''
        ##############################################################

        if compute_inter_conns:

            columns = ['group_from', 'group_to', 'from', 'to', 'value']

            conns = pd.DataFrame(columns=columns)

            top_feat_names = [self.feat_names[group][top_k] for group, top_k in enumerate(self.top_features)]

            for group_1_ind in range(len(layer_1_preds) - 1):

                for group_2_ind in range(group_1_ind + 1, len(layer_1_preds)):

                    '''Select Top feats data, whether its different sized data or not'''

                    if diff_size:

                        top_positions_group_1 = self.top_features[group_1_ind]
                        top_positions_group_2 = self.top_features[group_2_ind]

                        X_1 = self.select_rows_columns(X[group_1_ind], rows=sample_index)
                        X_2 = self.select_rows_columns(X[group_2_ind], rows=sample_index)

                    else:

                        group_1_positions = self.positions[group_1_ind]

                        top_group_1 = self.top_features[group_1_ind]

                        top_positions_group_1 = group_1_positions[top_group_1]

                        group_2_positions = self.positions[group_2_ind]

                        top_group_2 = self.top_features[group_2_ind]

                        top_positions_group_2 = group_2_positions[top_group_2]

                        X_1 = X
                        X_2 = X

                    top_group_1_data = self.select_rows_columns(X_1, cols=top_positions_group_1)
                    top_group_2_data = self.select_rows_columns(X_2, cols=top_positions_group_2)

                    '''Compute the features MCs and create a dataframe with the connections'''

                    for group_1_feat_ind in range(top_group_1_data.shape[1]):

                        for group_2_feat_ind in range(top_group_2_data.shape[1]):

                            f1 = self.select_rows_columns(top_group_1_data, cols=group_1_feat_ind)
                            f2 = self.select_rows_columns(top_group_2_data, cols=group_2_feat_ind)
                            y1 = layer_1_preds[group_1_ind]
                            y2 = layer_1_preds[group_2_ind]

                            conn = self.connection_inter_features(f1=f1, f2=f2,
                                                                  y1=y1, y2=y2,
                                                                  n_runs=self.corr_n_p_runs, corr=self.corr,
                                                                  k=self.k)

                            if self.group_names is None:

                                group_from = str(group_1_ind)
                                group_to = str(group_2_ind)

                            else:

                                group_from = self.group_names[group_1_ind]
                                group_to = self.group_names[group_2_ind]

                            if self.feat_names is None:

                                from_feat = str(group_1_feat_ind)
                                to_feat = str(group_2_feat_ind)

                            else:

                                from_feat = top_feat_names[group_1_ind][group_1_feat_ind]
                                to_feat = top_feat_names[group_2_ind][group_2_feat_ind]

                            values = [group_from, group_to, from_feat, to_feat, conn]

                            conn_row = pd.DataFrame(dict({*zip(columns, values)}), index=[0])

                            conns = conns.append(conn_row, ignore_index=True)

            conns['value'] = pd.to_numeric(conns['value'])

            self.conns = conns

    def predict(self, X, y=None, save_intermediate_scores=False):

        """
        Predict target values from test data
        :param X: test data
        :param y: target values, unnecessary (just there for pipeline purposes)
        :param save_intermediate_scores: If True, will save the scores of the individual models in the stack
        :return: y_pred
        """

        preds = []
        y_pred = []

        '''Check for different sized input datasets'''

        _, diff_size, _ = self.check_input(X)

        if diff_size:
            Exception('Cannot predict with datasets of different size')

        ##############################################################################################################
        # Manifold Alignment
        ##############################################################################################################

        if self.manifold_alignment:

            if not isinstance(X, list):

                if self.positions is None:

                    Exception('If passing unified dataset, must pass the positions of each domain')

                else:

                    X = self.man_ali.transform(X.T)
                    # X = np.hstack(X)

            else:

                X = [xi.T for xi in X]

                X = self.man_ali.transform(X)

        #########################################################
        """First Layer Prediction"""
        #########################################################

        for input_number in range(self.n_inputs):
            X_ = self.select_data(X, input_number)

            model = eval('self.layer_0[{}]'.format(input_number))
            model_pred = model.predict(X_)
            preds.append(model_pred)

            if self.intermediate_verbose and y is not None:

                if isinstance(y, list):

                    y_ = y[input_number]

                else:

                    y_ = y

                score = self.score(y_, model_pred)
                if save_intermediate_scores or self.all_layers_loss:
                    self.scores[0][input_number].append(score)

                print('layer_0 model_{} {}:{}'.format(input_number, self.score_type, score))

        preds_prev = np.c_[preds].T

        #########################################################
        """Rest of stack Prediction"""
        #########################################################

        for layer_number in range(1, self.stack_size):

            layer = eval('self.layer_{}'.format(layer_number))

            preds = []

            for model_number, model in enumerate(layer):

                if np.logical_and(self.cross, layer_number is not self.stack_size - 1):

                    """If feeding previous layer prediction plus the original data to the next layer"""

                    X_ = self.select_data(X, model_number)

                    """Select only top features in Layer 1"""

                    if np.logical_and(hasattr(self, 'top_features'), layer_number == 1):
                        X_ = self.select_rows_columns(X_, cols=self.top_features[model_number])

                    preds_model = np.c_[X_, preds_prev]

                else:

                    preds_model = preds_prev

                model_pred = model.predict(preds_model)

                if self.intermediate_verbose and y is not None:

                    if isinstance(y, list):

                        y_ = y[model_number]

                    else:

                        y_ = y

                    score = self.score(y_, model_pred)

                    if save_intermediate_scores or self.all_layers_loss:
                        self.scores[layer_number][model_number].append(score)

                    print('layer_{} model_{} {}:{}'.format(layer_number, model_number, self.score_type, score))

                if layer_number is not self.stack_size - 1:
                    preds.append(model.predict(preds_model))
                else:
                    y_pred = model_pred
            if layer_number is not self.stack_size - 1:
                preds_prev = np.c_[preds].T

        return y_pred

    def assign_parameters(self, parameter_values, bds, n_params_per_model):

        """
        Assigns parameter values to each model from the BO bounds dictionary
        :param parameter_values: values of the parameters to assign
        :param bds: BO bounds dictionary with names of the bounds per model
        :param n_params_per_model: list with the number of parameters to assign per model
        :param categorical_params: If there are categorical params, this is a map between index and the param string
        :return:
        """

        param_index = 0  # the parameter index in the set of all parameters to set in the stacked model
        model_index = 0  # index of model in relation to the set of all models in the stacked model

        for layer in range(self.stack_size):  # layer number

            for model_number in range(self.layer_size[layer]):  # model number in this layer

                for parameter in range(n_params_per_model[model_index]):  # index of parameters for this model

                    param_value = parameter_values[param_index]

                    """"Manifold Alignment"""

                    if self.manifold_alignment:

                        if bds[param_index]['name'] == 'domain_weights':
                            pass

                    """GPyOpt converts discrete values to float, force them to be discrete if they should"""
                    if isinstance(param_value, float):
                        if param_value.is_integer():
                            param_value = int(param_value)

                    if bds[param_index]['type'] == 'categorical':

                        param_value = bds[param_index]['categories'][param_value]

                        if isinstance(param_value, str):
                            param_value = '\'' + param_value + '\''

                    exec('self.layer_{}[{}].{} = {}'.format(layer, model_number, bds[param_index]['name'], param_value))

                    param_index += 1

                model_index += 1

        if self.manifold_alignment:

            domain_weights = []

            for i, bd in enumerate(bds[param_index:]):

                if bd['name'] == 'domain_weights':
                    domain_weights.append(parameter_values[param_index + i])

                elif bd['name'] == 'manifold_alignment_metric':
                    self.manifold_alignment_metric = bd['categories'][int(parameter_values[param_index + i])]

                elif bd['name'] == 'man_ali_k':

                    self.man_ali_k = bd['categories'][int(parameter_values[param_index + i])]

            domain_weights /= np.sum(domain_weights)
            self.man_ali.domain_weights = domain_weights

    @staticmethod
    def stratified_split_data(X, y, n_splits=50, rand_seed=0):

        """ Function that retrieves a generator that yields train, test stratified splits and works with
        different sized sets"""

        train_size = 0.8

        skf = StratifiedShuffleSplit(n_splits=n_splits, random_state=rand_seed, train_size=train_size,
                                     test_size=1 - train_size)

        if isinstance(X, list):

            datasets_lens = [len(X[i]) for i in range(len(X))]

            n_subsets = len(datasets_lens)

            shortest_dataset_index = np.argmin(datasets_lens)

            shortest_dataset = X[shortest_dataset_index]

            min_index = shortest_dataset.index

            if isinstance(y, list):

                y_ = y[shortest_dataset_index]

            else:

                y_ = y.loc[min_index]  # X and y must have matching indexes for this to work

            for train, test in skf.split(shortest_dataset, y_):

                '''Create different sized X_train, X_test and y_train'''

                train_list = []

                for i in range(n_subsets):

                    if i == shortest_dataset_index:

                        train_list.append(min_index[train])

                    else:

                        train_ = np.r_[min_index[train], X[i].index[len(min_index):]]

                        train_list.append(train_)

                yield (train_list, min_index[test])

        else:

            for train, test in skf.split(X, y):
                yield (train, test)

    def param_bayesian_search(self, X, y, bds, score='roc_auc', max_iterations=100, n_splits_bo=3, rand_seed=0,
                              num_cores=7, batch_size=1, verbose=True, exact_feval=False, acquisition_type='EI_MCMC',
                              maximize=None, evaluator_type='local_penalization', bds_manifold=None,
                              all_layers_loss_w=None):

        """
        Perform Bayesian Optimization on models hyperparameters. The search will be on P(p1 ^ p2 ^ p3)
        WARNING: format of params must be a list with the following structure:

        [[[bds for layer 1, model 1], [bds for layer 1, model 2], ....]], expanding bds:

        [[[{'name': '$model_1_param1_name$', 'type': '$continuous/discrete$', 'domain': ($min_value$, $max_value$)},
        {'name': '$model_1_param2_name$', 'type': '$categorical$', 'domain': ($cat_1$, $cat_2$ ,... )}],
        [{'name': '$model_2_param1_name$', 'type': '$continuous/discrete$', 'domain': ($min_value$, $max_value$)}],
        ....]]

        :param X: Data to optimize params on
        :param y: target values
        :param bds: Boundaries for models' params' values
        :param score: 'roc_auc' or 'r2' for regression
        :param max_iterations: number of samples for BayesOpt to take when optimizing
        :param n_splits_bo: Number of splits to evaluate performance of each set of parameters' values
        :param rand_seed: Random seed
        :param exact_feval: whether the target values are exact or have noise.
        :param acquisition_type: acquisition function for BO: 'EI', 'EI_MCMC', 'MPI', 'MPI_MCMC', 'LCB' or 'LCB_MCMC'
        :param evaluator_type: determines the way the objective is evaluated (all methods are equivalent if the batch size is one)
        - 'sequential', sequential evaluations.
        - 'random', synchronous batch that selects the first element as in a sequential policy and the rest randomly.
        - 'local_penalization', batch method proposed in (Gonzalez et al. 2016).
        - 'thompson_sampling', batch method using Thompson sampling.
        :return:
        """

        '''Setup the acquisiton function parameters for GPyopt optimization'''

        model_type = 'GP'

        acquisition_type_split = acquisition_type.split('_')

        if len(acquisition_type_split) > 1:

            if acquisition_type_split[1] == 'MCMC':

                model_type = 'GP_MCMC'

            else:

                Exception('acquisition function not known')

        '''Join all boundaries into one single list'''

        n_params_per_model = []

        if len(bds) > 1:

            bds_joined = []

            for layer_bds in bds:

                for model_number, bds_ in enumerate(layer_bds):

                    n_params_per_model.append(len(bds_))

                    for dic in bds_:
                        bds_joined.append(dic)

            bds = bds_joined

        if self.manifold_alignment:

            if bds_manifold is None:
                bds_manifold = [{'name': 'domain_weights', 'type': 'continuous', 'domain': [0.1, 0.9]}
                                for _ in range(self.n_inputs)]
                bds_manifold.append({'name': 'manifold_alignment_metric', 'type': 'categorical', 'domain': np.arange(5),
                                     'categories': ['braycurtis', 'correlation', 'cosine', 'canberra',
                                                    'cityblock']})

            bds.extend(bds_manifold)
        GP_kernel = GPy.kern.Matern52(input_dim=len(bds), variance=0.1, lengthscale=0.1)

        #########################################################################################################

        # bds_joined will look like this = [
        #     {'name': 'layer_1_model_1_C', 'type': 'continuous', 'domain': min_C, max_C)},
        #     {'name': 'layer_1_model_1_nu', 'type': 'continuous', 'domain': (min_nu, max_nu)},
        #     {'name': 'layer_1_model_2_n_trees', 'type': 'continuous', 'domain': (min_v, max_v)}},
        #     {'name': 'layer_2_model_1_n_trees', 'type': 'continuous', 'domain': (min_v, max_v)}},
        #     {'name': 'layer_2_model_1_criterion', 'type': 'categorical', 'domain': ('gini', 'entropy'), ... ]}}

        #########################################################################################################

        def cv_score(parameters):

            parameters = parameters[0]

            skf = StratifiedShuffleSplit(n_splits_bo, train_size=0.85, random_state=rand_seed)

            scores = []

            for train, test in self.stratified_split_data(X, y, n_splits=n_splits_bo, rand_seed=rand_seed):

                #print('\n OVERLAP?: {}'.format(len(np.argwhere(np.in1d(test, train)).flatten())))

                if isinstance(X, list):

                    X_train = [X[i].loc[train[i], :] for i in range(len(X))]
                    X_test = [X[i].loc[test, :] for i in range(len(X))]
                    y_train = [y[i].loc[train[i]] for i in range(len(X))]
                    y_test = y[0].loc[test]

                else:

                    X_train, X_test = X[train, :], X[test, :]
                    y_train, y_test = y[train], y[test]

                """Assign each parameter of each model its value"""

                self.assign_parameters(parameters, bds, n_params_per_model)

                """Fit the model and predict"""

                self.fit(X_train, y_train)

                y_pred = self.predict(X_test, y_test)

                """Record score"""

                score_ = self.score(y_test, y_pred)

                if self.all_layers_loss:

                    int_scores = []

                    for layer_number in range(len(self.layer_size)):

                        for m_n in range(self.layer_size[layer_number]):

                            if all_layers_loss_w is None:

                                w = 1

                            else:

                                w = all_layers_loss_w[layer_number][m_n]

                            int_scores.append(np.mean(self.scores[layer_number][m_n]) * w)

                    score_ = score_ * 0.7 + 0.3 * np.sum(int_scores)

                scores.append(score_)

            mean_score = np.mean(scores)

            if verbose:
                print('MEAN SCORE:{}'.format(mean_score))

            return mean_score

        if maximize is None:  # If user has not specified whether maximizing or minimizing target function, try to infer

            if self.score_type is None:

                if isinstance(y, list):
                    y_ = y[0]
                else:
                    y_ = y

                self.assign_score_type(y_)

            if np.isin('score', self.score_type.split('_')):

                maximize = True

            elif np.isin('loss', self.score_type.split('_')):

                maximize = False

            else:

                Exception('Please specify if maximizing or minimizing target function')

        optimizer = BayesianOptimization(f=cv_score,
                                         domain=bds,
                                         model_type=model_type,
                                         kernel=GP_kernel,
                                         acquisition_type=acquisition_type,
                                         evaluator_type=evaluator_type,
                                         acquisition_jitter=0.01,
                                         exact_feval=exact_feval,
                                         random_state=rand_seed,
                                         maximize=maximize,
                                         batch_size=batch_size,
                                         num_cores=num_cores,
                                         verbose = True)

        optimizer.run_optimization(max_iter=max_iterations)

        best_params = optimizer.x_opt

        params_names = []

        for bd in bds:
            params_names.append(bd['name'])

        """Update all models params to the best value"""

        self.assign_parameters(best_params, bds, n_params_per_model)

        return zip(params_names, best_params)

    @staticmethod
    def map_corr_to_method(corr):

        if corr.lower() == 'spearman':

            correlation = 'spearmanr'

        else:

            correlation = 'pearsonr'

        return correlation

    @staticmethod
    def double_correlation(f1, f2, y1, y2, corr='spearman'):

        """Computes the Multiple correlation between two features and a prediction, and averages for two predictions"""

        correlation = StackedModel.map_corr_to_method(corr)

        c_y1 = np.array([eval(correlation)(f1, y1)[0], eval(correlation)(f2, y1)[0]])

        c_y1 = np.reshape(c_y1, (len(c_y1),))

        c_y2 = np.array([eval(correlation)(f1, y2)[0], eval(correlation)(f2, y2)[0]])

        c_y2 = np.reshape(c_y2, (len(c_y2),))

        f1_intra_corr = eval(correlation)(f1, f1)[0]
        f2_intra_corr = eval(correlation)(f2, f2)[0]
        f1_f2_corr = eval(correlation)(f1, f2)[0]

        R = np.linalg.inv(np.array([[f1_intra_corr, f1_f2_corr], [f1_f2_corr, f2_intra_corr]]))

        C_y1 = reduce(np.matmul, [c_y1.T, R, c_y1])

        C_y2 = reduce(np.matmul, [c_y2.T, R, c_y2])

        return (C_y1 + C_y2) / 2

    @staticmethod
    def p_double_corr(f1, f2, y1, y2, C_y12, n_runs=None, corr='Spearman', rand_seed=0):

        """Computes the empirical p-value of the Multiple Correlation coefficient C_y12"""

        r = np.random.RandomState(rand_seed)

        if n_runs is None:
            n_runs = np.max([len(y1) + len(y2), 10])

        counter = 0

        for i in range(n_runs):

            y1_shuffled = y1.copy()

            r.shuffle(y1_shuffled)

            y2_shuffled = y2.copy()

            r.shuffle(y2_shuffled)

            C_y12_prime = StackedModel.double_correlation(f1, f2, y1_shuffled, y2_shuffled, corr)

            if np.abs(C_y12_prime) >= np.abs(C_y12):
                counter += 1

        return counter / n_runs

    @staticmethod
    def connection_inter_features(f1, f2, y1, y2, n_runs=None, corr='spearman',
                                  significance_value=0.05, significance_factor=0.4, k=None,
                                  rand_seed=0):

        """Computes Multiple correlation coefficient between two features and the prediction, weighted by the p-value"""

        C_y12 = StackedModel.double_correlation(f1, f2, y1, y2, corr)

        p_C_y12 = StackedModel.p_double_corr(f1, f2, y1, y2, C_y12, n_runs, corr, rand_seed=rand_seed)

        if k is None:
            k = np.log(significance_factor) / np.log(1 - significance_value)

        return C_y12 * (1 - p_C_y12) ** k

    def create_connections_dataframe(self, group_names=None, feat_names=None):

        """Creates a dataframe with the structure required to plot a hierarchical edge bundle plot"""

        if group_names is not None and feat_names is not None:

            group_from = []
            group_to = []
            from_feat = []
            to_feat = []

            for i in range(self.conns.shape(0)):
                group_from_i = self.conns['group_from'].iloc[i]
                group_from.append(group_names[group_from_i])
                group_to_i = self.conns['group_to'].iloc[i]
                group_to.append(group_names[group_to_i])
                from_feat_i = self.conns['from'].iloc[i]
                from_feat.append(feat_names[group_from_i][from_feat_i])
                to_feat_i = self.conns['to'].iloc[i]
                to_feat.append(feat_names[group_to_i][to_feat_i])

            self.conns['group_from'] = group_from
            self.conns['group_to'] = group_to
            self.conns['from'] = from_feat
            self.conns['to'] = to_feat

        conns_dataframe = self.conns

        return conns_dataframe