import numpy as np
import pandas as pd
from functools import reduce
from scipy.spatial.distance import pdist, squareform

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors


class ManifoldAlignment(BaseEstimator):

    def __init__(self, k=None, data_positions=None, metric='euclidean',
                 domain_weights=None, fit_reset=True, partition=True):

        """

        :param k: number of neighbors to consider
        :param data_positions: position of features of each domain. list of position array per domain
        :param metric: metric to measure distance between points
        :param domain_weights: How much weight per domain. list of weights
        :param fit_reset: Whether you want to reset the estimator every time you fit. Recommended value: True
        :param partition: Whether to partition the manifold into subspace based on curvature or not
        """

        self.k = k

        self.data_positions = data_positions

        self.neigh = []

        self.metric = metric

        self.X = None

        self.L = []

        self.total_dims = None

        self.dims = None

        self.n_domains = None

        self.domain_weights = domain_weights

        self.partitioned_space = []

        self.fit_reset = fit_reset

        self.partition = partition

        if self.domain_weights is not None:

            if np.sum(self.domain_weights) != 1:

                self.domain_weights /= np.sum(self.domain_weights)

    @staticmethod
    def measure_curvature(G, D):

        V, U = np.linalg.eig(G)
        two_rw_G = reduce(np.matmul, (U, np.diag(V)**2, U.T))
        I_G = (G==0).astype(int)

        curvature = []

        for i in range(G.shape[1]):

            second_neighbors_curved_dist = two_rw_G[i, :]*I_G[i, :]

            second_neighbors = np.nonzero(second_neighbors_curved_dist)

            curv = np.sum(np.abs(second_neighbors_curved_dist[np.newaxis, second_neighbors] - D[i, second_neighbors]))

            curvature.append(curv)

        return curvature

    @staticmethod
    def measure_curvature_fast(G, D):

        """

        :param G: Data structure with neighbors per node
        :param D: Direct Distances between every node
        :return: curvature of every node
        """

        curvature = []

        for i, node in enumerate(G):

            neigh_distances = []

            direct_distances = []

            for neigh in node:

                base_length = D[i, neigh]

                second_neighs = G[neigh, :]

                second_neighs = second_neighs[np.in1d(second_neighs, node, invert=True)]

                min_second_neigh = np.argmin(D[neigh, second_neighs])

                second_length = D[neigh, min_second_neigh]

                neigh_distances.append(base_length+second_length)

                direct_distances.append(D[i, min_second_neigh])

            curvature.append(np.sum(np.abs(np.array(neigh_distances) - np.array(direct_distances))))

        return curvature

    def partition_space(self, NN, curvature, avg_curvature, std_curvature):

        NN_ = []

        if not isinstance(NN[0], set):

            for row in range(NN.shape[0]):

                NN_.append(set(NN[row]))

        partitioned_space = []

        not_curved_group = set()

        queue = [0]
        # queue.extend(list(NN_[0]))

        visited = set()

        not_visited = set(np.arange(len(NN)))

        non_curvature_nodes = set()

        parent_is_partitioning = [False]

        partitioned_space, not_curved_group, _ = \
            self.partition_subspace(NN_, queue, curvature, non_curvature_nodes, avg_curvature, std_curvature, not_curved_group,
                                    partitioned_space, visited, parent_is_partitioning, not_visited)

        for i in range(len(partitioned_space)):

            partitioned_space[i] = set(partitioned_space[i])

        partitioned_space.append(set(not_curved_group))

        return partitioned_space

    def partition_subspace(self, NN, queue, curvature, non_curvature_nodes, avg_curvature, std_curvature, not_curved_group,
                           partitioned_space, visited, parent_is_partitioning, not_visited):

        i = queue[0]

        queue = queue[1:]

        if i in visited:
            return partitioned_space, not_curved_group, queue

        else:

            partitioning = curvature[i] >= avg_curvature + 1 * std_curvature

            visited.add(i)

            not_visited.remove(i)

            queue_n = list(NN[i])

            if partitioning:

                if parent_is_partitioning[0]:

                    partitioned_space[-1].append(i)

                else:

                    partitioned_space.append([i])

                partitioned_space[-1].extend(queue_n)

            else:

                if not parent_is_partitioning[0]:

                    not_curved_group.add(i)

            parent_is_partitioning = parent_is_partitioning[1:]

            if len(queue_n) == 0:

                return partitioned_space, not_curved_group, queue

            else:

                for n_ in queue_n:

                    # NN[n_].discard(i)

                    if n_ not in visited and n_ not in queue:

                        queue.append(n_)
                        parent_is_partitioning.extend([partitioning])

                while len(queue) > 0:

                    partitioned_space, not_curved_group, queue = \
                        self.partition_subspace(NN, queue, curvature, non_curvature_nodes, avg_curvature, std_curvature,
                                                not_curved_group, partitioned_space, visited, parent_is_partitioning, not_visited)

                if len(not_visited) > 0:

                    queue.append(next(iter(not_visited)))

                    partitioned_space, not_curved_group, queue = \
                        self.partition_subspace(NN, queue, curvature, non_curvature_nodes, avg_curvature, std_curvature,
                                                not_curved_group, partitioned_space, visited, [False], not_visited)

            return partitioned_space, not_curved_group, queue

    def verify_data_format(self, X):

        X_ = []

        if self.data_positions is None:

            if not isinstance(X, list):

                X_ = [X]

        else:

            X_ = [X[positions, :] for positions in self.data_positions]

        if self.X is not None:

            if len(X_) != self.n_domains:

                Exception('Data must comply with the fitted data dimensions')

        return X_

    def create_neighborhoods(self, X, metric='euclidean', **kwargs):

        neighbor_matrix = []

        if self.k is not None:

            for ind, x in enumerate(X):

                if len(self.neigh) < self.n_domains:

                    self.neigh.append(NearestNeighbors(self.k, metric=metric, **kwargs))

                    self.neigh[ind].fit(x.T)

                    neighbor_matrix_ = self.neigh[ind].kneighbors(x.T, n_neighbors=self.k+1, return_distance=False)

                    neighbor_matrix_ = neighbor_matrix_[:, 1:]

                else:

                    neighbor_matrix_ = self.neigh[ind].kneighbors(x.T, n_neighbors=self.k, return_distance=False)

                neighbor_matrix.append(neighbor_matrix_)

        else:

            pass  # TODO: write code for ball algorithm

        return neighbor_matrix

    @staticmethod
    def compute_xi_mapping(xi, neighbors_d_prime, L_xi_d_prime_d):

        K_i = np.matmul(L_xi_d_prime_d, neighbors_d_prime)

        K_i_inv = np.linalg.inv(np.matmul(K_i.T, K_i)+0.0000001)

        w_i = reduce(np.matmul, (K_i_inv, K_i.T, xi))

        mapped_xi = np.matmul(K_i, w_i)

        return mapped_xi

    def assign_subspace(self, nn, subset):

        for i, subspace in enumerate(self.partitioned_space[subset]):

            if nn in subspace:

                return i

    def transform_data(self, X, neighbors):

        if self.X is None:

            Exception('Model has not been fit to data yet')

        G = []

        if not self.partition:

            for x in X:

                G.append(np.linalg.inv(np.matmul(x, x.T)+0.00001))

        X_transformed = X.copy()

        if self.domain_weights is not None:
            # X_transformed *= self.domain_weights
            X_transformed = (np.array(X_transformed).T * self.domain_weights).T
        for i in range(self.n_domains):

            if len(self.L) < self.n_domains:

                self.L.append([])

            neighbors_i = neighbors[i]

            for j in range(self.n_domains):

                if i == j:

                    if len(self.L) < self.n_domains or len(self.L[i]) < self.n_domains:

                        self.L[i].append([])

                else:

                    if len(self.L[i]) < self.n_domains:

                        ############# TODO: Partition here the space and create multiple Ls

                        if self.partition:

                            self.L[i].append([])

                            for subspace in self.partitioned_space[i]:
                                G = np.linalg.inv(np.matmul(X[j][:, list(subspace)],
                                                            X[j][:, list(subspace)].T) + 0.00001)

                                self.L[i][-1].append(reduce(np.matmul, (X[i][:, list(subspace)],
                                                                        X[j][:, list(subspace)].T, G)))

                        else:

                            self.L[i].append(reduce(np.matmul, (X[i], X[j].T, G[j])))

                    for ind, xi in enumerate(X[i].T):

                        neighbors_d_prime = self.X[j][:, neighbors_i[ind]]

                        subspace_xi = self.assign_subspace(neighbors_i[ind][0], i)

                        ############## TODO: think how to integrate the multiple Ls here

                        if self.partition:

                            L_ = self.L[i][j][subspace_xi]

                        else:

                            L_ = self.L[i][j]

                        xi_transformed = self.compute_xi_mapping(xi, neighbors_d_prime, L_)

                        if self.domain_weights is None:

                            X_transformed[i][:, ind] = (1/(j+1))*(X_transformed[i][:, ind]*j + xi_transformed)

                        else:

                            X_transformed[i][:, ind] = X_transformed[i][:, ind] + xi_transformed \
                                                       * self.domain_weights[j]

        X_transformed = [x.T for x in X_transformed]

        return X_transformed

    def fit(self, X):

        if self.fit_reset:

            self.L = []
            self.partitioned_space = []
            self.neigh = []

        X_ = self.verify_data_format(X)

        self.X = X_

        n_domains = len(X_)

        self.n_domains = n_domains

        dims = []

        total_dims = 0

        for x in X_:

            dims.append(x.shape[0])

            total_dims += dims[-1]

        self.total_dims = total_dims

        self.dims = dims

        neighbors = self.create_neighborhoods(X_, self.metric)

        for i in range(len(X_)):

            
            distances = squareform(pdist(X_[i].T))  # probably this + nneighbors can be optimized

            curvature = self.measure_curvature_fast(neighbors[i], distances)

            mean_curvature = np.mean(curvature)

            std_curvature = np.std(curvature)

            partitioned_space = self.partition_space(neighbors[i], curvature, mean_curvature,
                                                     std_curvature)

            self.partitioned_space.append(partitioned_space)

        X_transformed = self.transform_data(X_, neighbors)

        return X_transformed

    def transform(self, X):

        X_ = self.verify_data_format(X)

        neighbors = self.create_neighborhoods(X_, self.metric)

        X_transformed = self.transform_data(X_, neighbors)

        return X_transformed
