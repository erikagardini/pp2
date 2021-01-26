import numpy as np
from scipy.spatial import distance
from scipy.sparse import csgraph
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.spatial import distance_matrix
from matplotlib import cm
from new_principalpath import utilities
import os

class PrincipalPath:

    def __init__(self, NC, boundary_ids, mode="log", batch_size=20):
        self.init_path = None
        self.boundary_ids = boundary_ids
        self.NC = NC

        self.We = 0
        self.W_start = None
        self.W_end = None
        self.W_way = None

        self.mode = mode
        self.sigma = None
        self.batch_size = batch_size

        self.reg_term_2 = None
        self.reg_term_3 = None

        self.models = None

    #Algorithm steps 1 and 2
    def init(self, X, k = 10, filename=""):

        #Put the boundaries at the beginning and the end of the input matrix X
        X_ = np.delete(X, (self.boundary_ids[0], self.boundary_ids[1]), axis=0)
        initial = X[self.boundary_ids[0],:].reshape(1, X.shape[1])#reshape(1,2)
        final = X[self.boundary_ids[1], :].reshape(1, X.shape[1])
        X = np.concatenate((initial, X_, final))

        #Compute Dijkstra
        dst_mat = distance.cdist(X, X, 'sqeuclidean')
        if k != X.shape[1]:
            idxs = np.argsort(dst_mat, axis=1)[:,1:k+1]
            for i in range(dst_mat.shape[0]):
                for j in range(dst_mat.shape[1]):
                    if j not in idxs[i,:]:
                        dst_mat[i,j] = 100000

        [path_dst, path_pre] = csgraph.dijkstra(dst_mat, False, 0, True)
        path = np.ndarray(0, int)

        i = X.shape[0] - 1
        while i != 0:
            path = np.hstack([i, path])
            i = path_pre[i]
        path = np.hstack([i, path])

        #Plot Dijkstra path path
        path_to_plot = X[path, :]
        self._plotPath(X, path_to_plot, filename = filename + "path_init")

        self.init_path = X[path, :]
        #Waypoints adjustment
        if self.NC == self.init_path.shape[0]:
            self.We = self._getPathLength()
        else:
            self.init_path, self.We = self.movePath()

            path_to_plot = self.init_path[:,:]
            self._plotPath(X, path_to_plot, filename = filename + "path_updated")

        #Self tuning sigma
        self.sigma = self.setSigma(X)

    def _getPathLength(self):
        p1 = self.init_path[:-1,]
        p2 = self.init_path[1:,]
        dist = np.sqrt(np.sum((p2 - p1)**2, axis=1))
        p_len = np.sum(dist)
        if self.NC == 0:
            we = p_len / p1.shape[0]
        else:
            we = p_len / (self.NC + 1)
        return we

    #Adjust the waypoints
    def movePath(self):
        new_path = []
        new_path.append(self.init_path[0,:])

        We = self._getPathLength()
        way_to_do = We

        i = 0
        end_p = self.init_path[i+1,:]
        start_p = self.init_path[i,:]

        while len(new_path) != (self.NC+1):
            d = np.sqrt(np.sum((end_p - start_p) ** 2))
            if d > way_to_do:
                #Insert a new point after We
                lam = way_to_do / d
                new_point = (lam * end_p) + ((1 - lam) * start_p)
                new_path.append(new_point)
                #Computing the remaining segment
                remaining = d - way_to_do
                if round(remaining, 4) < round(We,4):
                    i = i + 1
                    way_to_do = We - remaining
                    end_p = self.init_path[i + 1, :]
                    start_p = self.init_path[i, :]
                else:
                    way_to_do = We
                    end_p = self.init_path[i + 1, :]
                    start_p = new_point
            elif d == way_to_do:
                new_path.append(self.init_path[i+1,:])
                i = i + 1
                end_p = self.init_path[i + 1, :]
                start_p = self.init_path[i, :]
            elif d < way_to_do:
                way_to_do = way_to_do - d
                i = i + 1
                end_p = self.init_path[i + 1, :]
                start_p = self.init_path[i, :]

        new_path.append(self.init_path[-1, :])

        return np.array(new_path), We

    #Self tuning sigma
    def setSigma(self, X, k=7):
        n = X.shape[0]
        nl = self.init_path.shape[0]
        dmax = max([np.linalg.norm(p1 - p2) for p1 in self.init_path for p2 in self.init_path])

        if self.mode == 'log':
            s = dmax / np.log(n)
        elif self.mode == 'sqrt':
            s = dmax / np.sqrt(nl)
        elif self.mode == 'self_tuning':
            dist = distance_matrix(self.init_path, X)
            ind = np.argsort(dist, axis=1)
            sigma_c = np.zeros(dist.shape[0], dtype='float64')
            for i in range(dist.shape[0]):
                sigma_c[i] = dist[i, ind[i, k]]

            b = np.tile(sigma_c, (n, 1))
            temp = tf.multiply(b, b)
            s = tf.multiply(2, temp)
        elif self.mode == 'auto':
            s = 0.001
        else:
            print("Mode: " + self.mode + " is invalid.")
            s = -2
            exit(-2)

        sigma = tf.Variable(s, dtype='float64', trainable=False)
        return sigma

    def optimize(self, X, s_span_1, s_span_2, l_rates, epochs, filename="", y_mode='length', criterion='elbow', plot=True, mode='scatter'):
        best_s_span_1 = []
        for lr in l_rates:
            for s1 in s_span_1:
                best_s_span_2 = []
                for s2 in s_span_2:

                    sub_dir = filename + "s1=" + str(s1) + "-s2=" + str(s2) + "-lr=" + str(lr) + "/"
                    if not os.path.exists(sub_dir):
                        os.mkdir(sub_dir)

                    # Waypoints optimization and model selection for the epochs
                    [_, best_pp, parameter] = self.fit(X, reg_term_2=s1, reg_term_3=s2,
                                                     lr=lr, epochs=epochs, filename=sub_dir,
                                                     y_mode=y_mode, criterion=criterion)

                    if plot:
                        utilities.plotPath(X, best_pp, sub_dir+"s1="+str(s1)+"_s2="+str(s2), mode=mode)
                    best_s_span_2.append(best_pp)
                    print("Waypoints optimized for for s1 = " + str(s1) + " and s2= " + str(s2) + ". Selected epoch: " + str(parameter) + ".\n")

                # Model selection for s_span_2
                [best_s2, parameter] = utilities.model_selection(np.array(best_s_span_2), s_span_2, X,
                                                                 filename + "s1=" + str(s1), y_mode=y_mode,
                                                                 criterion=criterion, plot=True)
                print("With s1 " + str(s1) + "fixed, the best s2 is " + str(parameter) + ".\n")
                if plot:
                    utilities.plotPath(X, best_s2, filename + "s1=" + str(s1), mode=mode)
                best_s_span_1.append(best_s2)

            # Model selection for s_span_1
            [final_best_path, parameter] = utilities.model_selection(np.array(best_s_span_1), s_span_1, X,
                                                                     filename + "final", y_mode=y_mode, criterion=criterion,
                                                                     plot=True)
            if plot:
                utilities.plotPath(X, final_best_path, filename + "best_path_", mode=mode)

            print("The best s1 is " + str(parameter) + ".\n")
            return final_best_path

    #Algorithm step 3 and 4
    def fit(self, X, reg_term_2=1.0, reg_term_3=1.0, lr=0.01, epochs=20, filename="", y_mode='length', criterion='elbow', plot_proba=False):
        train_loss = []
        t1_loss = []
        t2_loss = []
        t3_loss = []

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.reg_term_2 = reg_term_2
        self.reg_term_3 = reg_term_3

        self.models = []

        self.W_start = tf.Variable(self.init_path[0, :], trainable=False)
        self.W_way = tf.Variable(self.init_path[1:-1, :])
        self.W_end = tf.Variable(self.init_path[-1, :], trainable=False)
        train_dataset = tf.data.Dataset.from_tensor_slices(X)
        train_dataset_batch = train_dataset.batch(self.batch_size)

        for epoch in range(epochs):
            # Training in one step:
            loss_value_train, t1_train, t2_train, t3_train = self._update(train_dataset_batch, optimizer)
            train_loss.append(loss_value_train)
            t1_loss.append(t1_train)
            t2_loss.append(t2_train)
            t3_loss.append(t3_train)

            self._plotPath(X, filename=filename + "path_epoch" + str(epoch) + "_s=" + str(reg_term_2) + "and" + str(reg_term_3))
            if plot_proba:
                self._plotProbability(X, filename=filename + str(epoch) + "_s=" + str(reg_term_2) + "and" + str(reg_term_3))
            self._store()

        #Plot loss curves
        plt.plot(range(epochs), train_loss, '-r')
        plt.savefig(filename + "train_loss.png")
        plt.close()

        plt.plot(range(epochs), t1_loss, '-b')
        plt.savefig(filename + "t1_loss.png")
        plt.close()

        plt.plot(range(epochs), t2_loss, '-m')
        plt.savefig(filename + "t2_loss.png")
        plt.close()

        plt.plot(range(epochs), t3_loss, '-g')
        plt.savefig(filename + "t3_loss.png")
        plt.close()

        [best_path, parameter] = utilities.model_selection(np.array(self.models), range(0, epochs), X,
                                              filename, y_mode=y_mode, criterion=criterion,
                                              r2=self.reg_term_2, r3=self.reg_term_3, plot=True)

        return [np.array(self.models), best_path, parameter]

    def _store(self):
        path = self.W_way.numpy()
        path_start = self.W_start.numpy().reshape(1, path.shape[1])
        path_end = self.W_end.numpy().reshape(1, path.shape[1])

        total_path = np.concatenate((path_start, path, path_end))

        self.models.append(total_path)

    def _plotPath(self, X, path=None, filename=""):
        if path is not None:
            path_to_plot = path
        else:
            path = self.W_way.numpy()
            path_start = self.W_start.numpy().reshape(1, X.shape[1])
            path_end = self.W_end.numpy().reshape(1, X.shape[1])
            path_to_plot = np.concatenate((path_start, path, path_end))

        if X.shape[1] < 3: #to distinguish between mnist or 2d examples
            utilities.plotPath(X, path_to_plot, filename=filename, mode="scatter")
        else:
            utilities.plotPath(X, path_to_plot, filename=filename, mode="figure")

    def _plotProbability(self, x_train, n_sample=100, filename=""):

        N = n_sample
        margin_x = ((np.max(x_train[:,0]) - np.min(x_train[:,0])) / 2)
        margin_y = ((np.max(x_train[:,1]) - np.min(x_train[:,1])) / 2)

        down_lim_x = np.min(x_train[:,0]) - margin_x
        up_lim_x = np.max(x_train[:, 0]) + margin_x

        down_lim_y = np.min(x_train[:,1]) - margin_y
        up_lim_y = np.max(x_train[:,1]) + margin_y

        x = np.linspace(-1.5, 1.5, N)
        y = np.linspace(-1.5, 1.5, N)

        X, Y = np.meshgrid(x, y)

        mesh2d = np.zeros((N * N, 2), dtype='float64')
        h = 0
        for i in range(N):
            for j in range(N):
                mesh2d[h, 0] = x[i]
                mesh2d[h, 1] = y[j]
                h = h + 1

        Z = np.zeros((N, N))
        h = 0

        if self.mode == "self_tuning":
            sigma = self.setSigma(mesh2d)
            current_batch = 0
            batch_size = 1

        for i in range(N):
            for j in range(N):
                if self.mode == "self_tuning":
                    Z[j, i] = self.potential(mesh2d[h].reshape(1,2), sigma, current_batch, batch_size).numpy()
                else:
                    Z[j, i] = self.potential(mesh2d[h].reshape(1, 2), self.sigma).numpy()
                h = h + 1

        #levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        contours = plt.contour(X, Y, Z, 10, colors='black')
        plt.clabel(contours, inline=True, fontsize=8)

        plt.contourf(X, Y, Z, 10, zdir='z', offset=-0.15, cmap=cm.viridis)
        plt.scatter(x_train[:,0], x_train[:,1], c='red')
        plt.savefig(filename + "_loss.png")
        plt.close()

    #RBF
    def _potential(self, X, sigma, current_batch=None, batch_size=None):
        if self.mode == "self_tuning":
            g_value = 1 / sigma
            g = tf.Variable(g_value, dtype='float64', trainable=False)

            start = current_batch * batch_size
            end = start + X.shape[0]

            dist = distance.cdist(X, self.W_start.numpy().reshape(1, X.shape[1]), 'sqeuclidean')
            d_init = tf.exp(dist * (-g[start:end, 0]))

            dist = distance.cdist(X, self.W_end.numpy().reshape(1, X.shape[1]), 'sqeuclidean')
            d_end = tf.exp(dist * (-g[start:end, -1]))

            #Distance matrix
            size_x = tf.shape(X)[0]
            size_y = tf.shape(self.W_way)[0]
            xx = tf.expand_dims(X, -1)
            xx = tf.tile(xx, tf.stack([1, 1, size_y]))

            yy = tf.expand_dims(self.W_way, -1)
            yy = tf.tile(yy, tf.stack([1, 1, size_x]))
            yy = tf.transpose(yy, perm=[2, 1, 0])

            diff = tf.subtract(xx, yy)
            square_diff = tf.square(diff)

            dist = tf.reduce_sum(square_diff, 1)
            d = tf.reduce_sum(tf.exp(dist * (-g[start:end,1:-1])))

            D_start = tf.reduce_sum(d_init)
            D_end = tf.reduce_sum(d_end)
            D = d#tf.reduce_sum(tf.exp(d))

            p = D_start + D + D_end

        else:
            g_value = 1 / (2 * tf.pow(sigma, 2))
            g = tf.Variable(g_value, dtype='float64', trainable=False)

            dist = distance.cdist(X, self.W_start.numpy().reshape(1, X.shape[1]), 'sqeuclidean')
            d_init = tf.exp(dist * (-g))

            dist = distance.cdist(X, self.W_end.numpy().reshape(1, X.shape[1]), 'sqeuclidean')
            d_end = tf.exp(dist * (-g))

            # Distance matrix
            size_x = tf.shape(X)[0]
            size_y = tf.shape(self.W_way)[0]
            xx = tf.expand_dims(X, -1)
            xx = tf.tile(xx, tf.stack([1, 1, size_y]))

            yy = tf.expand_dims(self.W_way, -1)
            yy = tf.tile(yy, tf.stack([1, 1, size_x]))
            yy = tf.transpose(yy, perm=[2, 1, 0])

            diff = tf.subtract(xx, yy)
            square_diff = tf.square(diff)
            dist = tf.reduce_sum(square_diff, 1)
            d = tf.reduce_sum(tf.exp(dist * (-g)))

            D_start = tf.reduce_sum(d_init)
            D_end = tf.reduce_sum(d_end)
            D = d

            p = D_start + D + D_end

        return p

    #Cost function
    def _lossFunction(self, X, current_batch):
        #RBF
        term1 = self._potential(X, self.sigma, current_batch=current_batch, batch_size=self.batch_size)

        #Equidistance
        Wi = tf.pow(tf.subtract(tf.reduce_sum(tf.pow(tf.subtract(self.W_way[0,:], self.W_start), 2)), tf.pow(self.We, 2)), 2)
        WiWj = tf.pow(tf.subtract(tf.reduce_sum(tf.pow(tf.subtract(self.W_way[1:,:], self.W_way[0:-1,:]), 2), axis=1), tf.pow(self.We, 2)), 2)
        Wj = tf.pow(tf.subtract(tf.reduce_sum(tf.pow(tf.subtract(self.W_end, self.W_way[-1, :]), 2)), tf.pow(self.We, 2)), 2)

        term2 = tf.multiply(Wi + tf.reduce_sum(WiWj) + Wj, self.reg_term_2)

        #Smooth
        Wi = tf.reduce_sum(tf.pow(tf.subtract(self.W_way[0, :], self.W_start), 2))
        WiWj = tf.reduce_sum(tf.pow(tf.subtract(self.W_way[1:, :], self.W_way[0:-1, :]), 2), axis=1)
        Wj = tf.reduce_sum(tf.pow(tf.subtract(self.W_end, self.W_way[-1, :]), 2))

        term3 = tf.multiply(Wi + tf.reduce_sum(WiWj) + Wj, self.reg_term_3)

        loss_value = -term1 + term2 + term3
        return loss_value, -term1, term2, term3

    #Compute the gradient
    def _grad(self, X, current_batch):
        trainable = []
        with tf.GradientTape() as tape:
            loss_value, t1, t2, t3 = self._lossFunction(X, current_batch)
            trainable.append(self.W_way)
        return loss_value, t1, t2, t3, tape.gradient(loss_value, trainable)

    #Update the loss after each batch
    def _update(self, X, optimizer):
        loss_avg = tf.keras.metrics.Mean()
        t1_avg = tf.keras.metrics.Mean()
        t2_avg = tf.keras.metrics.Mean()
        t3_avg = tf.keras.metrics.Mean()

        current_batch = 0
        for x in X:
            trainable = []
            loss_2, t1, t2, t3, grads = self._grad(x, current_batch)
            trainable.append(self.W_way)
            optimizer.apply_gradients(zip(grads, trainable))
            loss_avg.update_state(loss_2)
            t1_avg.update_state(t1)
            t2_avg.update_state(t2)
            t3_avg.update_state(t3)

            current_batch = current_batch + 1

        return loss_avg.result().numpy(), t1_avg.result().numpy(), t2_avg.result().numpy(), t3_avg.result().numpy()

