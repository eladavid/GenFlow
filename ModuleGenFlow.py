import operationUtils as op
import numpy as np
import sys
import time
from multiprocessing import Pool
MAX_INT = 1e6


class GenFlow:

    def __init__(self, n_gen_samples=1000, layer_list=None):
        # self.x_features_dim = x_features_dim
        # self.n_data_samples = n_data_samples
        self.n_gen_samples = n_gen_samples
        self.last_fitted_layer = -1
        self.init_seed = np.random.randint(MAX_INT)
        if layer_list is None:
            self.layers = []
        else:
            self.layers = layer_list

    def add_layer(self, layer):
        self.layers.append(layer)

    def get_last_exec_layer(self):
        return self.last_fitted_layer

    def get_num_of_layers(self):
        return len(self.layers)

    def get_layer_list(self):
        return self.layers

    def init_gen(self, n_samples=0, sigma=1.0, use_init_rng=0, is_rgb=0):
        if n_samples == 0:
            n_samples = self.n_gen_samples
        init_feature_dim = self.layers[0].get_dim()
        if use_init_rng == 1:
            rng = np.random.RandomState(self.init_seed)
        else:
            rng = np.random
        gen_input = sigma * rng.randn(init_feature_dim, n_samples)
        gen_input = op.reshape_as_pics(gen_input, is_rgb)
        return gen_input

    def fit(self, data_input, gen_input=None, start_layer=0, last_layer_idx=-1):
        if data_input.ndim == 4 and data_input.shape[3] == 3: # RGB data
            is_rgb = 1
        else:
            is_rgb = 0
        if last_layer_idx == -1:
            last_layer_idx = len(self.layers)
        if gen_input is None:
            gen_input = self.init_gen(use_init_rng=1, is_rgb=is_rgb)
        for i, layer in enumerate(self.layers):

            if i < start_layer:
                continue
            if i > last_layer_idx:
                break
            start_time = time.time()
            gen_input = layer.fit(data_input, gen_input)
            end_time = time.time()
            print('finished fit for Layer: {}, time elapsed: {:.3f}'.format(i, end_time - start_time))
        self.last_fitted_layer = last_layer_idx
        return gen_input

    def transform(self, data_input, n_samples, start_layer=0, last_layer_idx=-1):
        if data_input.ndim == 4 and data_input.shape[3] == 3: # RGB data
            is_rgb = 1
        else:
            is_rgb = 0
        if last_layer_idx == -1:
            last_layer_idx = len(self.layers)
        gen_samples = self.init_gen(n_samples, sigma=0.9, is_rgb=is_rgb)

        for i, layer in enumerate(self.layers):
            if i < start_layer:
                continue
            if i > last_layer_idx:
                break
            start_time = time.time()
            gen_samples = layer.transform(data_input, gen_samples)
            end_time = time.time()
            print('finished transforming at Layer: {}, time elapsed: {:.3f}'.format(i, end_time - start_time))
        return gen_samples

    def fit_transform(self, data_input, n_samples, last_layer_idx=-1):
        if data_input.ndim == 4 and data_input.shape[3] == 3: # RGB data
            is_rgb = 1
        else:
            is_rgb = 0
        if last_layer_idx == -1:
            last_layer_idx = len(self.layers)
        gen_fit_input = self.init_gen(is_rgb=is_rgb)
        gen_samples_input = self.init_gen(n_samples, is_rgb=is_rgb)
        for i, layer in enumerate(self.layers):
            if i > last_layer_idx:
                break
            start_time = time.time()
            gen_fit_input, gen_samples_input = layer.fit_transform(data_input, gen_fit_input, gen_samples_input)
            end_time = time.time()
            print('finished fit for Layer: {}, time elapsed: {:.3f}'.format(i, end_time - start_time))
        self.last_fitted_layer = last_layer_idx
        return gen_fit_input, gen_samples_input


class Layer:
    def __init__(self, features_dim, poly_deg, n_iter=200, patch_features_override=0, use_multiprocessing=0):
        self.features_dim = features_dim
        self.poly_deg = poly_deg
        if patch_features_override == 0:
            self.mapping_tensor = np.zeros((n_iter, poly_deg+1, features_dim))
            self.support_bounds = np.zeros((n_iter, 2, features_dim))
            #self.lin_map_tensor = np.zeros((n_iter, 2, features_dim))
        # use this override to enforce smaller tensors for conv layers
        else:
            self.mapping_tensor = np.zeros((n_iter, poly_deg + 1, patch_features_override))
            self.support_bounds = np.zeros((n_iter, 2, patch_features_override))
            #self.lin_map_tensor = np.zeros((n_iter, 2, patch_features_override))
        self.n_iter = n_iter
        self.proj_mat_list = np.random.randint(MAX_INT, size=(n_iter, 1))
        self.multiprocessing = use_multiprocessing
        #self.data_input = np.zeros((features_dim, n_samples))

    def get_proj_mat_list(self):
        return self.proj_mat_list

    def get_mapping_tensor(self):
        return self.mapping_tensor

    def fit(self, data_input, gen_input):
        # data_input = self.data_size_fit(data_input)
        # gen_last = self.data_size_fit(gen_input)
        features_dim = data_input.shape[0]
        gen_last = gen_input
        del gen_input
        if self.multiprocessing:
            my_pool = Pool(4)
        C = np.cov(data_input)
        gen_rot_trans = np.zeros(gen_last.shape)
        for i in range(self.n_iter):
            iter_score = 0
            W = op.gen_orthogonal_mat(data_input, C, self.proj_mat_list[i])
            gen_rot = W @ gen_last
            data_rot = W @ data_input
            if self.multiprocessing:
                my_pool = Pool(4)
                # setting input for multiprocesses
                args = [(data_axis, gen_rot[d], self.poly_deg) for d, data_axis in enumerate(data_rot)]
                # activating multiprocess calculation
                out_args = my_pool.map(op.fit_axis_and_apply_mapping_multiprocessing, args)
                # output assignment
                io_poly = [i[0] for i in out_args]
                gen = [i[1] for i in out_args]
                self.mapping_tensor[i, :, :] = np.array(io_poly).T
                gen_rot_trans = np.array(gen)
                gen_last = W.T @ gen_rot_trans
            else:
                for d in range(features_dim):
                    io_poly, axis_score, support_bounds, p_lin = \
                        op.fit_axis_mapping_func(data_rot[d, :], gen_rot[d, :], self.poly_deg)
                    self.mapping_tensor[i, :, d] = io_poly
                    self.support_bounds[i, :, d] = support_bounds
                    iter_score = iter_score + axis_score
                    #self.lin_map_tensor[i, :, d] = p_lin
                    gen_rot_trans[d, :] = \
                        op.apply_transformation_on_axis(gen_rot[d, :], io_poly, support_bounds)
                gen_last = W.T @ gen_rot_trans
                #print("iter: {}, score: {}".format(i, iter_score))
        if self.multiprocessing:
            my_pool.close()
            my_pool.join()
        return gen_last

    def transform(self, data_input, gen_input):
        # gen_last = self.data_size_fit(gen_input)
        features_dim = data_input.shape[0]
        gen_last = gen_input

        # exit if fit wasn't executed
        if np.all(data_input == 0):
            sys.exit("cannot transform before fit!")
        C = np.cov(data_input)
        gen_rot_trans = np.zeros(gen_last.shape)
        for i in range(self.n_iter):
            W = op.gen_orthogonal_mat(data_input, C, self.proj_mat_list[i])
            gen_rot = W @ gen_last
            for d in range(features_dim):
                io_poly = self.mapping_tensor[i, :, d]
                support_bounds = np.squeeze(self.support_bounds[i, :, d])
                #p_lin = np.squeeze(self.lin_map_tensor[i, :, d])
                gen_rot_trans[d, :] = \
                    op.apply_transformation_on_axis(gen_rot[d, :], io_poly, support_bounds, p_lin=None)

                #if np.any(np.isnan(gen_rot_trans[d, :])):
                #    print('iter: {}, feature: {}'.format(i, d))
                #    exit(1)

            gen_last = W.T @ gen_rot_trans
            #print(gen_last[0, :])
        return gen_last

    def fit_transform(self, data_input, gen_fit_input, gen_samples_input):
        # data_input = self.data_size_fit(data_input)
        # gen_fit_last = self.data_size_fit(gen_fit_input)
        # gen_samples_last = self.data_size_fit(gen_samples_input)
        features_dim = data_input.shape[0]
        gen_fit_last = gen_fit_input
        gen_samples_last = gen_samples_input
        C = np.cov(data_input)

        gen_fit_rot_trans = np.zeros(gen_fit_last.shape)
        gen_samples_rot_trans = np.zeros(gen_samples_last.shape)
        for i in range(self.n_iter):
            W = op.gen_orthogonal_mat(data_input, C, self.proj_mat_list[i])
            gen_fit_rot = W @ gen_fit_last
            gen_samples_rot = W @ gen_samples_last
            data_rot = W @ data_input
            for d in range(features_dim):
                io_poly, _, support_bounds, p_lin = \
                    op.fit_axis_mapping_func(data_rot[d, :], gen_fit_rot[d, :], self.poly_deg)
                self.mapping_tensor[i, :, d] = io_poly
                self.support_bounds[i, :, d] = support_bounds
                #self.lin_map_tensor[i, :, d] = p_lin
                gen_fit_rot_trans[d, :] = \
                    op.apply_transformation_on_axis(gen_fit_rot[d, :], io_poly, support_bounds, p_lin=None)
                gen_samples_rot_trans[d, :] = \
                    op.apply_transformation_on_axis(gen_samples_rot[d, :], io_poly, support_bounds, p_lin=None)
            gen_fit_last = W.T @ gen_fit_rot_trans
            gen_samples_last = W.T @ gen_samples_rot_trans
        return gen_fit_last, gen_samples_last

    def get_dim(self):
        return self.features_dim


class Linear(Layer):
    def __init__(self, features_dim, poly_deg, n_iter=200, use_multiprocessing=0):
        super(Linear, self).__init__(features_dim, poly_deg, n_iter, use_multiprocessing=use_multiprocessing)

    def data_size_fit(self, x_input):
        if x_input.ndim == 4 and x_input.shape[3] == 3:  # 3-channels (RGB data)
            pic_dim = int(np.sqrt(self.features_dim / 3))
            is_rgb = 1
        else:  # if data is single channel
            pic_dim = int(np.sqrt(self.features_dim))
            is_rgb = 0
        if pic_dim != x_input.shape[1]:
            x_resized = op.resize_dataset(x_input, pic_dim, is_rgb=is_rgb)
        else:
            x_resized = x_input
        x_resized = op.flatten_and_fit_dims(x_resized, is_rgb=is_rgb)
        return x_resized, is_rgb

    def fit(self, data_input, gen_input):
        # data resize
        data_input, is_rgb = self.data_size_fit(data_input)
        gen_input, _ = self.data_size_fit(gen_input)
        # fit
        gen_output = super(Linear, self).fit(data_input, gen_input)
        gen_output = op.reshape_as_pics(gen_output, is_rgb)
        return gen_output

    def transform(self, data_input, gen_input):
        # data resize
        data_input, is_rgb = self.data_size_fit(data_input)
        gen_input, _ = self.data_size_fit(gen_input)
        # transform
        gen_output = super(Linear, self).transform(data_input, gen_input)
        gen_output = op.reshape_as_pics(gen_output, is_rgb)

        return gen_output

    def fit_transform(self, data_input, gen_fit_input, gen_samples_input):
        # data resize
        data_input, is_rgb = self.data_size_fit(data_input)
        gen_fit_input, _ = self.data_size_fit(gen_fit_input)
        gen_samples_input, _ = self.data_size_fit(gen_samples_input)
        # fit_transform
        gen_fit_output, gen_samples_output = \
            super(Linear, self).fit_transform(data_input, gen_fit_input, gen_samples_input)
        gen_fit_output = op.reshape_as_pics(gen_fit_output, is_rgb)
        gen_samples_output = op.reshape_as_pics(gen_samples_output, is_rgb)
        return gen_fit_output, gen_samples_output


class Conv(Layer):
    def __init__(self, features_dim, kernel_dim, poly_deg, n_iter=200, is_rgb=0, use_multiprocessing=0):
        if not is_rgb:
            n_channels = 1
        else:
            n_channels = 3
        super(Conv, self).__init__(features_dim, poly_deg, n_iter, patch_features_override=int(n_channels*(kernel_dim ** 2)), use_multiprocessing=use_multiprocessing)
        self.kernel_dim = kernel_dim
        self.padding_factor = -1

    def data_size_fit(self, x_input):
        if x_input.ndim == 4 and x_input.shape[3] == 3:  # 3-channels (RGB data)
            pic_dim = int(np.sqrt(self.features_dim / 3))
            is_rgb = 1
        else:  # if data is single channel
            pic_dim = int(np.sqrt(self.features_dim))
            is_rgb = 0
        if pic_dim != x_input.shape[1]:
            x_resized = op.resize_dataset(x_input, pic_dim, is_rgb)
        else:
            x_resized = x_input
        x_resized = op.flatten_and_fit_dims(x_resized, is_rgb=is_rgb)
        x_resized_patches, padding_factor = op.split_to_patches(x_resized, self.kernel_dim, is_rgb=is_rgb)
        patches_shape = x_resized_patches.shape

        x_resized_patches = op.flatten_and_fit_dims(x_resized_patches, patches=1, is_rgb=is_rgb)

        self.padding_factor = padding_factor
        return x_resized_patches, patches_shape, is_rgb

    def fit(self, data_input, gen_input):
        # data resize
        data_input, _, is_rgb = self.data_size_fit(data_input)
        gen_input, patches_shape, _ = self.data_size_fit(gen_input)
        # fit
        gen_output = super(Conv, self).fit(data_input, gen_input)
        # data resize
        gen_output = gen_output.reshape(patches_shape)
        gen_output = op.reconstruct_from_patches(gen_output, self.kernel_dim, self.padding_factor, is_rgb=is_rgb)
        gen_output = op.reshape_as_pics(gen_output, is_rgb)
        return gen_output

    def transform(self, data_input, gen_input):
        # data resize
        data_input, _, is_rgb = self.data_size_fit(data_input)
        gen_input, patches_shape, _ = self.data_size_fit(gen_input)
        # transform
        gen_output = super(Conv, self).transform(data_input, gen_input)
        # data resize
        gen_output = gen_output.reshape(patches_shape)
        gen_output = op.reconstruct_from_patches(gen_output, self.kernel_dim, self.padding_factor, is_rgb=is_rgb)
        gen_output = op.reshape_as_pics(gen_output, is_rgb)
        return gen_output

    def fit_transform(self, data_input, gen_fit_input, gen_samples_input):
        # data resize
        data_input, _, is_rgb = self.data_size_fit(data_input)
        gen_fit_input, fit_patches_shape, _ = self.data_size_fit(gen_fit_input)
        gen_samples_input, samples_patches_shape, _ = self.data_size_fit(gen_samples_input)
        # fit_transform
        gen_fit_output, gen_samples_output = \
            super(Conv, self).fit_transform(data_input, gen_fit_input, gen_samples_input)
        # reshape to patches form
        gen_fit_output = gen_fit_output.reshape(fit_patches_shape)
        gen_samples_output = gen_samples_output.reshape(samples_patches_shape)
        # reshape to data form
        gen_fit_output = op.reconstruct_from_patches(gen_fit_output, self.kernel_dim, self.padding_factor, is_rgb=is_rgb)
        gen_samples_output = op.reconstruct_from_patches(gen_samples_output, self.kernel_dim, self.padding_factor, is_rgb=is_rgb)
        # reshape to pics form
        gen_fit_output = op.reshape_as_pics(gen_fit_output, is_rgb)
        gen_samples_output = op.reshape_as_pics(gen_samples_output, is_rgb)

        return gen_fit_output, gen_samples_output
