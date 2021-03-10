import operationUtils as op
import numpy as np
import sys

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

    def init_gen(self, n_samples=0, sigma=1.0, use_init_rng=0):
        if n_samples == 0:
            n_samples = self.n_gen_samples
        init_feature_dim = self.layers[0].get_dim()
        if use_init_rng == 1:
            rng = np.random.RandomState(self.init_seed)
        else:
            rng = np.random
        gen_input = sigma * rng.randn(init_feature_dim, n_samples)
        gen_input = op.reshape_as_pics(gen_input)
        return gen_input

    def fit(self, data_input, gen_input=None, start_layer=0, last_layer_idx=-1):
        if last_layer_idx == -1:
            last_layer_idx = len(self.layers)
        if gen_input is None:
            gen_input = self.init_gen(use_init_rng=1)
        for i, layer in enumerate(self.layers):
            if i < start_layer:
                continue
            if i > last_layer_idx:
                break
            gen_input = layer.fit(data_input, gen_input)
            print('finished fit for Layer: {}'.format(i))
        self.last_fitted_layer = last_layer_idx
        return gen_input

    def transform(self, data_input, n_samples, start_layer=0, last_layer_idx=-1):
        if last_layer_idx == -1:
            last_layer_idx = len(self.layers)
        gen_samples = self.init_gen(n_samples, sigma=0.9)

        for i, layer in enumerate(self.layers):
            if i < start_layer:
                continue
            if i > last_layer_idx:
                break
            gen_samples = layer.transform(data_input, gen_samples)
            print('finished transforming at Layer: {}'.format(i))
        return gen_samples

    def fit_transform(self, data_input, n_samples, last_layer_idx=-1):
        if last_layer_idx == -1:
            last_layer_idx = len(self.layers)
        gen_fit_input = self.init_gen()
        gen_samples_input = self.init_gen(n_samples)
        for i, layer in enumerate(self.layers):
            if i > last_layer_idx:
                break
            gen_fit_input, gen_samples_input = layer.fit_transform(data_input, gen_fit_input, gen_samples_input)
            print('finished fit for Layer: {}'.format(i))
        self.last_fitted_layer = last_layer_idx
        return gen_fit_input, gen_samples_input


class Layer:
    def __init__(self, features_dim, poly_deg, n_iter=200, patch_features_override=0):
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
        C = np.cov(data_input)
        gen_rot_trans = np.zeros(gen_last.shape)
        for i in range(self.n_iter):
            W = op.gen_orthogonal_mat(data_input, C, self.proj_mat_list[i])
            gen_rot = W @ gen_last
            data_rot = W @ data_input
            for d in range(features_dim):
                io_poly, _, support_bounds, p_lin = \
                    op.fit_axis_mapping_func(data_rot[d, :], gen_rot[d, :], self.poly_deg)
                self.mapping_tensor[i, :, d] = io_poly
                self.support_bounds[i, :, d] = support_bounds
                #self.lin_map_tensor[i, :, d] = p_lin
                gen_rot_trans[d, :] = \
                    op.apply_transformation_on_axis(gen_rot[d, :], io_poly, support_bounds)
            gen_last = W.T @ gen_rot_trans
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
    def __init__(self, features_dim, poly_deg, n_iter=200):
        super(Linear, self).__init__(features_dim, poly_deg, n_iter)

    def data_size_fit(self, x_input):
        pic_dim = int(np.sqrt(self.features_dim))
        if pic_dim != x_input.shape[1]:
            x_resized = op.resize_dataset(x_input, pic_dim)
        else:
            x_resized = x_input
        x_resized = op.flatten_and_fit_dims(x_resized)
        return x_resized

    def fit(self, data_input, gen_input):
        # data resize
        data_input = self.data_size_fit(data_input)
        gen_input = self.data_size_fit(gen_input)
        # fit
        gen_output = super(Linear, self).fit(data_input, gen_input)
        gen_output = op.reshape_as_pics(gen_output)
        return gen_output

    def transform(self, data_input, gen_input):
        # data resize
        data_input = self.data_size_fit(data_input)
        gen_input = self.data_size_fit(gen_input)
        # transform
        gen_output = super(Linear, self).transform(data_input, gen_input)
        gen_output = op.reshape_as_pics(gen_output)

        return gen_output

    def fit_transform(self, data_input, gen_fit_input, gen_samples_input):
        # data resize
        data_input = self.data_size_fit(data_input)
        gen_fit_input = self.data_size_fit(gen_fit_input)
        gen_samples_input = self.data_size_fit(gen_samples_input)
        # fit_transform
        gen_fit_output, gen_samples_output = \
            super(Linear, self).fit_transform(data_input, gen_fit_input, gen_samples_input)
        gen_fit_output = op.reshape_as_pics(gen_fit_output)
        gen_samples_output = op.reshape_as_pics(gen_samples_output)
        return gen_fit_output, gen_samples_output


class Conv(Layer):
    def __init__(self, features_dim, kernel_dim, poly_deg, n_iter=200):
        super(Conv, self).__init__(features_dim, poly_deg, n_iter, patch_features_override=int(kernel_dim**2))
        self.kernel_dim = kernel_dim
        self.padding_factor = -1

    def data_size_fit(self, x_input):
        pic_dim = int(np.sqrt(self.features_dim))
        if pic_dim != x_input.shape[1]:
            x_resized = op.resize_dataset(x_input, pic_dim)
        else:
            x_resized = x_input
        x_resized = op.flatten_and_fit_dims(x_resized)
        x_resized_patches, padding_factor = op.split_to_patches(x_resized, self.kernel_dim)
        patches_shape = x_resized_patches.shape

        x_resized_patches = op.flatten_and_fit_dims(x_resized_patches, patches=1)

        self.padding_factor = padding_factor
        return x_resized_patches, patches_shape

    def fit(self, data_input, gen_input):
        # data resize
        data_input, _ = self.data_size_fit(data_input)
        gen_input, patches_shape = self.data_size_fit(gen_input)
        # fit
        gen_output = super(Conv, self).fit(data_input, gen_input)
        # data resize
        gen_output = gen_output.reshape(patches_shape)
        gen_output = op.reconstruct_from_patches(gen_output, self.kernel_dim, self.padding_factor)
        gen_output = op.reshape_as_pics(gen_output)
        return gen_output

    def transform(self, data_input, gen_input):
        # data resize
        data_input, _ = self.data_size_fit(data_input)
        gen_input, patches_shape = self.data_size_fit(gen_input)
        # transform
        gen_output = super(Conv, self).transform(data_input, gen_input)
        # data resize
        gen_output = gen_output.reshape(patches_shape)
        gen_output = op.reconstruct_from_patches(gen_output, self.kernel_dim, self.padding_factor)
        gen_output = op.reshape_as_pics(gen_output)
        return gen_output

    def fit_transform(self, data_input, gen_fit_input, gen_samples_input):
        # data resize
        data_input, _ = self.data_size_fit(data_input)
        gen_fit_input, fit_patches_shape = self.data_size_fit(gen_fit_input)
        gen_samples_input, samples_patches_shape = self.data_size_fit(gen_samples_input)
        # fit_transform
        gen_fit_output, gen_samples_output = \
            super(Conv, self).fit_transform(data_input, gen_fit_input, gen_samples_input)
        # reshape to patches form
        gen_fit_output = gen_fit_output.reshape(fit_patches_shape)
        gen_samples_output = gen_samples_output.reshape(samples_patches_shape)
        # reshape to data form
        gen_fit_output = op.reconstruct_from_patches(gen_fit_output, self.kernel_dim, self.padding_factor)
        gen_samples_output = op.reconstruct_from_patches(gen_samples_output, self.kernel_dim, self.padding_factor)
        # reshape to pics form
        gen_fit_output = op.reshape_as_pics(gen_fit_output)
        gen_samples_output = op.reshape_as_pics(gen_samples_output)

        return gen_fit_output, gen_samples_output
