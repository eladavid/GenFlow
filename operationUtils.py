import numpy as np
from scipy.signal import convolve2d
import cv2
from matplotlib import pyplot as plt
from matplotlib import colors

def resize_dataset(x, new_dim):
    x_resized = np.zeros([x.shape[0], new_dim, new_dim])
    for i in range(x.shape[0]):
        x_resized[i,:,:] = cv2.resize(x[i,:,:], dsize=(new_dim,new_dim), interpolation=cv2.INTER_CUBIC)
    return x_resized


def flatten_and_fit_dims(x, patches=0):
    if patches == 0:
        x_cs = x.reshape(-1, x.shape[1]*x.shape[2]).T
    else:
        x_cs = x.reshape(-1, x.shape[1] * x.shape[2])
    return x_cs


def reshape_as_pics(x):
    pic_dim = int(np.sqrt(x.shape[0]))
    x_pics = x.T.reshape(x.shape[1], pic_dim, pic_dim)
    return x_pics

# whiten_data(x):
#   whitening of data matrix
#   INPUTS:
#       x - DxN data matrix
#   OUTPUT:
#       x_whitened - DxN whitened data (diagonal cov mat)
def whiten_data(x):
    assert (x.ndim == 1)
    D = np.size(x, 0)
    N = np.size(x, 1)
    C = np.cov(x)
    w, v = np.linalg.eig(C)
    mu = np.mean(x,1)
    x_whitened = v.T @ (x.T - mu).T
    return x_whitened


# gen_orthogonal_mat(x, x_cov_mat=0):
# generate random orthogonal matrix DxD
#   INPUTS:
#       x - DxN data matrix
#       x_cov_mat (optional) - data cov mat, to reduce calculations on serial calling
#       random_state (optional) - set random seed generator to keep track of simulation
#   OUTPUT:
#       W - DxD orthogonal matrix
def gen_orthogonal_mat(x, x_cov_mat=0, random_state=-1):
    assert x.ndim != 1
    if random_state != -1:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random
    D = np.size(x, 0)
    N = np.size(x,1)
    C = x_cov_mat
    if C.ndim == 1:
        C = np.cov(x)
    W = np.zeros([D, D])
    W[:, 0] = x[:, rng.randint(1, N)]
    if np.all(W[:, 0] == 0):
        W[0:, 0] = 1
    W[:, 1:] = rng.randn(D, D-1)
    W[:, 1:] = (np.eye(D) - W[:, 0] @ W[:, 0].T / sum(W[:, 0]**2)) @ W[:,1:]
    W, _, _ = np.linalg.svd(W)
    return W


def patches_projection(x_pics, p_gen_pics, patch_sz, debug=0):
    rand_kernel = np.random.randn(patch_sz, patch_sz)
    kernel_rms = np.sqrt(np.sum(rand_kernel**2)) / patch_sz**2
    rand_kernel = rand_kernel / kernel_rms
    # debug mode - using a unit kernel to perform the identity transform
    if debug==1:
        rand_kernel = np.zeros(rand_kernel.shape)
        rand_kernel[int(np.ceil(patch_sz/2))-1, int(np.ceil(patch_sz/2))-1] = 1

    N = np.size(x_pics, 0)
    M = np.size(p_gen_pics, 0)
    pic_axis = np.size(x_pics, 1)

    x_patches_projected = np.zeros((N, pic_axis, pic_axis))
    p_gen_patches_projected = np.zeros((M, pic_axis, pic_axis))
    for i in range(max(N, M)):
        if i < N:
            x_patches_projected[i, :, :] = convolve2d(x_pics[i, :, :], rand_kernel, mode='same')
        if i < M:
            p_gen_patches_projected[i, :, :] = convolve2d(p_gen_pics[i, :, :], rand_kernel, mode='same')

    x_patches_projected = np.squeeze(x_patches_projected.reshape(np.size(x_patches_projected), -1))
    p_gen_patches_projected = np.squeeze(p_gen_patches_projected.reshape(np.size(p_gen_patches_projected), -1))
    return x_patches_projected, p_gen_patches_projected, rand_kernel


# fit_axis_mapping_func(axis_data, axis_gen, poly_deg):
#   fits 2 sorted data series using polyfit
#   INPUTS:
#       axis_data - Dx1 data feature vector
#       axis_gen - Dx1 random-generated feature vector
#       poly_deg - polynomial fit degree
#   OUTPUT:
#       p - polynomial coeffs
def fit_axis_mapping_func(axis_data, axis_gen, poly_deg, method='TV'):
    axis_gen_sorted = np.sort(axis_gen)
    axis_data_sorted = np.sort(axis_data)
    if axis_gen_sorted.ndim > 1:
        axis_gen_sorted = np.squeeze(axis_gen_sorted, 0)
    if axis_data_sorted.ndim > 1:
        axis_data_sorted = np.squeeze(axis_data_sorted, 0)
    sampling_vec = np.floor(np.linspace(0, len(axis_data_sorted)-1, len(axis_gen_sorted))).astype(int)
    axis_data_sorted_sampled = axis_data_sorted[sampling_vec]
    if method == 'L1':
        axis_score = np.sum(np.abs(axis_data_sorted_sampled - axis_gen_sorted))
    elif method == 'TV':
        axis_score = np.max(np.abs(axis_data_sorted_sampled - axis_gen_sorted))
    else:
        assert 0, 'wrong method usage'

    p = np.polyfit(axis_gen_sorted, axis_data_sorted_sampled, poly_deg)
    support_bounds = (axis_gen_sorted[0], axis_gen_sorted[-1])

    p_lin = np.polyfit((axis_gen_sorted[0], axis_gen_sorted[-1]),
                       (axis_data_sorted_sampled[0], axis_data_sorted_sampled[-1]), 1)
    return p, axis_score, support_bounds, p_lin


# apply_transformation(x_gen, io_mapping_mat):
#   apply polynomial transformation on data series
#   INPUTS:
#       x_gen - DxN random-generated feature matrix
#       io_mapping_mat - poly_deg x D marginal fittings
#   OUTPUT:
#       x_gen_trans - DxN transformed feature matrix
def apply_transformation(x_gen, io_mapping_mat):
    D = np.size(x_gen, 0)
    x_gen_trans = np.zeros(x_gen.shape)
    for i in range(D):
        x_gen_trans[i, :] = np.polyval(io_mapping_mat[:, i], x_gen[i, :])
    return x_gen_trans


def apply_transformation_on_axis(axis_gen, io_mapping_vec, support_bounds=None, p_lin=None):
    if np.any(np.isnan(io_mapping_vec)):
        io_mapping_vec = np.zeros(io_mapping_vec.shape)
        io_mapping_vec[-2] = 1
    if np.any(np.isinf(io_mapping_vec)):
        io_mapping_vec[np.isinf(io_mapping_vec)] = 0
    if support_bounds is not None:
        axis_gen[axis_gen < support_bounds[0]] = support_bounds[0]
        axis_gen[axis_gen > support_bounds[-1]] = support_bounds[-1]
    axis_gen_trans = np.polyval(io_mapping_vec, axis_gen)
    #if support_bounds is not None:
        #axis_gen_trans[axis_gen_trans < support_bounds[0]] = np.polyval(p_lin, axis_gen[axis_gen_trans < support_bounds[0]])
    #    axis_gen_trans[axis_gen_trans < support_bounds[0]] = support_bounds[0]
        #axis_gen_trans[axis_gen_trans > support_bounds[-1]] = np.polyval(p_lin, axis_gen[axis_gen_trans > support_bounds[-1]])
    #   axis_gen_trans[axis_gen_trans > support_bounds[-1]] = support_bounds[-1]
    return axis_gen_trans


def fit_axis_and_apply_mapping(axis_data, axis_gen, poly_deg, method='TV'):
    p, _ = fit_axis_mapping_func(axis_data, axis_gen, poly_deg)
    axis_gen_trans = apply_transformation_on_axis(axis_gen, p)
    return p, axis_gen_trans


def fit_axis_and_apply_mapping_multiprocessing(args):
    axis_data, axis_gen, poly_deg = args
    p, _ = fit_axis_mapping_func(axis_data, axis_gen, poly_deg)
    axis_gen_trans = apply_transformation_on_axis(axis_gen, p)
    return p, axis_gen_trans


# cart2sph(x): need to fix
#   apply cartesian to spherical transformation of dim D
#   INPUTS:
#       x - DxN feature matrix
#   OUTPUT:
#       x_sph - DxN transformed feature matrix
def cart2sph(x):
    # need to implement what happens when all but one are almost 0
    scaling = 1e6
    x = x * scaling
    eps = 1e-6
    D = np.size(x, 0)
    if (x.ndim == 1):
        x = np.expand_dims(x,1)
    x_sph = np.zeros(x.shape)
    x_sph[0, :] = np.sqrt(sum((x/scaling)**2, 0))
    for i in (range(1, D)):
        if (i < D-1):
            denom = np.sqrt(sum(x[i:, :]**2, 0))
            arg = x[i-1, :] / (denom + eps)
            x_sph[i, :] = np.arctan(1/(arg+eps))
        else:
            denom = x[i, :]
            arg = (x[i-1, :] + np.sqrt(x[i-1, :]**2 + x[i, :]**2)) / (denom + eps)
            x_sph[i, :] = 2*np.arctan(1/(arg+eps))
    return x_sph


# sph2cart(x): need to fix
#   apply spherical to cartesian transformation of dim D
#   INPUTS:
#       x - DxN feature matrix
#   OUTPUT:
#       x_cart - DxN transformed feature matrix
def sph2cart(x):
    D = np.size(x, 0)
    if (x.ndim == 1):
        x = np.expand_dims(x,1)
    x_cart = np.zeros(x.shape)
    x_cart[0, :] = x[0, :] * np.cos(x[1, :])
    base_arg = x[0, :]
    for i in (range(1, D)):
        if (i < D-1):
            base_arg = base_arg * np.sin(x[i, :])
            x_cart[i, :] = base_arg * np.cos(x[i + 1, :])
        else:
            x_cart[i, :] = base_arg * np.sin(x[i, :])
    return x_cart


# convert_data_to_collage(x, pics_per_row=10, pics_per_col=10):
#   assemble a collage of pictures from an array of pictures
#   INPUTS:
#       x - DxN column stacked pictures (features) matrix
#       pics_per_row (default = 10) - num of pics per row in collage
#       pics_per_col (default = 10) - num of pics per column in collage
#       random_state (optional) - set random seed to control generation
#   OUTPUT:
#       collage - collage picture
def convert_data_to_collage(x, pics_per_row=10, pics_per_col=10, random_state=-1):
    # need to add color option
    D = np.size(x, 0)
    N = np.size(x, 1)
    if random_state != -1:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random
    pic_dim = int(np.sqrt(D))
    collage = np.zeros([pics_per_row * pic_dim, pics_per_col * pic_dim])
    samples_idcs = rng.choice(np.arange(N), pics_per_col*pics_per_row, replace=False)
    collage_cs = x[:, samples_idcs]
    collage_images = collage_cs.reshape(pic_dim, pic_dim, -1)
    for i in range(pics_per_row):
        for j in range(pics_per_col):
            collage[i*pic_dim:(i+1)*pic_dim, j*pic_dim:(j+1)*pic_dim] = collage_images[:, :, pics_per_row*i + j]
    return collage


def transform_sample(random_state_list, io_mapping_tensor, x_data, transform_input=0):
    features_dim = x_data.shape[0]
    if transform_input == 0:
        gen_sample_last = np.random.randn(features_dim, 1)
    else:
        gen_sample_last = transform_input
    C = np.cov(x_data)
    gen_samp_trans = np.zeros(gen_sample_last.shape)
    for i, seed in enumerate(random_state_list):
        W = gen_orthogonal_mat(x_data, C, seed)
        gen_samp_rot = W @ gen_sample_last
        for d in range(features_dim):
            gen_samp_trans[d] = np.polyval(io_mapping_tensor[i, :, d], gen_samp_rot[d])
        gen_sample_last = W.T @ gen_samp_trans
    return gen_sample_last


# calc_axis_io_score(axis_data, axis_gen, method='TV'):
#   assemble a collage of pictures from an array of pictures
#   INPUTS:
#       axis_data - 1xN feature vector
#       axis_gen - 1xN feature vector
#       method - string, to describe probability diff measure
#   OUTPUT:
#       collage - collage picture
def calc_axis_io_score(axis_data, axis_gen, method='TV'):
    # optional use kde to smooth probabilities
    axis_gen_sorted = np.sort(axis_gen)
    axis_data_sorted = np.sort(axis_data)
    if method == 'TV':
        axis_score = np.sum(np.abs(axis_data_sorted - axis_gen_sorted))
    else:
        print('wrong usage')
    return axis_score


def im2col_sliding_strided(im, blk_sz, stepsize=1):
    # Parameters
    m, n = im.shape
    s0, s1 = im.strides
    n_rows = m - blk_sz[0] + 1
    n_cols = n - blk_sz[1] + 1
    shp = blk_sz[0], blk_sz[1], n_rows, n_cols
    strd = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(im, shape=shp, strides=strd)
    return out_view.reshape(blk_sz[0] * blk_sz[1], -1)[:, ::stepsize]


# split_to_pathces(x, patch_w, patch_h, overlap):
#   split dataset to patches form
#   INPUTS:
#       x - DxN feature matrix
#       patch_sz - patch single dim (patch is square)
#       stride - stride for the moving filter
#   OUTPUT:
#       x_patches - patch_sz**2 x num_of_patches x N tensor
def split_to_patches(x, patch_sz, padding_flag=1, stride=1):
    D = np.size(x, 0)
    N = np.size(x, 1)
    #x_im_view = x.reshape(int(np.sqrt(D)), int(np.sqrt(D)), N)
    x_im_view = reshape_as_pics(x)
    if padding_flag:
        padding_factor = int(np.floor(patch_sz / 2))
        # do not pad 3-rd dim
        npad = ((0, 0), (padding_factor, padding_factor), (padding_factor, padding_factor))
        x_im_view_padded = np.pad(x_im_view, pad_width=npad, mode='constant', constant_values=0)#[:, :, 1:-1]
        padding_factor = 2 * int(np.floor(patch_sz/2))
    else:
        x_im_view_padded = x_im_view
        padding_factor = 0
    patches_per_im = int((np.sqrt(D) + 1 + padding_factor - patch_sz)**2)
    x_patches = np.zeros([(patch_sz**2), patches_per_im, N])
    for i in range(N):
        x_patches[:, :, i] = im2col_sliding_strided(x_im_view_padded[i, :, :], [patch_sz, patch_sz], stride)
    return x_patches, padding_factor


# reconstruct_from_patches(x_patches, patch_sz, padding_factor, stride=1):
#   this function samples the center of each patch and reconstructs images from them
#   INPUTS:
#       x_patches - patch_sz x patches_per_im x N feature matrix
#       patch_sz - patch single dim (patch is square)
#       padding factor - info about how the patches were generated
#       stride - stride for the moving filter - not in use at the moment
#   OUTPUT:
#       x_recon - D (features) x N (samples) matrix
def reconstruct_from_patches(x_patches, patch_sz, padding_factor, stride=1):
    D = int((np.sqrt(np.size(x_patches, 1)) - 1 - padding_factor + patch_sz)**2)
    N = np.size(x_patches, 2)
    center_elem_idx = int(np.ceil(patch_sz**2/2)) - 1 # -1 offset for pythonic indexing as opposed to matlab
    x_patches_centers = x_patches[center_elem_idx, :, :].squeeze()
    # this line might be redundant
    x_recon = x_patches_centers.reshape(D, N)
    return x_recon

