import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from numpy.lib.stride_tricks import as_strided
from src.sparse_uplap import BoundaryMatrix



def delete_row_col(matrix, i):
    # Delete the i-th column
    mask = np.ones(matrix.shape[1], dtype=bool)
    mask[i] = False
    
    # matrix = matrix[:, col_mask]  # Remove column
    
    col_i = (matrix[mask, i]).reshape(-1, 1)
    row_i = matrix[i, mask].reshape(1, -1)
    
    # Delete the i-th row
    # row_mask = torch.ones(matrix.size(0), dtype=torch.bool, device=matrix.device)
    # row_mask[i] = False
    matrix_sub = matrix[:, mask][mask, :]  # Remove row
    
    return matrix[i, i], row_i, col_i, matrix_sub


def compute_up_persistent(M, idx):
    m_ii, row_i, col_i, M_sub = delete_row_col(M, idx)
    if np.abs(m_ii) > 1e-10:
        M_schur = M_sub - (1 / m_ii) * col_i @ row_i
    else:
        M_schur = M_sub
    return M_schur


def generate_non_brch_mtx(shape):
    num_r, num_c = shape
    M = np.zeros(shape, dtype=int)
    for i in range(num_r):
        # Randomly choose to generate 1 or 2 integers
        num_integers = np.random.randint(1, 3, (1,)).item()

        # Generate the integers (adjust low/high as needed)
        result = np.random.randint(low=0, high=num_c, size=(num_integers,))

        if num_integers == 2:
            val = np.array([1, -1])
        elif num_integers == 1:
            val = np.random.randint(1, 3, size=(1,))
        
        M[i, result] = val
        M[M == 2] = -1
    return M


def create_input_data(num_r = 20, num_c = 15, idx_0 = 10):
    # idx_0 <= num_r
    idx_1 = num_r-1
    filt_idx_0_2 = np.zeros(num_r, dtype=bool)
    filt_idx_0_2[-idx_0:] = True

    filt_idx_1_2 = np.zeros(num_r, dtype=bool)
    filt_idx_1_2[-idx_1:] = True

    shape = (num_r, num_c)
    bdry_opr = generate_non_brch_mtx(shape)
    return filt_idx_0_2, filt_idx_1_2, bdry_opr


def max_pool2d(x, pool_size, stride):
    """
    Perform 2D max pooling on the input array.
    
    Parameters:
        x (ndarray): Input array of shape (..., H, W).
        pool_size (tuple): Pooling window size (ph, pw).
        stride (tuple): Stride values (sh, sw).
        
    Returns:
        ndarray: Pooled output array of shape (..., out_h, out_w).
    """
    ph, pw = pool_size
    sh, sw = stride
    # Extract input dimensions
    *batch_dims, H, W = x.shape
    # Calculate output dimensions
    out_h = (H - ph) // sh + 1
    out_w = (W - pw) // sw + 1
    # Create strided view of the input
    H_stride, W_stride = x.strides[-2], x.strides[-1]
    new_shape = (*batch_dims, out_h, out_w, ph, pw)
    new_strides = (*x.strides[:-2], sh * H_stride, sw * W_stride, H_stride, W_stride)
    x_strided = as_strided(x, shape=new_shape, strides=new_strides)
    # Compute max over the pooling window
    return x_strided.max(axis=(-2, -1))

def get_bdry_opr(X, filt_1, filt_2, filt_0=None):
    # X = get_cubical_data(label, data=data)
    bdry_class = BoundaryMatrix(X=X, filt_0=filt_0, filt_1=filt_1, filt_2=filt_2)
    bdry_class.chains_count()
    bdry_class.find_boundary_opr()
    bdry_class.find_cell_idx()
    
    bdry_opr = csr_matrix(bdry_class.bdry_opr)
    filt_idx_0_2 = bdry_class.filt_idx_0_2
    filt_idx_1_2 = bdry_class.filt_idx_1_2
    num_filt = np.sum(~filt_idx_1_2)
    
    return bdry_opr, filt_idx_0_2, filt_idx_1_2, num_filt