import numpy as np
from scipy.sparse import csr_matrix #, coo_matrix
from scipy import sparse
# from numba import njit
import scipy.sparse as sp


class UnionFind:
    def __init__(self, sparse_mtx):
        self.sparse_mtx = sparse_mtx
        self.num_e, self.num_v = sparse_mtx.shape
        self.parent = list(range(self.num_v))
        self.rank = [1] * self.num_v 
        self.loops = None

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # Path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            # Union by rank
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                if self.rank[root_x] == self.rank[root_y]:
                    self.rank[root_y] += 1

    def connected_components_from_sparse_incidence(self):
        """
        Find connected components from a sparse incidence matrix using Union-Find with while loops.
        """
        edge_row = 0
        while edge_row < self.num_e:
            row_data = self.sparse_mtx.getrow(edge_row)
            nodes = row_data.indices
            if len(nodes) == 2:
                u = nodes[0]
                v = nodes[1]
                self.union(u, v)
            edge_row += 1

        node = 0
        while node < self.num_v:
            root = self.find(node)
            if self.parent[node] != root:
                self.parent[node] = root
            node += 1
       
    def find_loops(self):
        # if not isinstance(self.sparse_mtx, csr_matrix):
        #     self.sparse_mtx = csr_matrix(self.sparse_mtx)
        row_lengths = self.sparse_mtx.indptr[1:] - self.sparse_mtx.indptr[:-1]
        single_entry_rows = np.where(row_lengths == 1)[0]
        if len(single_entry_rows) == 0:
            self.loops = np.array([], dtype=int)
            #return np.array([], dtype=int)
        start_indices = self.sparse_mtx.indptr[single_entry_rows]
        column_indices = self.sparse_mtx.indices[start_indices]
        # return np.unique(column_indices)
        self.loops = np.unique(column_indices)

    # @njit
    def find_loops_roots(self):
        self.loops_roots = []
        for node in self.loops:
            while self.parent[node] != node:
                self.parent[node] = self.parent[self.parent[node]]  # Path compression
                node = self.parent[node]
            self.loops_roots.append(node)
        # return roots
        
class UpPersistentLaplacianSparse:
    def __init__(self, bdry_opr, filt_idx_0_2=None, filt_idx_1_2=None):
        self.bdry_opr = bdry_opr 
        if not isinstance(self.bdry_opr, csr_matrix):
            self.bdry_opr = csr_matrix(self.bdry_opr)
        self.filt_idx_0_2 = filt_idx_0_2 # boolean valued
        self.filt_idx_1_2 = filt_idx_1_2 #boolean valued
        self.rel_bdry_opr = None
        # self.rel_bdry_opers_list = []
        # self.indices_it = None

        if filt_idx_0_2 is not None:
            self.temp_labels = np.where(self.filt_idx_1_2)[0]
            sub_filt_idx = np.where(filt_idx_1_2 != filt_idx_0_2)[0]
            self.indices_it = iter(sub_filt_idx)
            self.count = 0
            # self.indices_it = iter(range(len(sub_filt_idx)))

        '''
        if filt_idx_0_2 is not None: 
            filt_idx_0_1 = np.where(filt_idx_1_2 != filt_idx_0_2)[0]
            self.filt_idx_0_1 = filt_idx_0_1
        else:
            self.filt_idx_0_1 = None
            # self.idx_0_1 = None
        
        if self.filt_idx_0_1 is not None:
            if self.filt_idx_0_1.size != 0:
            # self.filt_indices = np.flatnonzero(self.filt_idx_0_1)
            # if self.filt_indices.size != 0:
                self.indices_it = iter(self.filt_idx_0_1)
        '''
    
    def find_D(self):
        if self.filt_idx_1_2.size != 0:
            D_mtx = self.bdry_opr[~self.filt_idx_1_2, :]
        else:
            D_mtx = self.bdry_opr

        if D_mtx.shape[0] == 0:
            num_col = D_mtx.shape[1]
            D_mtx = csr_matrix((1, num_col), dtype=int)

        self.D_mtx = D_mtx

    def find_root(self):
        self.uf = UnionFind(self.D_mtx)
        self.uf.connected_components_from_sparse_incidence()

    def find_rel_bdry_opr(self):
        loopisin = np.isin(self.uf.parent, np.unique(self.uf.loops_roots))
        no_loop_parent = np.array(self.uf.parent)[~loopisin]
        no_loop_bdry_opr = self.bdry_opr[:, ~loopisin]

        
        # Get unique labels and their corresponding indices
        roots, inverse = np.unique(no_loop_parent, return_inverse=True)
        n_unique = roots.size # len(roots)
        n_columns = no_loop_bdry_opr.shape[1]
        
        # Create grouping matrix in CSR format
        # rows = inverse
        cols = np.arange(n_columns)
        data = np.ones(n_columns, dtype=int)
        
        self.G = sparse.csr_matrix((data, (inverse, cols)), shape=(n_unique, n_columns))
        
        # Compute result by matrix multiplication
        self.rel_bdry_opr = (no_loop_bdry_opr[self.filt_idx_1_2, :]).dot(self.G.T)
        self.roots = roots
        # self.inverse = inverse
        
        # return result, unique_labels

    def find_diag(self):
        self.diag_array = np.array((1 / self.G.sum(axis=1)).flatten())[0]
        # diag_list = (1 / self.G.sum(axis=1)).flatten().tolist()[0]
        self.diag = sparse.diags(diagonals=self.diag_array)

    def find_up_persistent_Laplacian(self):
        # self.find_boundary_opr()
        # self.find_cell_idx_check()
        # self.find_D()
        # self.find_graph_D()
        # self.find_rel_bdry_opr()
        # self.find_diag()

        self.up_lap = self.rel_bdry_opr @ self.diag @ self.rel_bdry_opr.transpose()
        # return up_lap

    def recheck_rel_bdry_opr(self):
        if self.rel_bdry_opr.shape[1] == 0:
            num_row = self.rel_bdry_opr.shape[0]
            self.rel_bdry_opr = csr_matrix((num_row, 1), dtype=int)

    def recheck_diag(self):
        if 0 in self.diag.shape:
            self.diag = csr_matrix((1, 1), dtype=int)

    def add_edge_temp(self):
        edge = next(self.indices_it)
        self.rel_bdry_opr = self.rel_bdry_opr[np.arange(self.rel_bdry_opr.shape[0]) != edge - self.count, :]
        col_indices = self.bdry_opr.indices[self.bdry_opr.indptr[edge]:self.bdry_opr.indptr[edge+1]]
        self.count += 1

        if col_indices.size == 1:
            loop_root = self.uf.find(col_indices[0])
            if loop_root in self.roots:
                is_not_loop = self.roots != loop_root
                self.rel_bdry_opr = self.rel_bdry_opr[:, is_not_loop]
                self.roots = self.roots[is_not_loop]

                # new_diag = self.diag.diagonal()[is_not_loop]
                self.diag_array = self.diag_array[is_not_loop]
                self.diag = sparse.diags(diagonals=self.diag_array)
                
            # else: self.rel_bdry_opr = temp_rel_bdry_opr

        elif col_indices.size == 2:
            u, v = col_indices
            u_root = self.uf.find(u)
            v_root = self.uf.find(v)
            
            u_in_roots = u_root in self.roots
            v_in_roots = v_root in self.roots
            
            if (not u_in_roots) and (not v_in_roots):
                # self.rel_bdry_opr = temp_rel_bdry_opr
                pass
            elif u_in_roots and (not v_in_roots):
                # u_loop_root = self.uf.find(u)
                is_not_loop = self.roots != u_root
                self.rel_bdry_opr = self.rel_bdry_opr[:, is_not_loop]

                self.diag_array = self.diag_array[is_not_loop]
                self.diag = sparse.diags(diagonals=self.diag_array)
                
                self.roots = self.roots[is_not_loop]

            elif (not u_in_roots) and v_in_roots:
                is_not_loop = self.roots != v_root
                self.rel_bdry_opr = self.rel_bdry_opr[:, is_not_loop]
                
                self.diag_array = self.diag_array[is_not_loop]
                self.diag = sparse.diags(diagonals=self.diag_array)
                
                self.roots = self.roots[is_not_loop]
                
            elif u_in_roots and v_in_roots:
                if u_root == v_root:
                    pass
                    # self.rel_bdry_opr = temp_rel_bdry_opr
                elif u_root != v_root:
                    where_u = self.roots == u_root
                    where_v = self.roots == v_root
                
                    self.uf.union(u, v)
                    uv_root = self.uf.find(u)
                    where_uv = self.roots == uv_root
                    self.roots[where_u] = uv_root
                    self.roots[where_v] = uv_root

                    # Get unique labels and their inverse indices
                    self.roots, inv = np.unique(self.roots, return_inverse=True)
                    n_cols = self.rel_bdry_opr.shape[1]
                    n_unique = self.roots.size

                    # Construct transformation matrix in csr format for efficient multiplication
                    cols = np.arange(n_cols)
                    data = np.ones(n_cols, dtype=int)
                    transformation = csr_matrix((data, (inv, cols)), shape=(n_unique, n_cols))

                    # Compute the sum by matrix multiplication
                    self.rel_bdry_opr = self.rel_bdry_opr.dot(transformation.T)

                    # Update diagonal matrix
                    diag_u = self.diag_array[where_u]
                    diag_v = self.diag_array[where_v]
                    harm_mean = diag_u * diag_v / (diag_u + diag_v)

                    self.diag_array[where_uv] = harm_mean

                    root_idx = np.where(where_uv)[0].item()
                    u_idx = np.where(where_u)[0].item()
                    v_idx = np.where(where_v)[0].item()
                    
                    if root_idx == u_idx:
                        self.diag_array = np.delete(self.diag_array, v_idx)
                    elif root_idx == v_idx:
                        self.diag_array = np.delete(self.diag_array, u_idx)
                        
                    self.diag = sparse.diags(diagonals=self.diag_array)
                    

class BoundaryMatrix:
    def __init__(self, X, filt_0=None, filt_1=None, filt_2=None):
        # cell_dim is the dim of up persistent Laplacian
        self.X = X
        self.filt_0 = filt_0
        self.filt_1 = filt_1
        self.filt_2 = filt_2
        self.filt_idx = None

    def chains_count(self):  # X is a tensor of shape: (Channel, Row, Column)
        m, n = self.X.shape
        self.num_v = int((m + 1) / 2 * (n + 1) / 2)
        self.num_s = int((m - 1) / 2 * (n - 1) / 2)
        self.num_e = int(m * n - self.num_v - self.num_s)
        
    def cell_filt(self, filt, cell_dim):  # Return the coordinates of filtered dim-dimensional cubes.
        global cells
        boolean_val = np.where(self.X <= filt)
        X_idx = np.concatenate((boolean_val[0].reshape(-1, 1), boolean_val[1].reshape(-1, 1)), axis=1)

        if len(X_idx) != 0:
            if cell_dim == 2:
                X_idx_odds = (X_idx[:, 0] % 2 == 1) & (X_idx[:, 1] % 2 == 1)
                cells = X_idx[X_idx_odds, :]

            elif cell_dim == 1:
                X_idx_sum = X_idx.sum(axis=1)
                cells = X_idx[X_idx_sum % 2 == 1, :]

            elif cell_dim == 0:
                X_idx_evens = (X_idx[:, 0] % 2 == 0) & (X_idx[:, 1] % 2 == 0)
                cells = X_idx[X_idx_evens, :]

            if len(cells) != 0:
                return cells
            elif len(cells) == 0:
                return np.array([], dtype=int)

        elif len(X_idx) == 0:
            return np.array([], dtype=int)

    def bdry_idx(self, cell_coords, cell_dim):  # cell_coords are a tensor of cell coordinates,
        global bdry_idxs
        m, n = self.X.shape
        if len(cell_coords) != 0:
            if cell_dim == 2:  # Square cells
                x_0s = ((cell_coords[:, 0] * n + cell_coords[:, 1]) / 2 - 1).astype(int)
                x_0s = x_0s.reshape(-1, 1)
                y_0s = ((n * cell_coords[:, 0] - n + cell_coords[:, 1] + 1) / 2 - 1).astype(int)
                y_0s = y_0s.reshape(-1, 1)
                bdry_idxs = np.concatenate((y_0s, x_0s, x_0s + 1, y_0s + self.X.shape[1]), axis=1)
                
                # bdry_idxs = torch.transpose(torch.stack([y_0s, x_0s, x_0s + 1, y_0s + self.X.shape[2]]), 0, 1)

            elif cell_dim == 1:  # Edge cells
                cell_coords_v = cell_coords[cell_coords[:, 0] % 2 == 1, :]  # coordinates of vertical edges
                cell_coords_h = cell_coords[cell_coords[:, 1] % 2 == 1, :]  # coordinates of horizontial edges

                v_x_0s = ((cell_coords_v[:, 0] - 1) * (n + 1) / 4 + cell_coords_v[:, 1] / 2).astype(int)
                v_x_0s = v_x_0s.reshape(-1, 1)
                v_x_1s = (v_x_0s + (n + 1) / 2).astype(int)
                v_x_1s = v_x_1s.reshape(-1, 1)
                
                # x_0 = p * (n + 1) / 4 + (q + 1) / 2 - 1
                # x_1 = x_0 + 1
                h_x_0s = (cell_coords_h[:, 0] * (n + 1) / 4 + (cell_coords_h[:, 1] + 1) / 2 - 1).astype(int)
                h_x_0s = h_x_0s.reshape(-1, 1)
                h_x_1s = h_x_0s + 1
                h_x_1s = h_x_1s.reshape(-1, 1)
                
                bdry_idxs = np.zeros((cell_coords.shape[0], 2)).astype(int)

                bdry_idxs[cell_coords[:, 0] % 2 == 1, :] = np.concatenate((v_x_0s, v_x_1s), axis=1)
                # torch.transpose(torch.stack((v_x_0s, v_x_1s)), 0, 1)
                bdry_idxs[cell_coords[:, 1] % 2 == 1, :] = np.concatenate((h_x_0s, h_x_1s), axis=1)
                # torch.transpose(torch.stack((h_x_0s, h_x_1s)), 0, 1)

            elif cell_dim == 0:  # Vertices cells
                bdry_idxs = np.array([]).astype(int)

        elif len(cell_coords) == 0:
            bdry_idxs = np.array([]).astype(int)

        return bdry_idxs

    def find_boundary_opr(self):
        # c, num_v, num_e, num_s = self.torch_chains_count()
        chain_space_dim = self.num_e 
        cell_coords = self.cell_filt(filt=self.filt_2, cell_dim=2)
        bdry_coords = self.bdry_idx(cell_coords=cell_coords, cell_dim=2)

        if len(bdry_coords) != 0:
            num_cells, bdry_cells = bdry_coords.shape
            which_column = np.tensordot(np.arange(num_cells).astype(int), np.ones(bdry_cells).astype(int), axes=0)
            # bdry_indices = np.concatenate((bdry_coords.flatten().reshape(-1, 1), which_column.flatten().reshape(-1, 1)), axis=1)

            row_indices = bdry_coords.flatten()
            col_indices = which_column.flatten()
            
            vals = np.tile(np.array([-1, 1, -1, 1]).astype(int), num_cells)
            bdry_opr = csr_matrix((vals, (row_indices, col_indices)))  # recheck coo_matrix
            mask = bdry_opr.getnnz(axis=1) > 0  # Boolean mask of non-zero rows
            bdry_opr = bdry_opr[mask, :]       # Apply mask to filter rows

            # bdry_opr = np.zeros([chain_space_dim, num_cells]).astype(int)

            # bdry_opr[bdry_indices[:, 0], bdry_indices[:, 1]] = np.tile(np.array([-1, 1, -1, 1]).astype(int), num_cells)
            # bdry_opr = bdry_opr[~np.all(bdry_opr == 0, axis=1)]
        
            # bdry_opr = remove_zero_rows(bdry_opr)

        else:
            bdry_opr = np.zeros((chain_space_dim, 1)).astype(int)
        # return bdry_opr # a tensor object
        self.bdry_opr = bdry_opr

    def find_cell_idx(self):
        cell_filt_2 = self.cell_filt(filt=self.filt_2, cell_dim=1)
        cell_filt_1 = self.cell_filt(filt=self.filt_1, cell_dim=1)
        filt_idx_1_2 = get_boolean_mask(cell_filt_2, cell_filt_1)
        self.filt_idx_1_2 = filt_idx_1_2

        if self.filt_0 is not None:
            cell_filt_0 = self.cell_filt(filt=self.filt_0, cell_dim=1)
            filt_idx_0_2 = get_boolean_mask(cell_filt_2, cell_filt_0)
        elif self.filt_0 is None:
            filt_idx_0_2 = None
        self.filt_idx_0_2 = filt_idx_0_2   

# def remove_zero_rows(X):
#     if not isinstance(X, coo_matrix):
#         X = X.tocoo()
#     # Identify rows with at least one non-zero entry
#     data_non_zero = X.data != 0
#     rows_with_non_zero = X.row[data_non_zero]
#     non_zero_rows = np.unique(rows_with_non_zero)
#     # Filter entries in those rows
#     mask = np.isin(X.row, non_zero_rows)
#     new_data = X.data[mask]
#     new_col = X.col[mask]
#     new_row = X.row[mask]
#     # Create mapping from original row indices to new indices
#     new_row_indices = np.full(X.shape[0], -1, dtype=int)
#     new_row_indices[non_zero_rows] = np.arange(non_zero_rows.size)
#     adjusted_row = new_row_indices[new_row]
#     # Update shape
#     new_shape = (non_zero_rows.size, X.shape[1])
#     # Build new COO matrix
#     return coo_matrix((new_data, (adjusted_row, new_col)), shape=new_shape)

def get_boolean_mask(N, M):
    # Ensure contiguous arrays for memory alignment
    N_cont = np.ascontiguousarray(N)
    M_cont = np.ascontiguousarray(M)
    
    # Create a dtype that views each row as a single "void" element
    row_dtype = np.dtype((np.void, N_cont.dtype.itemsize * N_cont.shape[1]))
    
    # Convert rows to 1D void arrays for vectorized comparison
    N_view = N_cont.view(row_dtype).ravel()  # Shape: (n,)
    M_view = M_cont.view(row_dtype).ravel()  # Shape: (m,)
    
    # Check if each row in N exists in M
    return np.isin(N_view, M_view)



                    
def create_symmetric_matrix(D):
    m, n = D.shape
    top_left = sp.csr_matrix((m, m))
    bottom_right = sp.csr_matrix((n, n))
    return sp.bmat([[top_left, D], [D.T, bottom_right]], format='csr')                    
                    
                    
                    
                    
            
