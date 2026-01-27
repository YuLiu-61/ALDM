import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity


class SPDLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SPDLayer, self).__init__()

        self.L = nn.Parameter(torch.tril(torch.randn(out_features, in_features)).cuda())
        self.D = nn.Parameter(torch.diag_embed(torch.rand(out_features)).cuda())

    def forward(self, x):
        # W = L * D * L^T
        L = self.L
        D = torch.diag_embed(torch.diagonal(self.D, dim1=-2, dim2=-1))
        W = torch.matmul(L, torch.matmul(D, L.t()))
        return torch.matmul(x, W)

def dup_matrix(mat):
    mat0 = torch.tril(mat, diagonal=-1)[:,:, :-1]
    mat0 += torch.triu(mat, diagonal=1)[:, :,1:]
    mat1 = torch.cat([mat0,mat],dim=2)
    mat2 = torch.cat([mat,mat0],dim=2)
    return mat1, mat2

##############################################################################
## 6 Different ways of generating time lags
##############################################################################
def compute_cosine_similarity(x):
    # x  (B, T, N)

    batch_size, seq_len, feature_dim = x.size()

    distance_matrix = torch.zeros((batch_size, seq_len, seq_len))
    model=SPDLayer(feature_dim,feature_dim)

    for b in range(batch_size):

        x_batch = x[b]
        x_batch_expanded = x_batch.unsqueeze(1)  # (T, 1, N)
        x_batch_tiled = x_batch.unsqueeze(0)  # (1, T, N)
        diff_matrix = x_batch_expanded - x_batch_tiled  # (T, T, N)
        diff_matrix=diff_matrix.unsqueeze(-2)

        out=model(diff_matrix)

        dis=torch.matmul(out,diff_matrix.transpose(3,2))
        dis=dis.squeeze(-1).squeeze(-1)
        dis = torch.sqrt(dis)
        distance_matrix[b] = dis


    return distance_matrix

def timelag_cosine(x,sigma=1):
    matrix = compute_cosine_similarity(x)
    # matrix=abs(1-matrix)
    matrix = 2 / (1 + torch.exp(matrix * sigma))
    matrix = torch.where(matrix < 1e-6, 0, matrix)  # set very small values to 0
    return matrix
def timelag_sigmoid(T,sigma=1):
    dist = np.arange(T)
    dist = np.abs(dist - dist[:, np.newaxis])
    matrix = 2 / (1 +np.exp(dist*sigma))
    matrix = np.where(matrix < 1e-6, 0, matrix)  # set very small values to 0         
    return matrix
def timelag_gaussian(T,sigma):
    dist = np.arange(T)
    dist = np.abs(dist - dist[:, np.newaxis])
    matrix = np.exp(-(dist**2)/(2 * sigma ** 2))
    matrix = np.where(matrix < 1e-6, 0, matrix) 
    return matrix

def timelag_same_interval(T):
    d = np.arange(T)
    X, Y = np.meshgrid(d, d)
    matrix = 1 - np.abs(X - Y) / T
    return matrix

def timelag_sigmoid_window(T, sigma=1, window_ratio=1.0):
    dist = np.arange(T)
    dist = np.abs(dist - dist[:, np.newaxis])
    matrix = 2 / (1 +np.exp(dist*sigma))
    matrix = np.where(matrix < 1e-6, 0, matrix)          
    dist_from_diag = np.abs(np.subtract.outer(np.arange(dist.shape[0]), np.arange(dist.shape[1])))
    matrix[dist_from_diag > T*window_ratio] = 0
    return matrix

def timelag_sigmoid_threshold(T, threshold=1.0):
    dist = np.ones((T,T))
    dist_from_diag = np.abs(np.subtract.outer(np.arange(dist.shape[0]), np.arange(dist.shape[1])))
    dist[dist_from_diag > T*threshold] = 0
    return dist

##############################################################################

