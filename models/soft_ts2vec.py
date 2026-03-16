from models.soft_losses import *
from utils import *
from models.encoder import TSEncoder
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
class SPDLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SPDLayer, self).__init__()

        self.L = nn.Parameter(torch.tril(torch.randn(out_features, in_features)))
        self.D = nn.Parameter(torch.diag_embed(torch.rand(out_features)))

    def forward(self, x):

        L = self.L
        D = torch.diag_embed(torch.diagonal(self.D, dim1=-2, dim2=-1))
        M = torch.matmul(L, torch.matmul(D, L.t()))


        return torch.matmul(x, M)

def dup_matrix(mat):
    mat0 = torch.tril(mat, diagonal=-1)[:,:, :-1]
    mat0 += torch.triu(mat, diagonal=1)[:, :,1:]
    mat1 = torch.cat([mat0,mat],dim=2)
    mat2 = torch.cat([mat,mat0],dim=2)
    return mat1, mat2

# def dup_matrix(mat):
#     mat0 = torch.tril(mat, diagonal=-1)[:, :-1]
#     mat0 += torch.triu(mat, diagonal=1)[:, 1:]
#     mat1 = torch.cat([mat0,mat],dim=1)
#     mat2 = torch.cat([mat,mat0],dim=1)
#     return mat1, mat2


##############################################################################
##  Different ways of generating time lags
##############################################################################
def compute_metric_learning(x):
    # x: (B, T, dimension of metric learning)

    batch_size, seq_len, feature_dim = x.size()
    distance_matrix = torch.zeros((batch_size, seq_len, seq_len), device=x.device, dtype=x.dtype)#B,T,T

    model = SPDLayer(feature_dim, feature_dim).to(x.device) #Chlosky Decomposition

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

def timelag_dynamic(x,sigma=1):
    matrix = compute_metric_learning(x)
    matrix = 2 / (1 + torch.exp(matrix * sigma))
    matrix = torch.where(matrix < 1e-6, 0, matrix)  # set very small values to 0


    return matrix

def timelag_sigmoid(T,sigma=1):
    dist = np.arange(T)
    dist = np.abs(dist - dist[:, np.newaxis])
    matrix = 2 / (1 +np.exp(dist*sigma))
    matrix = np.where(matrix < 1e-6, 0, matrix)  # set very small values to 0
    return matrix

def timelag_cosine(x, sigma=1):
    batch_size, seq_len, feature_dim = x.size()
    matrix = torch.zeros((batch_size, seq_len, seq_len), device=x.device, dtype=x.dtype)#B,T,T

    for b in range(batch_size):

        x_batch = x[b].cpu().detach().numpy()
        similarity_matrix = cosine_similarity(x_batch)
        matrix[b] = torch.tensor(similarity_matrix).to(x.device)
    matrix = torch.where(matrix < 1e-6, torch.zeros_like(matrix), matrix)

    return matrix

def timelag_dtw(x, sigma=1):

    from scipy.spatial.distance import cdist
    batch_size, seq_len, feature_dim = x.size()
    matrix = torch.zeros((batch_size, seq_len, seq_len))  # B, T, T

    for b in range(batch_size):
        x_b = x[b].cpu().detach().numpy()

        distances = cdist(x_b, x_b, metric='euclidean')

        matrix[b] = torch.tensor(distances)
    matrix = torch.where(matrix < 1e-6, 0, matrix)
    matrix = 2 / (1 + torch.exp(matrix * sigma))
    return matrix



def inst_CL_hard(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temp_CL_hard(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

class TS2Vec(nn.Module):
    '''The TS2Vec model'''
    
    def __init__(
        self,
        input_dims, output_dims=320, hidden_dims=128,
        soft_instance=False,
        soft_temporal=True,
        depth=10, device=None,
        lambda_ = 0.5, tau_temp = 1,
        temporal_unit=0,

    ):
        
        super().__init__()
        self.device = device
        self.tau_temp = tau_temp
        self.lambda_ = lambda_
        self.temporal_unit = temporal_unit
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        self.soft_instance = soft_instance
        self.soft_temporal = soft_temporal
        self.SPD = SPDLayer(128, 128)
    
    def fit(self, train_data,z,soft_labels):
        ''' Training the TS2Vec model.

        Args:
            train_
            data shape:(B,T,N)
            z shape:(B,T,D)
        '''
        ts_l = train_data.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=train_data.size(0))
        x_left = take_per_row(train_data, crop_offset + crop_eleft, crop_right - crop_eleft)
        x_right = take_per_row(train_data, crop_offset + crop_left, crop_eright - crop_left)#cropping



        left_start = crop_offset + crop_eleft
        left_end = left_start + (crop_right - crop_eleft)

        right_start = crop_offset + crop_left
        right_end = right_start + (crop_eright - crop_left)

        overlap_start = np.maximum(left_start, right_start)#the start of overlap: crop_offset + crop_left
        overlap_end = np.minimum(left_end, right_end)#the end of overlap: crop_offset + crop_right

        z_overlap=take_per_row(z, overlap_start, crop_l)

        out1_all = self._net(x_left)
        out2_all = self._net(x_right)
        out = self._net(train_data)#lantent encoding
        out1 = out1_all[:, -crop_l:]
        out2 = out2_all[:, :crop_l]


        temporal_hierarchy=True
        lambda_ = self.lambda_
        tau_temp = self.tau_temp
        temporal_unit = self.temporal_unit
        soft_temporal = self.soft_temporal
        soft_instance = self.soft_instance
        z1 = out1
        z2 = out2

        if soft_labels is not None:#
            soft_labels = torch.tensor(soft_labels, device=z1.device)
            soft_labels_L, soft_labels_R = dup_matrix(soft_labels)


        total_loss = torch.tensor(0., device=z1.device)
        d = 0
        while z1.size(1) > 1:
            if lambda_ != 0:
                if soft_instance:
                    total_loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
                else:
                    total_loss += lambda_ * inst_CL_hard(z1, z2)
            if d >= temporal_unit:
                if 1 - lambda_ != 0:
                    if soft_temporal:
                        if temporal_hierarchy:
                            timelag = timelag_dynamic(z_overlap, tau_temp * (2 ** d))#Adaptive Distance Computation
                        else:
                            timelag = timelag_cosine(z_overlap, tau_temp)
                        timelag = timelag.clone().detach().requires_grad_(True).to(z1.device)
                        timelag_L, timelag_R = dup_matrix(timelag)

                        total_loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                    else:
                        total_loss += (1 - lambda_) * temp_CL_hard(z1, z2)
            d += 1
            z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z_overlap = F.max_pool1d(z_overlap.transpose(1, 2), kernel_size=2).transpose(1, 2)

        if z1.size(1) == 1:
            if lambda_ != 0:
                if soft_instance:
                    total_loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
                else:
                    total_loss += lambda_ * inst_CL_hard(z1, z2)
            d += 1
        total_loss=total_loss/d

        return total_loss,out




    
