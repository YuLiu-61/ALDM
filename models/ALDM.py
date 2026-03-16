import numpy as np
#%%
import torch.nn as nn
import torch.nn.functional as F
from models.NF import MAF
import torch
from models.soft_ts2vec import TS2Vec


def interpolate(tensor, index, target_size, mode = 'nearest', dim = 0):
    print(tensor.shape)
    source_length = tensor.shape[dim]
    if source_length > target_size:
        raise AttributeError('no need to interpolate')
    if dim == -1:
        new_tensor = torch.zeros((*tensor.shape[:-1], target_size),dtype=tensor.dtype, device=tensor.device)
    if dim == 0:
        new_tensor = torch.zeros((target_size, *tensor.shape[1:], ),dtype=tensor.dtype, device=tensor.device)
    scale = target_size // source_length
    reset = target_size % source_length
    # if mode == 'nearest':
    new_index = index
    new_tensor[new_index, :] = tensor
    new_tensor[:new_index[0], :] = tensor[0,:].unsqueeze(0)
    for i in range(source_length-1):
        new_tensor[new_index[i]:new_index[i+1] , :] = tensor[i,:].unsqueeze(0)
    new_tensor[new_index[i+1] :,:] = tensor[i+1,:].unsqueeze(0)
    return new_tensor


class GNN(nn.Module):
    """
    The GNN module applied in GANF
    """
    def __init__(self, input_size, hidden_size):

        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, A):
        ## A: K X K
        ## H: N X K  X L X D
        # print(h.shape, A.shape)
        # h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,A))
        # h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,A))
        # print(h.shape, A.shape)
        h_n = self.lin_n(torch.einsum('nkld,nkj->njld',h,A))
        h_r = self.lin_r(h[:,:,:-1])
        h_n[:,:,1:] += h_r
        h = self.lin_2(F.relu(h_n))

        return h

import math
import torch.nn as nn
import matplotlib.pyplot as plt
def plot_attention(data, i, X_label=None, Y_label=None):
  '''
    Plot the attention model heatmap
    Args:
      data: attn_matrix with shape [ty, tx], cutted before 'PAD'
      X_label: list of size tx, encoder tags
      Y_label: list of size ty, decoder tags
  '''
  fig, ax = plt.subplots(figsize=(20, 8)) # set figure size
  heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)
  fig.colorbar(heatmap)
  # Set axis labels
  if X_label != None and Y_label != None:
    X_label = [x_label for x_label in X_label]
    Y_label = [y_label for y_label in Y_label]
    
    xticks = range(0,len(X_label))
    ax.set_xticks(xticks, minor=False) # major ticks
    ax.set_xticklabels(X_label, minor = False, rotation=45)   # labels should be 'unicode'
    
    yticks = range(0,len(Y_label))
    ax.set_yticks(yticks, minor=False)
    ax.set_yticklabels(Y_label[::-1], minor = False)   # labels should be 'unicode'
    
    ax.grid(True)
    # plt.show()
    plt.savefig('attention{:04d}.jpg'.format(i))



class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, c):
        super(ScaleDotProductAttention, self).__init__()
        self.w_q = nn.Linear(c, c)
        self.w_k = nn.Linear(c, c)
        self.w_v = nn.Linear(c, c)
        self.softmax = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(0.2)
        # swat_0.2
    def forward(self, x,mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        shape = x.shape
        x_shape = x.reshape((shape[0],shape[1], -1))
        batch_size, length, c = x_shape.size()
        q = self.w_q(x_shape)
        k = self.w_k(x_shape)
        k_t = k.view(batch_size, c, length)  # transpose
        score = (q @ k_t) / math.sqrt(c)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        # 3. pass them softmax to make [0, 1] range
        score = self.dropout(self.softmax(score))



        return score, k


class ALDM(nn.Module):

    def __init__ (self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, outdim_scl,hiddendim_scl,soft_instance,soft_temporal,dim_ml,dropout = 0.1, model="MAF", batch_norm=True):
        super(ALDM, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,hidden_size=hidden_size,batch_first=True, dropout=dropout)
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        flow_model = str(model).upper()
        if flow_model == "ALDM":
            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh')
        else:
            raise ValueError(f"Unsupported flow model: {model}")
        self.scl=TS2Vec(input_dims=n_sensor, output_dims=outdim_scl, hidden_dims=hiddendim_scl,soft_instance=soft_instance,soft_temporal=soft_temporal, depth=10)
        self.attention = ScaleDotProductAttention(window_size*input_size)
        self.linear=nn.Linear(outdim_scl,n_sensor)
        self.linear_scl=nn.Linear(hidden_size*n_sensor,dim_ml)
    def forward(self, x, ):
        y,loss_scl=self.test(x, )
        y=y.mean()
        loss=-y+0.01*loss_scl
        return loss

    def test(self, x, ):
        # x: B,N,T,1
        B,N,T,_ = x.shape
        graph,_ = self.attention(x)#context learning module
        self.graph = graph

        x_for_rnn=x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x_for_rnn)
        h = h.reshape((B, N, h.shape[1], h.shape[2]))

        z_for_scl=h.permute(0,2,1,3).reshape((B,T,-1))
        z_for_scl=self.linear_scl(z_for_scl)#the input of Adaptive Distance Computation(B,T,D,)

        h = self.gcn(h, graph)
        h = h.reshape((-1,h.shape[3]))#spatio-temporal conditions


        x_for_scl=x.squeeze(-1).permute(0,2,1)
        loss_scl,out=self.scl.fit(x_for_scl,z_for_scl,soft_labels=None)#Latent Encoding
        out=self.linear(out).permute(0,2,1).unsqueeze(-1)

        full_shape = out.shape
        out = out.reshape((-1,full_shape[3]))
        log_prob = self.nf.log_prob(out, full_shape[1], full_shape[2], h)#Distribution Projection(Conditional Normalizing Flow)
        log_prob=log_prob.reshape([full_shape[0],-1])
        log_prob = log_prob.mean(dim=1)

        return log_prob, loss_scl, out

    def get_graph(self):
        return self.graph
