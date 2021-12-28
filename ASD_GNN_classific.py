#!/usr/bin/env python
# coding: utf-8

import sys
print(sys.version) # Check Pyton version

from os.path import join, exists, dirname, basename
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl
import xlrd
import scipy as sc
from glob import glob
from brainspace import gradient
from torch_geometric.data import Dataset
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, OneHotEncoder
from torch_geometric.utils import from_networkx
from networkx.convert_matrix import from_numpy_matrix
from sklearn.metrics import f1_score, classification_report
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import TopKPooling, GCNConv, EdgeConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_scatter import scatter
import nilearn
from nilearn.connectome import ConnectivityMeasure, sym_matrix_to_vec, vec_to_sym_matrix
from brainspace import gradient
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy import sparse
from torch_geometric.utils import from_networkx
from networkx.convert_matrix import from_numpy_matrix

print(torch.cuda.get_device_name(0)) 
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.__version__)
print(torch.cuda.current_device())
print(torch.cuda.device_count())

GPU_NUM = 2 # Select GPU number

device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU

print ('Current cuda device ', torch.cuda.current_device()) # check


# # Data definition
path_data = 'path name'

demo = pd.read_excel(join(path_data, 'data', 'Phenotypic.xlsx'), sheet_name='surf_n=211', skiprows=0)

sub_list = demo['FILE_ID']
label = demo['DX_GROUP']
site_id = demo['SITE_ID']
site_label = demo['SITE_Label']
Age = demo['AGE_AT_SCAN']
FD = demo['func_mean_fd']
IQ = demo['FIQ']
eye_status = demo['EYE_STATUS_AT_SCAN']
med_status = demo['CURRENT_MED_STATUS']
comorbidity = demo['COMORBIDITY']
comorbidity_bool = demo['COMORBIDITY_BOOL']

ASD_index = np.where(label == 1)[0]                
TD_index = np.where(label == 2)[0]
Total_index = np.concatenate((ASD_index,TD_index)) 
sorted_idx = np.concatenate((ASD_index,TD_index), axis = 0)

site_label_sorted = np.array(site_label)[sorted_idx]


y_target = np.array(label)-1


file_list = [join(path_data,'data/ABIDE_1',i) for i in sub_list]
FC_meanhar_pear_list = [np.load(join(f,'surf_conn_mat_Mean_har_KS.npy')) for f in file_list]
FC_meanhar_par_list = [np.load(join(f,'surf_par_conn_mat_Mean_har.npy')) for f in file_list]

FC_z_meanhar_pear_list = [np.arctanh(m) for m in FC_meanhar_pear_list]
FC_z_meanhar_par_list = [np.arctanh(m) for m in FC_meanhar_par_list]


# nan check
[np.isnan(i).sum() for i in FC_z_meanhar_pear_list]
[i.shape for i in FC_meanhar_par_list]



FC_adjmat_list = []

for i, FC in enumerate(FC_meanhar_par_list):
    print(i+1, '', end='', flush = True)
    index = np.abs(FC).argsort(axis=1)
    n_rois = FC.shape[0]
    dumy_FC = np.zeros((n_rois, n_rois))

    # Take only the top k correlates to reduce number of edges
    for j in range(n_rois):
        for k in range(n_rois - 10):
            dumy_FC[j, index[j, k]] = 0
        for k in range(n_rois - 10, n_rois):
            dumy_FC[j, index[j, k]] = 1
            
    FC_adjmat_list.append(dumy_FC)



gradient.is_symmetric(FC_meanhar_par_list[0])


FC_pear_vec_list = []
g_data_list = []

for i in range(len(file_list)):
    print(f'{i+1}',' ', end='', flush=True)
    
    # node feature (FC pearson coefficient)
    FC_pear = FC_meanhar_pear_list[i]
    FC_pear_vec = sym_matrix_to_vec(FC_pear, discard_diagonal = False)
    FC_pear_vec_list.append(FC_pear_vec)
    
    pcorr_matrix_nx = from_numpy_matrix(FC_adjmat_list[i])
    g_data = from_networkx(pcorr_matrix_nx)
    
    n_feature = FC_pear
    
    g_data.x = torch.tensor(n_feature).float()
    g_data.y = torch.tensor(y_target[i], dtype=torch.long)
    
    g_data_list.append(g_data)


# # Dataloader
x_data = g_data_list
num_node_features = g_data_list[0].num_node_features

nsbj = len(x_data)

list_idx = np.random.permutation(np.arange(nsbj))
itrain_idx = list_idx[:int(nsbj*0.6)]         # 100 subjects for training
itest_idx = list_idx[int(nsbj*0.6):]       # 10  subjects for validation
otest_idx = list_idx[int(nsbj*0.6):]          # left subjects for test

x_train = [x_data[i] for i in itrain_idx]
x_valid = [x_data[i] for i in itest_idx]
x_test = [x_data[i] for i in otest_idx]

bs_train = 16
bs_valid = 16
bs_test = 16

train_loader = DataLoader(x_train, batch_size=bs_train)
valid_loader = DataLoader(x_valid, batch_size=bs_valid)
test_loader = DataLoader(x_test, batch_size=bs_test)

print(f'train : {len(itrain_idx)}, valid : {len(itest_idx)}, test : {len(otest_idx)}')


# # Model construction
embed_dim = num_node_features
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Initialize MLPs used by EdgeConv layers
        self.mlp1 = nn.Sequential(torch.nn.Linear(2*embed_dim, 256), torch.nn.ReLU())
        self.mlp2 = nn.Sequential(torch.nn.Linear(2*256, 128), torch.nn.ReLU())
        self.mlp3 = nn.Sequential(torch.nn.Linear(2*128, 64), torch.nn.ReLU())
        
        self.conv1 = EdgeConv(self.mlp1, aggr='max')
        self.conv2 = EdgeConv(self.mlp2, aggr='max')
        self.conv3 = EdgeConv(self.mlp3, aggr='max')

        self.lin1 = torch.nn.Linear(64, 2)
        self.lin2 = torch.nn.Linear(32, 2)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
  
    def forward(self, data):
        # 1. Graph Convolution Layer
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
#         print(x.size())
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
#         x = F.dropout(x, p = 0.5, training = self.training)
#         print(x.size())
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
#         print(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
#         x = F.relu(x)
             
        # 2. Read out layer
    
        x = gap(x, batch) # gap(x, batch) torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
#         print(x)


        # 3. Fully-Connected lyaer (for graph classification/regression)
        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.lin1(x)
        x = F.softmax(x, dim=1)
#         x = self.act1(x)
#         x = self.lin2(x)
#         print(x)

        return x, x


def train(model):
    model.train()
    
    read_out_list = []
    y_pred = []
    y_true = []
    
    correct = 0
    loss_all = 0
    i =1
    for data in train_loader:
#         print(i, '', end='', flush=True)
        i +=1
        data = data.to(device)
        optimizer.zero_grad()
        output, read_out = model(data)
        label = data.y.to(device)
#         print(output, label)

        pred = output.argmax(dim=1)
        
        correct += int((pred == data.y).sum())
            
        read_out_list.append(read_out.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        y_true.append(label.detach().cpu().numpy())
        
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
        
    loss_train = loss_all / len(train_loader.dataset)
    acc = correct / len(train_loader.dataset)

    print(f'Train loss : {loss_train:.4f}    ', end='')

    loss_valid, y_valid_pred, y_valid_true, valid_acc = evaluate(valid_loader)
    print(f'Valid loss : {loss_valid:.4f}    ', end='')
    print(f'Train acc : {np.round(acc,2)}    Valid acc : {np.round(valid_acc,2)}')
        
    return loss_train, read_out_list, y_pred, y_true, loss_valid, y_valid_pred, y_valid_true

def evaluate(loader):
    model.eval()

    y_valid_pred = []
    y_valid_true = []
    
    correct = 0
    loss_all = 0

    with torch.no_grad():
        for data in loader:

            data = data.to(device)
            output, _ = model(data)
            label = data.y.to(device)
            
            pred = output.argmax(dim=1).to(device)
            correct += int((pred == data.y).sum())
            
            y_valid_pred.append(pred.detach().cpu().numpy())
            y_valid_true.append(label.detach().cpu().numpy())
            
            loss = crit(output, label)
            loss_all += data.num_graphs * loss.item()   
        acc = correct / len(loader.dataset)
            
    return loss_all / len(loader.dataset), y_valid_pred, y_valid_true, acc

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay = 0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
crit = torch.nn.CrossEntropyLoss() # MSELoss L1Loss nn.CrossEntropyLoss()
# FCE loss= 1*(1-torch.exp(-ce))**5*ce

best_val_loss = float('inf')
best_model = None

loss_train_list = []
loss_valid_list = []

num_epochs = 500
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} of {num_epochs}    ", end='')
    loss_train, read_out_list, y_pred, y_true, loss_valid, y_valid_pred, y_valid_true = train(model)
    loss_train_list.append(loss_train)
    loss_valid_list.append(loss_valid)
    

    if loss_valid < best_val_loss:
        print('[Updated New Record!]')
        best_val_loss = loss_valid
        best_model = model
        best_y_pred = y_pred
        best_y_valid_pred = y_valid_pred
        best_epoch = epoch
    if epoch > best_epoch +20:
        print('Stop!')
        break
    
    scheduler.step()


def loss_plot(x, y, label_x, label_y):
    plt.figure(1, (7,5))
    plt.plot(x, label = label_x, color= 'k', linewidth=5)
    plt.plot(y, label = label_y, color = 'gray', linewidth=5)
    plt.xlabel('Epochs', fontsize=15)
    plt.xticks(fontsize=15)
#     plt.ylim([0.4,0.71])
    plt.yticks(fontsize=15)
    plt.legend(fontsize = 15)


x = loss_train_list
y = loss_valid_list

label_x = 'Train loss'
label_y = 'Valid loss'

loss_plot(x, y, label_x, label_y)



y_pred_total = np.array([j for i in y_pred for j in i])
y_true_total = np.array([j for i in y_true for j in i])

y_valid_pred_total = np.array([j for i in y_valid_pred for j in i])
y_valid_true_total = np.array([j for i in y_valid_true for j in i])


print(y_true_total.shape, y_valid_pred_total.shape)


acc_train = (y_pred_total == y_true_total).sum() / len(train_loader.dataset)
acc_valid = (y_valid_pred_total == y_valid_true_total).sum() / len(valid_loader.dataset)

print(acc_train)
print(acc_valid)




print(classification_report(y_true_total, y_pred_total))
print(classification_report(y_valid_true_total, y_valid_pred_total))