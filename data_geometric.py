from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from pathlib import Path
import meshio
from preprocessing_utils import get_X_y_with_inflow, get_X_y_acc_type, get_edges_dir_info
from DL_utils import train_test_sets
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt



def transform_acc_type(mesh_id, time_step, data_dir):
    nodes_features, edges_index , y = get_X_y_acc_type(mesh_id, time_step, data_dir)
    data = Data(x=nodes_features, edge_index=edges_index, edge_attr=None, y=y, pos=nodes_features[:, :3])
    return data

def transform_acc_type_dir(mesh_id, time_step, data_dir):
    nodes_features, edges_index , y = get_X_y_with_inflow(mesh_id, time_step, data_dir, replace_inflow=True)
    edges_index_dir, edge_attr = get_edges_dir_info(nodes_features, edges_index)
    data = Data(x=nodes_features[:,3:], edge_index=edges_index_dir, edge_attr=edge_attr, y=y, pos=nodes_features[:, :3])
    return data

def transform_with_inflow(mesh_id, time_step, data_dir, replace=True):
    nodes_features, edges_index , y = get_X_y_with_inflow(mesh_id, time_step, data_dir, replace_inflow=replace)
    data = Data(x=nodes_features, edge_index=edges_index, edge_attr=None, y=y, pos=nodes_features[:, :3])
    return data

class MeshDataset_(Dataset):
    def __init__(self, src_data_dir, mesh_ids, n_times, offset=1, func_to_data=transform_acc_type):
        self.data_dir = src_data_dir
        self.mesh_ids = mesh_ids
        self.n_times = n_times
        self.offset = offset
        self.n_X_y = self.n_times - self.offset - 1
        self.func_to_data = func_to_data
    
    def __len__(self):
        return len(self.mesh_ids) * self.n_X_y

    def __getitem__(self, idx):
        mesh_idx = idx // self.n_X_y
        time_step = self.offset + idx % self.n_X_y
        mesh_id = self.mesh_ids[mesh_idx]
        data_sample = self.func_to_data(mesh_id, time_step=time_step, data_dir=self.data_dir)
        return data_sample


#example
#dataset_dir = MeshDataset_(data_dir, train_ids, n_times=80, offset=1, func_to_data=transform_acc_type_dir)
#loader = DataLoader(dataset_dir, batch_size=1, shuffle=True)
#
#dataset_dir_test = MeshDataset_(data_dir, test_ids, n_times=80, offset=1, func_to_data=transform_acc_type_dir)
#test_loader = DataLoader(dataset_dir_test, batch_size=1, shuffle=False)