from sklearn.model_selection import train_test_split
import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from preprocessing_utils import get_X_y, compute_edge_weights
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

def train_test_sets(mesh_ids, test_size=0.2, random_state=42):
    """
    Divise une liste de mesh_ids en ensembles d'entraînement et de test.

    Args:
        mesh_ids (list): Liste des identifiants de maillage.
        test_size (float): Proportion de données à utiliser pour le test (0.2 = 20%).
        random_state (int): Graine pour garantir la reproductibilité.

    Returns:
        tuple: (train_ids, test_ids) - Deux listes contenant les identifiants respectifs.
    """
    # Fixer la graine pour la reproductibilité
    torch.manual_seed(random_state)

    # Calculer la taille des ensembles
    total_size = len(mesh_ids)
    test_size_count = int(total_size * test_size)
    train_size_count = total_size - test_size_count

    # Mélanger et diviser les données de manière reproductible
    train_ids, test_ids = torch.utils.data.random_split(
        mesh_ids, [train_size_count, test_size_count]
    )

    # Convertir en listes Python pour une utilisation facile
    train_ids = list(train_ids)
    test_ids = list(test_ids)

    return train_ids, test_ids

class MeshDataset(Dataset):
    def __init__(self, mesh_ids, features_method = get_X_y):
        self.mesh_ids = mesh_ids
        self.features_method = features_method
    def __len__(self):
        return len(self.mesh_ids) * 79  # 79 time steps par mesh_id

    def __getitem__(self, idx):
        mesh_idx = idx // 79
        time_step = idx % 79
        mesh_id = self.mesh_ids[mesh_idx]
        
        X_nodes, X_edges, y = self.features_method(mesh_id, time_step)
        edge_weights = compute_edge_weights(X_edges, X_nodes)
        
        return X_nodes, X_edges, edge_weights, y




from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Vérification de la disponibilité de MPS ou bascule sur CPU

from collections import deque
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

def train_model(model, optimizer, mesh_ids, epochs=1000, batch_size=1, features_method=get_X_y, device="cpu",
                save = False, **kwargs):
    model.to(device)  # Déplacer le modèle sur le GPU MPS
    dataset = MeshDataset(mesh_ids, features_method=features_method) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_array = []
    sliding_window = deque(maxlen=100)  # File d'attente pour la moyenne glissante

    for epoch in range(epochs):
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for X_nodes, X_edges, edge_weights, y in dataloader:
                # Déplacer les données sur MPS
                X_nodes = X_nodes.to(device)
                X_edges = X_edges.to(device).squeeze()
                edge_weights = edge_weights.to(device).squeeze()
                y = y.to(device)
                
                optimizer.zero_grad()
                out = model(X_nodes, X_edges, edge_weights)
                loss = F.mse_loss(out, y)
                loss.backward()
                optimizer.step()

                loss_array.append(loss.item())
                sliding_window.append(loss.item())
                
                # Calculer la moyenne glissante
                sliding_mean = sum(sliding_window) / len(sliding_window)
                
                pbar.set_postfix(loss=sliding_mean)
                pbar.update(1)

        # Libérer la mémoire MPS après chaque epoch
        torch.mps.empty_cache()

    return loss_array

def evaluate_model(model, mesh_ids, batch_size=1, features_method=get_X_y,device = "cpu", **kwargs):
    model.to(device)  # Déplacer le modèle sur le GPU MPS
    model.eval()
    dataset = MeshDataset(mesh_ids, features_method=features_method)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0
    with torch.no_grad():
        for X_nodes, X_edges, edge_weights, y in dataloader:
            # Déplacer les données sur MPS
            X_nodes = X_nodes.to(device)
            X_edges = X_edges.to(device).squeeze()
            edge_weights = edge_weights.to(device).squeeze()
            y = y.to(device)

            y_pred = model(X_nodes, X_edges, edge_weights)
            loss = F.mse_loss(y_pred, y)
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss
