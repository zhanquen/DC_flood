from sklearn.model_selection import train_test_split
import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from preprocessing_utils import get_X_y, compute_edge_weights
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
    train_ids, test_ids = train_test_split(mesh_ids, test_size=test_size, random_state=random_state)
    return train_ids, test_ids


def train_model(model, optimizer, mesh_ids, epochs=1000, exclude_xyz = True):
    loss_array = []
    for epoch in range(epochs):
        mesh_id = random.choice(mesh_ids)
        time_step = random.randint(0, 70)
        X_nodes, X_edges, y = get_X_y(mesh_id, time_step=time_step)
        if exclude_xyz:
            X_nodes = X_nodes[:,3:] # Remove the coordinates
        edge_weights = compute_edge_weights(X_edges, X_nodes)
        optimizer.zero_grad()
        out = model(X_nodes, X_edges, edge_weights)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        loss_array.append(loss.item())
        print(f'Epoch {epoch}, Loss: {loss.item()}')
    return loss_array

# Train the model

def evaluate_model(model, mesh_ids):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for mesh_id in mesh_ids:
            for time_step in range(70):  # Assuming there are 70 time steps
                X_nodes, X_edges, y = get_X_y(mesh_id, time_step=time_step)
                X_nodes_input = X_nodes[:, 3:]  # Remove the coordinates
                edge_weights = compute_edge_weights(X_edges, X_nodes_input)
                y_pred = model(X_nodes_input, X_edges, edge_weights)
                loss = F.mse_loss(y_pred, y)
                total_loss += loss.item()
    
    average_loss = total_loss / (len(mesh_ids) * 70)
    return average_loss