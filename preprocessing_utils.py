from pathlib import Path
import torch
import numpy as np
import meshio
from typing import List
from tqdm import tqdm
from mesh_handler import xdmf_to_meshes

CWD = Path.cwd()

def torch_input_nodes(mesh : meshio.Mesh) -> torch.Tensor:
    nodes_xyz = mesh.points
    nodes_v = mesh.point_data["Vitesse"]
    nodes_P = mesh.point_data["Pression"]
    nodes_all = torch.tensor(np.hstack((nodes_xyz, nodes_v, np.expand_dims(nodes_P, axis=1))), dtype= torch.float)
    return nodes_all

def torch_input_edges(mesh : meshio.Mesh) -> torch.Tensor:
    if "tetra" in mesh.cells_dict:
        tetrahedrons = mesh.cells_dict["tetra"]
    else:
        raise ValueError("Le maillage ne contient pas de tétraèdres.")
    edges = np.vstack([
        tetrahedrons[:, [0, 1]],  # Arête entre sommets 0 et 1
        tetrahedrons[:, [0, 2]],  # Arête entre sommets 0 et 2
        tetrahedrons[:, [0, 3]],  # Arête entre sommets 0 et 3
        tetrahedrons[:, [1, 2]],  # Arête entre sommets 1 et 2
        tetrahedrons[:, [1, 3]],  # Arête entre sommets 1 et 3
        tetrahedrons[:, [2, 3]]   # Arête entre sommets 2 et 3
    ])
    # Suppression des doublons et tri des indices pour éviter les arêtes en double
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    # Convertir en format edge_index pour PyTorch Geometric (format [2, num_edges])
    edge_index = torch.tensor(edges.T, dtype=torch.int64)
    return edge_index

def raw_to_torch(meshes : List[meshio.Mesh]) -> List[torch.Tensor]:
    """
    Transform a list of meshio.Mesh into PyTorch Geometric Data compatible format.
    Returns:
    X_nodes : torch.Tensor
        The input features for each node at each time step, shape [num_timesteps-1, num_nodes, num_features]
    X_edges : torch.Tensor
        The edges of the mesh, shape [2, num_edges]
    Y : torch.Tensor
        The output features for each node at each time step, shape [num_timesteps-1, num_nodes, num_output_features]
    """
    N_meshes = len(meshes)
    X_edges = torch_input_edges(meshes[0])   # [2, num_edges]   invariant of time
    X_nodes = []
    for t in range(0,N_meshes):
        mesh = meshes[t]   #mesh at t
        nodes = torch_input_nodes(mesh)
        X_nodes.append(nodes)     # input for t to t+1
    X_nodes = torch.stack(X_nodes, dim=0)  # [num_timesteps-1, num_nodes, num_features]  at t
    return X_nodes, X_edges

def get_X_y(mesh_id: str, time_step: int) -> torch.Tensor:
    """
    Retrieve the node features, edges, and output features at a specific timestep for a given mesh.

    Parameters:
    mesh_id (str): Identifier for the mesh file to load.
    time_step (int): The timestep for which to retrieve the data.

    Returns:
    tuple: A tuple containing:
        - X_nodes (torch.Tensor): The input features of the nodes at the specified timestep.
        - X_edges (torch.Tensor): The edges of the mesh, invariant of the timestep.
        - Y (torch.Tensor): The output features for each node at the next timestep.
    """
    data = torch.load(CWD / f"data_cleaned/mesh_{mesh_id}.pth")
    X_nodes = data['nodes']
    X_edges = data['edges']
    try:
        y = X_nodes[time_step + 1][:,-4:]
        return X_nodes[time_step], X_edges, y
    except IndexError:
        print(f"Time step {time_step} is out of range for mesh {mesh_id}.")
    

def compute_edge_weights(edge_index, X):
    # Extraire les indices des nœuds source et cible
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    # Obtenir les coordonnées des nœuds source et cible
    source_coords = X[source_nodes,:3]  # Coordonnées (x, y, z) des nœuds source
    target_coords = X[target_nodes,:3]  # Coordonnées (x, y, z) des nœuds cible

    # Calcul de la distance euclidienne entre les nœuds source et cible
    distances = torch.norm(source_coords - target_coords, dim=1)

    # Éviter la division par zéro en ajoutant une petite valeur epsilon
    epsilon = 1e-8
    edge_weights = 1.0 / (distances + epsilon)

    return edge_weights

if __name__ == "__main__":
    folder_path = CWD / "4Students_AnXplore03"
    xdmf_files = list(folder_path.glob("*.xdmf"))
    out_dir = CWD / "data_cleaned"
    if not out_dir.exists():
        out_dir.mkdir(exist_ok=True)
    for xdmf_fp in tqdm(xdmf_files):
        extension = xdmf_fp.name.split("_")[-1]
        mesh_id = extension[:-5]
        new_filepath = out_dir / f"mesh_{mesh_id}.pth"
        meshes = xdmf_to_meshes(str(xdmf_fp), verbose=False)
        X_nodes, X_edges = raw_to_torch(meshes)
        torch.save({'nodes': X_nodes, 'edges': X_edges}, new_filepath)
        #torch.load(new_filepath)