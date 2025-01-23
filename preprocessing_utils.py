from pathlib import Path
import torch
import numpy as np
import meshio
from typing import List
from tqdm import tqdm
from mesh_handler import xdmf_to_meshes

CWD = Path.cwd()
DATA_DIR = CWD / "data_cleaned"

def torch_input_nodes(mesh: meshio.Mesh) -> torch.Tensor:
    """
    Converts mesh node data into a PyTorch tensor.

    Parameters:
    mesh (meshio.Mesh): A meshio Mesh object containing the mesh data.

    Returns:
    torch.Tensor: A tensor containing the concatenated node coordinates, velocity, and pressure data.
    """
    nodes_xyz = mesh.points
    nodes_v = mesh.point_data["Vitesse"]
    nodes_P = mesh.point_data["Pression"]
    nodes_all = torch.tensor(np.hstack((nodes_xyz, nodes_v, np.expand_dims(nodes_P, axis=1))), dtype= torch.float)
    return nodes_all

def torch_input_edges(mesh: meshio.Mesh) -> torch.Tensor:
    """
    Converts the tetrahedral elements of a mesh into edge indices suitable for use with PyTorch Geometric.

    Parameters:
    mesh (meshio.Mesh): The input mesh containing tetrahedral elements.

    Returns:
    torch.Tensor: A tensor of shape [2, num_edges] containing the edge indices.

    Raises:
    ValueError: If the mesh does not contain tetrahedral elements.
    """
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

def torch_to_meshes(vP_tensor: torch.Tensor, src_filepath) -> meshio.Mesh:
    """
    Converts PyTorch tensors back into a meshio Mesh object.

    Parameters:
    vP_tensor (torch.Tensor): A tensor containing the node velocity, and pressure data. (T, N, 4)
    src_filepath (str): The path to the source xdmf file.
    Returns:
    meshio.Mesh: A meshio Mesh object containing the mesh data.
    """
    meshes = xdmf_to_meshes(src_filepath, verbose=False)
    velocities = vP_tensor[:, 0:3]
    pressures = vP_tensor[:, 4]
    for i in range(len(meshes)):
        meshes[i].point_data = {"Vitesse": np.array(velocities[i]), "Pression": np.array(pressures[i])}
    return meshes

def get_X_y_acc_type(mesh_id: str, time_step: int, data_dir=DATA_DIR) -> torch.Tensor:
    """
    Extracts and processes node and edge data from a preprocessed mesh file for a given time step.
    Args:
        mesh_id (str): Identifier for the mesh file to load.
        time_step (int): The specific time step to extract data for.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            - X_nodes_t: Processed node features tensor for the given time step, including velocities, accelerations, time step, and wall mask.
            - X_edges: Edge features tensor.
            - y: Target tensor for the next time step, including velocities and additional features.
    """
    data = torch.load(data_dir / f"mesh_{mesh_id}.pth")
    X_nodes = data['nodes']
    #print(X_nodes.shape)
    X_nodes_t= X_nodes[time_step,:,:]
    vitesses= X_nodes[time_step,:,3:6]
    accelerations = torch.zeros_like(vitesses)
    accelerations[:,:] =  (X_nodes[time_step,:,3:6] -  X_nodes[time_step-1,:,3:6])/0.01
    
    time_steps=torch.full((X_nodes_t.shape[0], 1), time_step)
    walls_mask = torch.norm(X_nodes_t[:,3:6], p=2, dim=1, keepdim=True) > 1e-10
    X_nodes_t = torch.cat([X_nodes_t, accelerations, time_steps, walls_mask], dim=-1)
    #time_tensor = torch.full((X_nodes.shape[0], X_nodes.shape[1], 1), time_step)
    #X_nodes = torch.cat([X_nodes, time_tensor,accelerations], dim=-1)
    X_edges = data['edges']
    y = X_nodes[time_step + 1][:,3:7]
    return X_nodes_t, X_edges, y

def get_X_y_with_inflow(mesh_id: str, t: int, data_dir=DATA_DIR, replace_inflow=False) -> torch.Tensor:
    """
    Extracts and processes node and edge data from a preprocessed mesh file for a given time step.
    Args:
        mesh_id (str): Identifier for the mesh file to load.
        t (int): The specific time step to extract data for.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            - X_nodes_t: Processed node features tensor for the given time step, including velocities, accelerations, time step, and wall mask.
            - X_edges: Edge features tensor.
            - y: Target tensor for the next time step, including velocities and additional features.
    """
    data = torch.load(data_dir / f"mesh_{mesh_id}.pth")
    nodes_feat = data['nodes']
    edges_index = data['edges']
    X_init = nodes_feat[0,:,:]
    N_nodes = X_init.shape[0]

    X_nodes_now = nodes_feat[t,:,:]  #pos, speeds, P on col 0 to 6
    X_nodes_past = nodes_feat[t-1,:,:]
    X_nodes_future = nodes_feat[t+1,:,:]
    
    accelerations = torch.zeros((N_nodes, 3))  #acceleration column 7 to 9
    accelerations[:,:] =  (X_nodes_now[:,3:6] -  X_nodes_past[:,3:6])/0.01
    time_steps=torch.full((N_nodes, 1), t)   #time on column 10
    wall_mask = torch.norm(X_init[:,3:6], p=2, dim=1, keepdim=True) > 1e-10  #wall on column 11
    inflow_mask = ((X_init[:, 1] < 1e-2) & (X_init[:,4] > 0)).unsqueeze(-1)  #inflow on column 12

    if replace_inflow:
        X_nodes_replace = X_nodes_now.clone()
        X_nodes_replace[inflow_mask] = X_nodes_future[inflow_mask]
        X_nodes_t = torch.cat([X_nodes_replace, accelerations, time_steps, wall_mask, inflow_mask], dim=-1)
    else:
        X_nodes_t = torch.cat([X_nodes_now, accelerations, time_steps, wall_mask, inflow_mask], dim=-1)
    
    return X_nodes_t, edges_index, X_nodes_future[:,3:7]


def get_X_y_acc(mesh_id: str, time_step: int, data_dir=DATA_DIR) -> torch.Tensor:
    """
    Extracts node features, edge features, and target values for a given mesh and time step.
    Args:
        mesh_id (str): Identifier for the mesh.
        time_step (int): The specific time step to extract data from.
    Returns:
        tuple: A tuple containing:
            - X_nodes_t (torch.Tensor): Node features at the given time step, including velocities, accelerations, and time step.
            - X_edges (torch.Tensor): Edge features of the mesh.
            - y (torch.Tensor): Target values for the next time step.
    """
    data = torch.load(data_dir / f"mesh_{mesh_id}.pth")
    X_nodes = data['nodes']
    # print(X_nodes.shape)
    X_nodes_t= X_nodes[time_step,:,:]
    vitesses= X_nodes[time_step,:,3:6]
    accelerations = torch.zeros_like(vitesses)
    accelerations[:,:] =  (X_nodes[time_step,:,3:6] -  X_nodes[time_step-1,:,3:6])/0.01
    
    time_steps=torch.full((X_nodes_t.shape[0], 1), time_step)
    X_nodes_t = torch.cat([X_nodes_t, accelerations,time_steps ], dim=-1)
    #time_tensor = torch.full((X_nodes.shape[0], X_nodes.shape[1], 1), time_step)
    #X_nodes = torch.cat([X_nodes, time_tensor,accelerations], dim=-1)
    X_edges = data['edges']
    y = X_nodes[time_step + 1][:,3:7]
    return X_nodes_t, X_edges, y
   
def get_X_y(mesh_id: str, time_step: int, data_dir=DATA_DIR) -> torch.Tensor:
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
    data = torch.load(data_dir / f"mesh_{mesh_id}.pth")
    X_nodes = data['nodes']
    time_tensor = torch.full((X_nodes.shape[0], X_nodes.shape[1], 1), time_step)
    X_nodes = torch.cat([X_nodes, time_tensor], dim=-1)
    X_edges = data['edges']
    try:
        y = X_nodes[time_step + 1][:,3:7]
        return X_nodes[time_step], X_edges, y
    except IndexError:
        print(f"Time step {time_step} is out of range for mesh {mesh_id}.")
    

def compute_edge_weights(edge_index, X):
    """
    Compute the edge weights for a graph based on the Euclidean distance 
    between source and target nodes.
    Parameters:
    edge_index (torch.Tensor): A tensor of shape (2, num_edges) containing 
                               the indices of the source and target nodes 
                               for each edge.
    X (torch.Tensor): A tensor of shape (num_nodes, num_features) containing 
                      the features of each node, where the first three 
                      features are the (x, y, z) coordinates of the nodes.
    Returns:
    torch.Tensor: A tensor of shape (num_edges,) containing the computed 
                  edge weights, which are the inverse of the Euclidean 
                  distances between the source and target nodes.
    """
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

def create_edges_index_dir(X_edges_undir):
    """
    Create a directed edges index from an undirected edges index.

    This function takes an undirected edges index tensor and converts it into a directed edges index tensor by 
    concatenating the original undirected edges with their reversed counterparts.

    Parameters:
    X_edges_undir (torch.Tensor): A tensor of shape (2, N) representing N undirected edges, where each column 
                                  contains a pair of node indices representing an undirected edge.

    Returns:
    torch.Tensor: A tensor of shape (2, 2N) representing 2N directed edges, where the first N columns are the 
                  original undirected edges and the next N columns are the reversed edges.
    """
    X_edges_dir = torch.cat([X_edges_undir, X_edges_undir[[1, 0], :]], dim=1)
    return X_edges_dir

def create_edge_attributes(X_nodes, X_edges_dir):
    """
    Create edge attributes for a graph based on node positions and directed edges.

    Parameters:
    X_nodes (torch.Tensor): A tensor of shape (num_nodes, num_features) containing the positions of the nodes.
                            The first three columns are expected to be the x, y, z coordinates of the nodes.
    X_edges_dir (tuple of torch.Tensor): A tuple containing two tensors of shape (num_edges,) representing the 
                                         indices of the source and target nodes for each directed edge.

    Returns:
    torch.Tensor: A tensor of shape (num_edges, 4) containing the edge attributes. The first three columns 
                  represent the normalized direction vector from the source node to the target node, and the 
                  fourth column represents the distance between the nodes.
    """
    positions = X_nodes[:,0:3]
    i, j = X_edges_dir
    direction = positions[j,:] - positions[i,:]
    distance = torch.norm(direction, dim=1, keepdim=True)
    direction /= distance
    edge_attr = torch.cat([direction, distance], dim=1)
    return edge_attr

def get_edges_dir_info(X_nodes, X_edges_undir):
    """
    Generates directed edges information from undirected edges and node attributes.

    Parameters:
    X_nodes (array-like): The attributes of the nodes.
    X_edges_undir (array-like): The undirected edges.

    Returns:
    tuple: A tuple containing:
        - edges_index_dir (array-like): The indices of the directed edges.
        - edges_attr (array-like): The attributes of the directed edges.
    """
    edges_index_dir = create_edges_index_dir(X_edges_undir)
    edges_attr = create_edge_attributes(X_nodes, edges_index_dir)
    return edges_index_dir, edges_attr

def convert_xdmf_to_torch(folder_path, out_dir):
    xdmf_files = list(folder_path.glob("*.xdmf"))
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

def get_inflow_idx(y_tsr, v_y_tsr):    
    """
    Identify the indices of inflow points in the given tensors.

    Parameters:
    y_tsr (torch.Tensor): A tensor representing some quantity, where inflow is defined as values less than 1e-2.
    v_y_tsr (torch.Tensor): A tensor representing velocity, where inflow is defined as values greater than 0.

    Returns:
    torch.Tensor: A tensor containing the indices of the inflow points.
    """
    inflow = (y_tsr < 1e-2) & (v_y_tsr > 0)
    inflow_idx = torch.where(inflow)[0]
    return inflow_idx

def get_wall_idx(speeds_tsr):    
    """
    Identify the indices of wall (border) elements in a tensor of speeds.

    Args:
        speeds_tsr (torch.Tensor): A tensor containing speed vectors.

    Returns:
        torch.Tensor: A tensor containing the indices of elements where the speed is effectively zero (considered as walls).
    """
    velocities = torch.norm(speeds_tsr, p=2, dim=1)
    border = velocities < 1e-10
    border_idx = torch.where(border)[0]
    return border_idx

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