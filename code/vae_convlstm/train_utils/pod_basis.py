import numpy as np
import matplotlib.pyplot as plt
import torch

def compute_pod_basis(snapshot: np.ndarray, num_basis: int, weight: np.ndarray = None):
    """
    Compute POD basis using NumPy.

    Parameters:
        snapshot: (num_samples, num_dof) complex or real ndarray
        num_basis: number of POD modes to extract
        weight: (num_dof, num_dof) ndarray, optional

    Returns:
        sqrt_eigvals: (num_basis,) ndarray of sqrt of top eigenvalues
        pod_basis: (num_dof, num_basis) ndarray of POD modes
    """
    num_samples, num_dof = snapshot.shape
    assert num_basis < num_dof

    if weight is None:
        C = snapshot @ snapshot.T
    else:
        C = snapshot @ weight @ snapshot.T

    eigvals, eigvecs = np.linalg.eigh(C)  # ascending order
    eigvals = eigvals[-num_basis:]
    eigvecs = eigvecs[:, -num_basis:]


    # Sort eigenvalues and eigenvectors in descending order
    # Note that np.flip is not in-place, need to mannually copy to accelerate the following computation
    eigvals = np.flip(eigvals, axis=0).copy() 
    eigvecs = np.flip(eigvecs, axis=1).copy()

    temp = snapshot.T @ eigvecs
    pod_basis = np.zeros((num_dof, num_basis), dtype=snapshot.dtype)
    for i in range(num_basis):
        pod_basis[:, i] = temp[:, i] / np.sqrt(eigvals[i])

    return np.sqrt(eigvals), pod_basis


def plot_eigenvalues(eigenvalues: np.ndarray, title: str):
    indices = np.arange(1, len(eigenvalues) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.log10(indices), np.log10(eigenvalues), color='blue', marker='o', markersize=3)
    ax.set_xlabel(r'$\log_{10}(i)$', fontsize=15)
    ax.set_ylabel(r'$\log_{10}(\lambda_i)$', fontsize=15, rotation=0, labelpad=30)
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='both', labelsize=15) 
    plt.close()
    return fig


def compute_reduced_states(states: torch.Tensor, pod_basis_tensor: torch.Tensor):
    """"
    Inputs: 
    states: (batch_size, num_timestamps, num_features, height, width) 
    pod_basis_tensor: (num_features, height * width, num_pod_modes) 

    Outputs:
    reduced_states: (batch_size, num_timestamps, num_features, num_pod_modes)
    """
    batch_size, num_timestamps, num_features, height, width = states.shape
    num_pod_modes = pod_basis_tensor.shape[2]
    reduced_states = torch.zeros((batch_size, num_timestamps, num_features, num_pod_modes), dtype=states.dtype, device=states.device)
    for i in range(num_features):
        # Reshape the states for the current feature
        reshaped_states = states[:, :, i, :, :].reshape(batch_size * num_timestamps, height * width)
        # Compute the reduced states using the POD basis
        reduced_states[:, :, i, :] = (reshaped_states @ pod_basis_tensor[i, :, :]).reshape(batch_size, num_timestamps, num_pod_modes)
    return reduced_states


def reconstruct_full_states(reduced_states: torch.Tensor, pod_basis_tensor: torch.Tensor, height: int, width: int):
    """
    Reconstruct full states from reduced states using the POD basis.

    Inputs: 
    reduced_states: (batch_size, num_timestamps, num_features, num_pod_modes)
    pod_basis_tensor: (num_features, height * width, num_pod_modes)

    Outputs:
    full_states: (batch_size, num_timestamps, num_features, height, width)
    """
    batch_size, num_timestamps, num_features, num_pod_modes = reduced_states.shape
    full_states = torch.zeros((batch_size, num_timestamps, num_features, height, width), dtype=reduced_states.dtype, device=reduced_states.device)
    for i in range(num_features):
        # Compute the full states using the POD basis
        full_states[:, :, i, :, :] = (reduced_states[:, :, i, :] @ pod_basis_tensor[i, :, :].T).reshape(batch_size, num_timestamps, height, width)
    return full_states