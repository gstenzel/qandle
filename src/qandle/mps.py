"""
Matrix Product State (MPS) implementation for qandle.

This module provides efficient tensor network representations for quantum states
as an alternative to full state vectors.
"""

import torch
import typing
from dataclasses import dataclass


@dataclass
class MPSState:
    """
    Matrix Product State representation of a quantum state.
    
    The state |ψ⟩ is represented as:
    |ψ⟩ = Σ A₁[i₁] A₂[i₂] ... Aₙ[iₙ] |i₁i₂...iₙ⟩
    
    where each Aₖ[iₖ] is a matrix of bond dimension χ.
    """
    tensors: typing.List[torch.Tensor]
    num_qubits: int
    
    def __post_init__(self):
        assert len(self.tensors) == self.num_qubits
        # Validate tensor shapes
        for i, tensor in enumerate(self.tensors):
            if i == 0:
                assert tensor.shape == (2, tensor.shape[1]), f"First tensor wrong shape: {tensor.shape}"
            elif i == self.num_qubits - 1:
                assert tensor.shape == (tensor.shape[0], 2), f"Last tensor wrong shape: {tensor.shape}"
            else:
                assert tensor.shape[1] == 2, f"Tensor {i} wrong physical dimension: {tensor.shape}"
    
    @property
    def bond_dims(self) -> typing.List[int]:
        """Get bond dimensions between tensors."""
        dims = []
        for i in range(len(self.tensors) - 1):
            dims.append(self.tensors[i].shape[-1])
        return dims
    
    @property
    def max_bond_dim(self) -> int:
        """Get maximum bond dimension."""
        if len(self.tensors) <= 1:
            return 1
        return max(self.bond_dims)


def create_zero_state(num_qubits: int) -> MPSState:
    """Create |00...0⟩ state in MPS form."""
    tensors = []
    for i in range(num_qubits):
        if i == 0:
            # First tensor: shape (2, 1)
            tensor = torch.zeros((2, 1), dtype=torch.cfloat)
            tensor[0, 0] = 1.0
        elif i == num_qubits - 1:
            # Last tensor: shape (1, 2)
            tensor = torch.zeros((1, 2), dtype=torch.cfloat)
            tensor[0, 0] = 1.0
        else:
            # Middle tensors: shape (1, 2, 1)
            tensor = torch.zeros((1, 2, 1), dtype=torch.cfloat)
            tensor[0, 0, 0] = 1.0
        tensors.append(tensor)
    return MPSState(tensors, num_qubits)


def mps_to_state_vector(mps: MPSState) -> torch.Tensor:
    """Convert MPS to full state vector."""
    if mps.num_qubits == 1:
        return mps.tensors[0].flatten()
    
    # Start with first tensor: shape (2, bond_1)
    result = mps.tensors[0]
    
    for i in range(1, mps.num_qubits):
        tensor = mps.tensors[i]
        
        if i == mps.num_qubits - 1:
            # Last tensor: shape (bond_i, 2)
            # Contract: (..., bond_i) x (bond_i, 2) -> (..., 2)
            result = torch.einsum('...i,ij->...j', result, tensor)
        else:
            # Middle tensor: shape (bond_i, 2, bond_{i+1})
            # Contract: (..., bond_i) x (bond_i, 2, bond_{i+1}) -> (..., 2, bond_{i+1})
            result = torch.einsum('...i,ijk->...jk', result, tensor)
            # Reshape to flatten physical index with previous indices
            old_shape = result.shape
            # old_shape = (..., 2, bond_{i+1})
            # new_shape = (...*2, bond_{i+1})
            new_shape = (old_shape[0] * old_shape[1], old_shape[2])
            result = result.reshape(new_shape)
    
    # result now has shape (2^n,)
    return result.flatten()


def state_vector_to_mps(state: torch.Tensor, max_bond_dim: int = None) -> MPSState:
    """
    Convert state vector to MPS.
    For now, this is a fallback that doesn't do compression.
    """
    num_qubits = int(torch.log2(torch.tensor(len(state))).item())
    assert 2**num_qubits == len(state), "State vector length must be power of 2"
    
    # For simplicity, just create an MPS that can represent any state
    # This is not compressed but works correctly
    state = state.to(torch.cfloat)
    
    if num_qubits == 1:
        return MPSState([state.reshape(2, 1)], 1)
    
    # Create MPS with maximum bond dimension
    # This is not efficient but it's correct
    tensors = []
    
    # First tensor: extract first qubit
    # Reshape state as [2, 2^(n-1)]
    matrix = state.reshape(2, -1)
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    
    # For maximum generality, keep all singular values
    tensors.append(U)  # [2, min(2, 2^(n-1))]
    
    # Middle tensors - for simplicity, use the identity approach
    # This creates a valid MPS but not compressed
    current = torch.diag(S.to(torch.cfloat)) @ Vh  # [bond_dim, 2^(n-1)]
    
    for i in range(1, num_qubits - 1):
        bond_in = current.shape[0]
        # Reshape to [bond_in, 2, 2^(n-i-1)]
        remaining_size = current.shape[1] // 2
        current = current.reshape(bond_in, 2, remaining_size)
        
        # Convert to matrix and SVD
        matrix = current.reshape(bond_in * 2, remaining_size)
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        
        # Store tensor
        tensors.append(U.reshape(bond_in, 2, -1))
        
        # Update current
        current = torch.diag(S.to(torch.cfloat)) @ Vh
    
    # Last tensor
    tensors.append(current.reshape(-1, 2))
    
    return MPSState(tensors, num_qubits)


def apply_single_qubit_gate(mps: MPSState, qubit: int, gate_matrix: torch.Tensor) -> MPSState:
    """Apply single-qubit gate to MPS."""
    new_tensors = []
    
    for i, tensor in enumerate(mps.tensors):
        if i == qubit:
            # Apply gate to this qubit
            if i == 0:
                # First tensor: (2, bond) -> (2, bond)
                new_tensor = torch.einsum('ij,jk->ik', gate_matrix, tensor)
            elif i == mps.num_qubits - 1:
                # Last tensor: (bond, 2) -> (bond, 2)
                new_tensor = torch.einsum('ij,kj->ki', gate_matrix, tensor)
            else:
                # Middle tensor: (bond_left, 2, bond_right) -> (bond_left, 2, bond_right)
                new_tensor = torch.einsum('ij,kjl->kil', gate_matrix, tensor)
            new_tensors.append(new_tensor)
        else:
            new_tensors.append(tensor.clone())
    
    return MPSState(new_tensors, mps.num_qubits)


def apply_two_qubit_gate(mps: MPSState, qubit1: int, qubit2: int, gate_matrix: torch.Tensor, max_bond_dim: int = None) -> MPSState:
    """
    Apply two-qubit gate to MPS.
    
    For non-adjacent qubits, this is expensive and may require converting to state vector.
    For adjacent qubits, can be done efficiently with SVD.
    """
    # Ensure qubit1 < qubit2
    if qubit1 > qubit2:
        qubit1, qubit2 = qubit2, qubit1
        # Need to swap the gate matrix indices
        gate_matrix = gate_matrix.reshape(2, 2, 2, 2).transpose(0, 1).transpose(2, 3).reshape(4, 4)
    
    # For simplicity, always convert to state vector for two-qubit gates
    # This can be optimized later for specific cases
    state_vec = mps_to_state_vector(mps)
    
    # Apply gate using standard matrix multiplication
    n = mps.num_qubits
    gate_full = torch.eye(2**n, dtype=torch.cfloat)
    
    # Build full gate matrix efficiently
    for i in range(2**n):
        for j in range(2**n):
            # Extract bits for the two qubits
            i1 = (i >> (n - 1 - qubit1)) & 1
            i2 = (i >> (n - 1 - qubit2)) & 1
            j1 = (j >> (n - 1 - qubit1)) & 1 
            j2 = (j >> (n - 1 - qubit2)) & 1
            
            # Check if other qubits are the same
            mask = ~((1 << (n - 1 - qubit1)) | (1 << (n - 1 - qubit2)))
            if (i & mask) == (j & mask):
                gate_full[i, j] = gate_matrix[i1 * 2 + i2, j1 * 2 + j2]
    
    new_state = state_vec @ gate_full
    return state_vector_to_mps(new_state, max_bond_dim)


def _apply_adjacent_two_qubit_gate(mps: MPSState, qubit1: int, qubit2: int, gate_matrix: torch.Tensor, max_bond_dim: int = None) -> MPSState:
    """Apply two-qubit gate to adjacent qubits efficiently."""
    new_tensors = list(mps.tensors)
    
    # Contract the two tensors and apply gate
    tensor1 = mps.tensors[qubit1]
    tensor2 = mps.tensors[qubit2]
    
    if qubit1 == 0 and qubit2 == 1:
        # First two qubits
        # tensor1: (2, bond), tensor2: (bond, 2, ...)
        if mps.num_qubits == 2:
            # tensor2: (bond, 2)
            combined = torch.einsum('ij,jk->ijk', tensor1, tensor2)
            # combined: (2, bond, 2) -> (2, 2, bond)
            combined = combined.transpose(1, 2)
        else:
            # tensor2: (bond, 2, bond_right)  
            combined = torch.einsum('ij,jkl->ikl', tensor1, tensor2)
            # combined: (2, 2, bond_right)
        
        # Apply gate: reshape to (4, ...) and apply
        original_shape = combined.shape
        combined_flat = combined.reshape(4, -1)
        result_flat = gate_matrix @ combined_flat
        result = result_flat.reshape(original_shape)
        
        # SVD to split back
        if mps.num_qubits == 2:
            # result: (2, 2, bond) -> split along first dimension
            result_matrix = result.reshape(4, -1)
        else:
            # result: (2, 2, bond_right) -> split along first dimension  
            result_matrix = result.reshape(4, -1)
        
        U, S, Vh = torch.linalg.svd(result_matrix)
        
        if max_bond_dim is not None:
            bond_dim = min(max_bond_dim, len(S))
            U = U[:, :bond_dim]
            S = S[:bond_dim]
            Vh = Vh[:bond_dim, :]
        
        # Ensure complex dtype consistency
        S = S.to(torch.cfloat)
        
        new_tensor1 = U.reshape(2, -1)
        if mps.num_qubits == 2:
            new_tensor2 = (torch.diag(S) @ Vh).reshape(-1, 2)
        else:
            new_tensor2 = (torch.diag(S) @ Vh).reshape(-1, 2, original_shape[-1])
        
        new_tensors[qubit1] = new_tensor1
        new_tensors[qubit2] = new_tensor2
        
    elif qubit2 == mps.num_qubits - 1:
        # Last two qubits
        # tensor1: (bond_left, 2, bond), tensor2: (bond, 2)
        combined = torch.einsum('ijk,kl->ijl', tensor1, tensor2)
        # combined: (bond_left, 2, 2)
        
        # Apply gate
        combined_reshaped = combined.reshape(-1, 4)
        result_reshaped = combined_reshaped @ gate_matrix.T
        result = result_reshaped.reshape(-1, 2, 2)
        
        # SVD to split back
        result_matrix = result.reshape(-1, 4)
        U, S, Vh = torch.linalg.svd(result_matrix)
        
        if max_bond_dim is not None:
            bond_dim = min(max_bond_dim, len(S))
            U = U[:, :bond_dim]
            S = S[:bond_dim]
            Vh = Vh[:bond_dim, :]
        
        # Ensure complex dtype consistency
        S = S.to(torch.cfloat)
        
        new_tensor1 = U.reshape(-1, 2, U.shape[-1])
        new_tensor2 = Vh.reshape(-1, 2)
        
        new_tensors[qubit1] = new_tensor1  
        new_tensors[qubit2] = new_tensor2
        
    else:
        # Middle qubits
        # tensor1: (bond_left, 2, bond), tensor2: (bond, 2, bond_right)
        combined = torch.einsum('ijk,klm->ijlm', tensor1, tensor2)
        # combined: (bond_left, 2, 2, bond_right)
        
        # Apply gate
        original_shape = combined.shape
        combined_reshaped = combined.reshape(original_shape[0], 4, original_shape[3])
        result_reshaped = torch.einsum('ijk,lm->iljk', combined_reshaped, gate_matrix)
        result = result_reshaped.reshape(original_shape)
        
        # SVD to split back 
        bond_left, _, _, bond_right = result.shape
        result_matrix = result.reshape(bond_left * 2, 2 * bond_right)
        U, S, Vh = torch.linalg.svd(result_matrix)
        
        if max_bond_dim is not None:
            bond_dim = min(max_bond_dim, len(S))
            U = U[:, :bond_dim]
            S = S[:bond_dim]
            Vh = Vh[:bond_dim, :]
        
        # Ensure complex dtype consistency
        S = S.to(torch.cfloat)
        
        new_tensor1 = U.reshape(bond_left, 2, -1)
        new_tensor2 = (torch.diag(S) @ Vh).reshape(-1, 2, bond_right)
        
        new_tensors[qubit1] = new_tensor1
        new_tensors[qubit2] = new_tensor2
    
    return MPSState(new_tensors, mps.num_qubits)