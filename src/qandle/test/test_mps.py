"""
Tests for Matrix Product State (MPS) implementation.
"""

import torch
import pytest
import qandle
from qandle import mps


def test_mps_creation():
    """Test MPS state creation and basic properties."""
    # Test 1-qubit MPS
    mps1 = mps.create_zero_state(1)
    assert mps1.num_qubits == 1
    assert len(mps1.tensors) == 1
    assert mps1.tensors[0].shape == (2, 1)
    
    # Test 2-qubit MPS
    mps2 = mps.create_zero_state(2)
    assert mps2.num_qubits == 2
    assert len(mps2.tensors) == 2
    assert mps2.tensors[0].shape == (2, 1)
    assert mps2.tensors[1].shape == (1, 2)
    
    # Test 3-qubit MPS
    mps3 = mps.create_zero_state(3)
    assert mps3.num_qubits == 3
    assert len(mps3.tensors) == 3
    assert mps3.tensors[0].shape == (2, 1)
    assert mps3.tensors[1].shape == (1, 2, 1)
    assert mps3.tensors[2].shape == (1, 2)


def test_mps_to_state_vector():
    """Test MPS to state vector conversion."""
    # Test 1-qubit |0⟩
    mps1 = mps.create_zero_state(1)
    state1 = mps.mps_to_state_vector(mps1)
    expected1 = torch.tensor([1.0, 0.0], dtype=torch.cfloat)
    assert torch.allclose(state1, expected1)
    
    # Test 2-qubit |00⟩
    mps2 = mps.create_zero_state(2)
    state2 = mps.mps_to_state_vector(mps2)
    expected2 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.cfloat)
    assert torch.allclose(state2, expected2)
    
    # Test 3-qubit |000⟩
    mps3 = mps.create_zero_state(3)
    state3 = mps.mps_to_state_vector(mps3)
    expected3 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.cfloat)
    assert torch.allclose(state3, expected3)


def test_single_qubit_gates():
    """Test single-qubit gate operations on MPS."""
    # X gate on |0⟩ should give |1⟩
    mps_state = mps.create_zero_state(2)
    x_gate = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat)
    
    # Apply X to qubit 0
    mps_x = mps.apply_single_qubit_gate(mps_state, 0, x_gate)
    state_x = mps.mps_to_state_vector(mps_x)
    expected = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.cfloat)  # |10⟩
    assert torch.allclose(state_x, expected)
    
    # Apply X to qubit 1
    mps_x1 = mps.apply_single_qubit_gate(mps_state, 1, x_gate)
    state_x1 = mps.mps_to_state_vector(mps_x1)
    expected1 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.cfloat)  # |01⟩
    assert torch.allclose(state_x1, expected1)


def test_two_qubit_gates():
    """Test two-qubit gate operations on MPS."""
    # Test CNOT gate
    mps_state = mps.create_zero_state(2)
    
    # First apply X to control qubit
    x_gate = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat)
    mps_state = mps.apply_single_qubit_gate(mps_state, 0, x_gate)
    
    # Then apply CNOT
    cnot_matrix = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0], 
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.cfloat)
    
    mps_cnot = mps.apply_two_qubit_gate(mps_state, 0, 1, cnot_matrix)
    state_cnot = mps.mps_to_state_vector(mps_cnot)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.cfloat)  # |11⟩
    assert torch.allclose(state_cnot, expected)


def test_mps_circuit_compatibility():
    """Test MPS circuit gives same results as state vector circuit."""
    # Test various gate combinations
    gates_to_test = [
        [qandle.RX(qubit=0, theta=0.5)],
        [qandle.RY(qubit=0, theta=0.3)],
        [qandle.RZ(qubit=0, theta=0.7)],
        [qandle.RX(qubit=0, theta=0.5), qandle.CNOT(control=0, target=1)],
        [qandle.RY(qubit=0, theta=0.3), qandle.RZ(qubit=1, theta=0.7), qandle.CNOT(control=0, target=1)],
    ]
    
    for gates in gates_to_test:
        # Determine number of qubits
        max_qubit = 0
        for gate in gates:
            if hasattr(gate, 'qubit'):
                max_qubit = max(max_qubit, gate.qubit)
            elif hasattr(gate, 'c'):  # CNOT
                max_qubit = max(max_qubit, gate.c, gate.t)
        num_qubits = max_qubit + 1
        
        # Create circuits
        circuit_statevec = qandle.Circuit(gates, num_qubits=num_qubits, use_mps=False)
        circuit_mps = qandle.Circuit(gates, num_qubits=num_qubits, use_mps=True)
        
        # Test with default |00...0⟩ state
        result_statevec = circuit_statevec()
        result_mps = circuit_mps()
        
        assert torch.allclose(result_statevec, result_mps, atol=1e-6), \
            f"Results differ for gates {gates}"
        
        # Test with custom input state
        initial_state = torch.rand(2**num_qubits, dtype=torch.cfloat)
        initial_state = initial_state / torch.norm(initial_state)
        
        result_statevec_custom = circuit_statevec(initial_state)
        result_mps_custom = circuit_mps(initial_state)
        
        assert torch.allclose(result_statevec_custom, result_mps_custom, atol=1e-5), \
            f"Results differ for gates {gates} with custom input"


def test_mps_properties():
    """Test MPS state properties and invariants."""
    # Test that MPS states are properly normalized
    for num_qubits in range(1, 4):
        mps_state = mps.create_zero_state(num_qubits)
        state_vec = mps.mps_to_state_vector(mps_state)
        assert torch.allclose(torch.norm(state_vec), torch.tensor(1.0))
    
    # Test that gates preserve normalization
    mps_state = mps.create_zero_state(2)
    x_gate = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat)
    mps_x = mps.apply_single_qubit_gate(mps_state, 0, x_gate)
    state_x = mps.mps_to_state_vector(mps_x)
    assert torch.allclose(torch.norm(state_x), torch.tensor(1.0))


def test_parametrized_gates():
    """Test parametrized gates work correctly with MPS."""
    angles = [0.0, 0.5, 1.0, 1.5, 3.14159]
    
    for angle in angles:
        # Test RX gate
        circuit_statevec = qandle.Circuit([qandle.RX(qubit=0, theta=angle)], 
                                        num_qubits=2, use_mps=False)
        circuit_mps = qandle.Circuit([qandle.RX(qubit=0, theta=angle)], 
                                   num_qubits=2, use_mps=True)
        
        result_statevec = circuit_statevec()
        result_mps = circuit_mps()
        
        assert torch.allclose(result_statevec, result_mps, atol=1e-6), \
            f"RX({angle}) results differ"
        
        # Test RY gate
        circuit_statevec = qandle.Circuit([qandle.RY(qubit=0, theta=angle)], 
                                        num_qubits=2, use_mps=False)
        circuit_mps = qandle.Circuit([qandle.RY(qubit=0, theta=angle)], 
                                   num_qubits=2, use_mps=True)
        
        result_statevec = circuit_statevec()
        result_mps = circuit_mps()
        
        assert torch.allclose(result_statevec, result_mps, atol=1e-6), \
            f"RY({angle}) results differ"
        
        # Test RZ gate
        circuit_statevec = qandle.Circuit([qandle.RZ(qubit=0, theta=angle)], 
                                        num_qubits=2, use_mps=False)
        circuit_mps = qandle.Circuit([qandle.RZ(qubit=0, theta=angle)], 
                                   num_qubits=2, use_mps=True)
        
        result_statevec = circuit_statevec()
        result_mps = circuit_mps()
        
        assert torch.allclose(result_statevec, result_mps, atol=1e-6), \
            f"RZ({angle}) results differ"


def test_custom_gates():
    """Test custom gates work with MPS."""
    # Test Hadamard gate
    h_matrix = torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / torch.sqrt(torch.tensor(2.0))
    h_gate = qandle.CustomGate(qubit=0, matrix=h_matrix, num_qubits=2)
    
    circuit_statevec = qandle.Circuit([h_gate], num_qubits=2, use_mps=False)
    circuit_mps = qandle.Circuit([h_gate], num_qubits=2, use_mps=True)
    
    result_statevec = circuit_statevec()
    result_mps = circuit_mps()
    
    assert torch.allclose(result_statevec, result_mps, atol=1e-6)
    
    # Check that Hadamard on |0⟩ gives (|0⟩ + |1⟩)/√2 for qubit 0, with qubit 1 in |0⟩
    # So the full state should be (|00⟩ + |10⟩)/√2
    expected = torch.tensor([1/torch.sqrt(torch.tensor(2.0)), 0, 1/torch.sqrt(torch.tensor(2.0)), 0], dtype=torch.cfloat)
    assert torch.allclose(result_mps, expected, atol=1e-6)


def test_backward_compatibility():
    """Test that existing code works unchanged (use_mps=False by default)."""
    # This should work exactly as before
    circuit = qandle.Circuit([qandle.RX(qubit=0, theta=0.5), qandle.CNOT(control=0, target=1)], 
                           num_qubits=2)
    
    result = circuit()
    assert isinstance(result, torch.Tensor)
    assert torch.allclose(torch.norm(result), torch.tensor(1.0))
    
    # Verify it's using state vector by default (not MPS)
    assert not hasattr(circuit.circuit, 'use_mps') or not circuit.circuit.use_mps