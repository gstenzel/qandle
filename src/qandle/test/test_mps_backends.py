import pytest
import torch
import qandle
from qandle import Circuit

def test_ghz_state_and_measure():

    gates = [
        qandle.H(0, num_qubits=3),
        qandle.CNOT(0, 1),
        qandle.CNOT(1, 2)
    ]
    c = Circuit(gates, num_qubits=3)
    vec = c(backend="statevector")
    mps = c(backend="mps")
    torch.testing.assert_close(vec.state, mps._to_statevector())
    torch.testing.assert_close(vec.measure(), mps.measure())

def test_measure_subsets():
    gates = [
        qandle.H(0, num_qubits=2),
        qandle.CNOT(1, 0),
        qandle.CNOT(0, 1),
        qandle.H(1, num_qubits=2)
    ]
    c = Circuit(gates, num_qubits=2)
    vec = c(backend="statevector")
    mps = c(backend="mps")
    for qs in (None, [0], [1], [0,1]):
        torch.testing.assert_close(vec.measure(qs), mps.measure(qs))

def test_state_vector_simulator_correctness():
    gates = [qandle.H(0, num_qubits=2), qandle.CNOT(0, 1)]
    c = Circuit(gates, num_qubits=2)
    direct_state = c()
    backend = c(backend="statevector")
    torch.testing.assert_close(direct_state, backend.state)
    torch.testing.assert_close(qandle.MeasureJointProbability()(direct_state),
                               backend.measure())