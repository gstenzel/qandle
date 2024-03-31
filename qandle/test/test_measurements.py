import pennylane as qml
import torch
import qandle


def test_measureprob():
    @qml.qnode(device=qml.device("default.qubit.torch", wires=3), interface="torch")
    def pl_circuit(inp):
        qml.QubitStateVector(inp, wires=range(3))
        return [qml.probs(w) for w in range(3)]

    inp = torch.rand(2**3, dtype=torch.cfloat)
    inp = inp / torch.linalg.norm(inp, dim=-1, keepdim=True)
    pl_result = pl_circuit(inp)
    pl_result = torch.tensor([r[0] for r in pl_result], dtype=torch.float)
    qandle_mes = qandle.MeasureProbabilityBuilt(num_qubits=3)
    qandle_result = qandle_mes(inp).to(torch.float)
    assert torch.allclose(pl_result, qandle_result)


def test_measureprob_batched():
    @qml.qnode(device=qml.device("default.qubit.torch", wires=3), interface="torch")
    def pl_circuit(inp):
        qml.QubitStateVector(inp, wires=range(3))
        return [qml.probs(w) for w in range(3)]

    inp = torch.rand(17, 2**3, dtype=torch.cfloat)
    inp = inp / torch.linalg.norm(inp, dim=-1, keepdim=True)
    pl_result = pl_circuit(inp)
    pl_result = torch.tensor([list(r[..., 0]) for r in pl_result], dtype=torch.float).T
    qandle_mes = qandle.MeasureProbabilityBuilt(num_qubits=3)
    qandle_result = qandle_mes(inp).to(torch.float)
    assert torch.allclose(pl_result, qandle_result)


def test_measurejoint_batched():
    @qml.qnode(device=qml.device("default.qubit.torch", wires=3), interface="torch")
    def pl_circuit(inp):
        qml.QubitStateVector(inp, wires=range(3))
        return qml.probs(wires=range(3))

    inp = torch.rand(17, 2**3, dtype=torch.cfloat)
    inp = inp / torch.linalg.norm(inp, dim=-1, keepdim=True)
    pl_result = pl_circuit(inp).to(torch.float)

    qandle_mes = qandle.MeasureJointProbability()
    qandle_result = qandle_mes(inp).to(torch.float)
    assert torch.allclose(pl_result, qandle_result)
