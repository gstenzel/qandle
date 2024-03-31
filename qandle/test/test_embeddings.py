import torch
from qandle import embeddings
import pennylane as qml


def test_amplitude_embedding_unpadded():
    w = 4
    inp = torch.arange(2**w).to(torch.float)
    emb = embeddings.AmplitudeEmbedding(qubits=range(w), normalize=True, pad_with=None)
    out = emb(inp)
    assert torch.allclose(out.real, inp / inp.norm())
    assert torch.allclose(out.imag, torch.zeros_like(inp))

    @qml.qnode(device=qml.device("default.qubit.torch", wires=w), interface="torch")
    def pl_circuit():
        qml.AmplitudeEmbedding(inp, wires=range(w), normalize=True)
        return qml.state()

    pl_out = pl_circuit().to(torch.cfloat)
    assert torch.allclose(out, pl_out)


def test_amplitude_embedding_unpadded_batched():
    w = 4
    inp = torch.rand(10, 2**w).to(torch.float)
    inp = inp / inp.norm(dim=1, keepdim=True)
    emb = embeddings.AmplitudeEmbedding(qubits=range(w), normalize=False, pad_with=None)
    out = emb(inp)
    assert torch.allclose(out.real, inp)
    assert torch.allclose(out.imag, torch.zeros_like(inp))

    @qml.qnode(device=qml.device("default.qubit.torch", wires=w), interface="torch")
    def pl_circuit():
        qml.AmplitudeEmbedding(inp, wires=range(w), normalize=True)
        return qml.state()

    pl_out = pl_circuit().to(torch.cfloat)
    assert torch.allclose(out, pl_out)


def test_amplitude_embedding_padded():
    inp = torch.rand(11).to(torch.float)
    n_inp = inp / inp.norm()
    emb = embeddings.AmplitudeEmbedding(qubits=range(4), normalize=True, pad_with=0)
    out = emb(inp)
    assert torch.allclose(out.real[:11], n_inp)
    assert torch.allclose(out.imag, torch.zeros(16))


def test_angle_embedding_x3():
    w = 3
    inp = torch.arange(w).to(torch.float) + 0.5

    @qml.qnode(device=qml.device("default.qubit.torch", wires=w), interface="torch")
    def pl_circuit():
        qml.AngleEmbedding(inp, wires=range(w), rotation="X")
        return qml.state()

    pl_out = pl_circuit().to(torch.cfloat)
    emb = embeddings.AngleEmbedding(num_qubits=w, rotation="rx")
    out = emb(inp)
    assert torch.allclose(out, pl_out)

def test_angle_embedding_x3_batched():
    w = 3
    inp = torch.rand(10, w).to(torch.float) + 0.5

    @qml.qnode(device=qml.device("default.qubit.torch", wires=w), interface="torch")
    def pl_circuit():
        qml.AngleEmbedding(inp, wires=range(w), rotation="X")
        return qml.state()

    pl_out = pl_circuit().to(torch.cfloat)
    emb = embeddings.AngleEmbedding(num_qubits=w, rotation="rx")
    out = emb(inp)
    assert torch.allclose(out, pl_out)



def test_angle_embedding_y4():
    w = 4
    inp = torch.arange(w).to(torch.float) + 2.5

    @qml.qnode(device=qml.device("default.qubit.torch", wires=w), interface="torch")
    def pl_circuit():
        qml.AngleEmbedding(inp, wires=range(w), rotation="Y")
        return qml.state()

    pl_out = pl_circuit().to(torch.cfloat)
    emb = embeddings.AngleEmbedding(num_qubits=w, rotation="y")
    out = emb(inp)
    assert torch.allclose(out, pl_out)

def test_angle_embedding_y4_batched():
    w = 4
    inp = torch.rand(10, w).to(torch.float) + 2.5

    @qml.qnode(device=qml.device("default.qubit.torch", wires=w), interface="torch")
    def pl_circuit():
        qml.AngleEmbedding(inp, wires=range(w), rotation="Y")
        return qml.state()

    pl_out = pl_circuit().to(torch.cfloat)
    emb = embeddings.AngleEmbedding(num_qubits=w, rotation="y")
    out = emb(inp)
    assert torch.allclose(out, pl_out)


def test_angle_embedding_z5():
    w = 5
    inp = torch.arange(w).to(torch.float) + 2.5

    @qml.qnode(device=qml.device("default.qubit.torch", wires=w), interface="torch")
    def pl_circuit():
        qml.AngleEmbedding(inp, wires=range(w), rotation="Z")
        return qml.state()

    pl_out = pl_circuit().to(torch.cfloat)
    emb = embeddings.AngleEmbedding(num_qubits=w, rotation="z")
    out = emb(inp)
    assert torch.allclose(out, pl_out)

def test_angle_embedding_z5_batched():
    w = 5
    inp = torch.rand(10, w).to(torch.float) + 2.5

    @qml.qnode(device=qml.device("default.qubit.torch", wires=w), interface="torch")
    def pl_circuit():
        qml.AngleEmbedding(inp, wires=range(w), rotation="Z")
        return qml.state()

    pl_out = pl_circuit().to(torch.cfloat)
    emb = embeddings.AngleEmbedding(num_qubits=w, rotation="z")
    out = emb(inp)
    assert torch.allclose(out, pl_out)
