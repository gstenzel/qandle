import pennylane as qml
import torch
import qandle


def test_sel():
    """Test StronglyEntanglingLayers"""
    num_qubits = 4
    depth = 10
    pl_dev = qml.device("default.qubit.torch", wires=num_qubits)
    inp = torch.rand(2**num_qubits, dtype=torch.cfloat)
    inp = inp / inp.norm()
    weights = torch.rand(depth, num_qubits, 3)

    @qml.qnode(device=pl_dev, interface="torch")
    def pl_circuit():
        qml.QubitStateVector(inp, wires=range(num_qubits))
        qml.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits))
        return qml.state()

    pl_result = pl_circuit().to(torch.cfloat)
    qandle_sel = qandle.StronglyEntanglingLayer(
        num_qubits=num_qubits, depth=depth, q_params=weights, remapping=None
    ).build()
    qandle_result = qandle_sel(inp)
    assert torch.allclose(pl_result, qandle_result, rtol=1e-6, atol=1e-6)


def test_sel_batched():
    num_qubits = 5
    depth = 7
    batch = 17
    pl_dev = qml.device("default.qubit.torch", wires=num_qubits)
    inp = torch.rand(batch, 2**num_qubits, dtype=torch.cfloat)
    inp = inp / torch.linalg.norm(inp, dim=1, keepdim=True)
    weights = torch.rand(depth, num_qubits, 3)

    @qml.qnode(device=pl_dev, interface="torch")
    def pl_circuit():
        qml.QubitStateVector(inp, wires=range(num_qubits))
        qml.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits))
        return qml.state()

    pl_result = pl_circuit().to(torch.cfloat)
    qandle_sel = qandle.StronglyEntanglingLayer(
        num_qubits=num_qubits, depth=depth, q_params=weights, remapping=None
    ).build()
    qandle_result = qandle_sel(inp)
    assert torch.allclose(pl_result, qandle_result, rtol=1e-6, atol=1e-6)


def test_sel_general():
    num_qubits = 5
    qandle_sel_ub = qandle.StronglyEntanglingLayer(num_qubits=num_qubits, depth=10)
    qandle_sel = qandle_sel_ub.build()
    inp = torch.rand(2**num_qubits, dtype=torch.cfloat)
    assert isinstance(qandle_sel(inp), torch.Tensor)
    assert isinstance(qandle_sel.decompose(), list)
    # assert isinstance(qandle_sel._to_openqasm2(), str)
    assert isinstance(qandle_sel.__str__(), str)


def test_twolocal():
    utwo = qandle.TwoLocal(num_qubits=4)
    # assert isinstance(utwo._to_openqasm2(), str)
    assert isinstance(utwo.decompose(), list)
    assert isinstance(utwo.__str__(), str)

    inp = torch.rand(2**4, dtype=torch.cfloat)
    inp = inp / inp.norm()
    two = utwo.build()
    assert isinstance(two(inp), torch.Tensor)
    # assert isinstance(two._to_openqasm2(), str)
    assert isinstance(two.decompose(), list)
    assert isinstance(two.__str__(), str)


def test_su():
    for num_w in [2, 3, 6]:
        inp = torch.rand(2**num_w, dtype=torch.cfloat)
        inp = inp / inp.norm()
        for reps in [0, 1, 2, 10]:
            for rots in [["ry"], ["rz"], ["ry", "rx"]]:
                su = qandle.SU(num_w, reps=reps, rotations=rots).build()
                assert isinstance(su(inp), torch.Tensor)
                assert isinstance(su.decompose(), list)
                # assert isinstance(su._to_openqasm2(), str)
                assert isinstance(su.__str__(), str)
                assert len(su.decompose()) == len(rots) * num_w * (1 + reps) + reps * (
                    num_w - 1
                )
