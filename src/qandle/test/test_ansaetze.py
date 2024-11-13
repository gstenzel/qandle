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
        qubits=list(range(num_qubits)), depth=depth, q_params=weights, remapping=None
    ).build(num_qubits=num_qubits)
    qandle_result = qandle_sel(inp)
    assert torch.allclose(pl_result, qandle_result, rtol=1e-6, atol=1e-6)


def test_sel_to_matrix():
    num_qubits = 5
    depth = 11
    pl_dev = qml.device("default.qubit.torch", wires=num_qubits)
    weights = torch.rand(depth, num_qubits, 3)

    @qml.qnode(device=pl_dev, interface="torch")
    def pl_circuit():
        qml.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits))
        return qml.state()

    pl_matrix = qml.matrix(qml.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits)))
    qandle_sel = qandle.StronglyEntanglingLayer(
        qubits=list(range(num_qubits)), depth=depth, q_params=weights, remapping=None
    ).build(num_qubits=num_qubits)
    qandle_matrix = qandle_sel.to_matrix()
    assert torch.allclose(pl_matrix, qandle_matrix, rtol=1e-6, atol=1e-6)


def test_sel_sub():
    num_qubits = 5
    batch = 17

    qandle_sel1 = qandle.StronglyEntanglingLayer(qubits=[0, 1, 2, 3], depth=7).build(num_qubits=5)
    qandle_sel2 = qandle.StronglyEntanglingLayer(qubits=[1, 2, 3, 4], depth=7).build(num_qubits=5)
    qandle_sel3 = qandle.StronglyEntanglingLayer(qubits=[0, 1, 3, 4], depth=7).build(num_qubits=5)

    inp = torch.rand(batch, 2**num_qubits, dtype=torch.cfloat)
    inp = inp / torch.linalg.norm(inp, dim=1, keepdim=True)
    qandle_sel1(inp)
    qandle_sel2(inp)
    qandle_sel3(inp)


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
        qubits=list(range(num_qubits)), depth=depth, q_params=weights, remapping=None
    ).build(num_qubits=num_qubits)
    qandle_result = qandle_sel(inp)
    assert torch.allclose(pl_result, qandle_result, rtol=1e-6, atol=1e-6)


def test_sel_budget():
    num_qubits = 5
    depth = 4
    rots = ["rz", "ry", "rz"]
    sel = qandle.StronglyEntanglingLayer(
        qubits=list(range(num_qubits)), depth=depth, rotations=rots, num_qubits_total=num_qubits
    ).build(num_qubits=num_qubits)
    sel_budget = qandle.StronglyEntanglingLayerBudget(
        num_qubits_total=num_qubits,
        rotations=rots,
        param_budget=num_qubits * depth * len(rots),
        qubits=list(range(num_qubits)),
        control_gate_spacing=3,
    )
    sel_dec = sel.decompose()
    sel_budget_dec = sel_budget.layers
    assert len(sel_dec) == len(sel_budget_dec)
    ssel_dec = sorted(sel_dec, key=lambda x: str(x))
    ssel_budget_dec = sorted(sel_budget_dec, key=lambda x: str(x))
    for s, sb in zip(ssel_dec, ssel_budget_dec):
        assert type(s) is type(sb)
    inp = torch.rand(7, 2**num_qubits, dtype=torch.cfloat)
    inp = inp / torch.linalg.norm(inp, dim=1, keepdim=True)
    sel(inp)


def test_sel_general():
    num_qubits = 5
    qandle_sel_ub = qandle.StronglyEntanglingLayer(qubits=list(range(num_qubits)), depth=10)
    qandle_sel = qandle_sel_ub.build(num_qubits=num_qubits)
    inp = torch.rand(2**num_qubits, dtype=torch.cfloat)
    assert isinstance(qandle_sel(inp), torch.Tensor)
    assert isinstance(qandle_sel.decompose(), list)
    assert isinstance(qandle_sel.__str__(), str)


def test_twolocal():
    utwo = qandle.TwoLocal(qubits=list(range(4)))
    assert isinstance(utwo.decompose(), list)
    assert isinstance(utwo.__str__(), str)

    inp = torch.rand(2**4, dtype=torch.cfloat)
    inp = inp / inp.norm()
    two = utwo.build(num_qubits=4)
    assert isinstance(two(inp), torch.Tensor)
    assert isinstance(two.decompose(), list)
    assert isinstance(two.__str__(), str)
    two.to_qasm()

    inp2 = torch.rand(2**5, dtype=torch.cfloat)
    inp2 = inp2 / inp2.norm()
    two2 = utwo.build(num_qubits=5)
    assert isinstance(two2(inp2), torch.Tensor)


def test_su():
    for num_w in [2, 3, 6]:
        inp = torch.rand(2**num_w, dtype=torch.cfloat)
        inp = inp / inp.norm()
        for reps in [0, 1, 2, 10]:
            for rots in [["ry"], ["rz"], ["ry", "rx"]]:
                su = qandle.SU(reps=reps, rotations=rots).build(num_qubits=num_w)
                assert isinstance(su(inp), torch.Tensor)
                assert isinstance(su.decompose(), list)
                assert isinstance(su.__str__(), str)
                assert len(su.decompose()) == len(rots) * num_w * (1 + reps) + reps * (num_w - 1)
        for additional in [0, 1, 2]:
            su1 = qandle.SU(qubits=list(range(additional, num_w + additional)))
            inp = torch.rand(2 ** (num_w + additional * 2), dtype=torch.cfloat)
            inp = inp / inp.norm()
            su1.build(num_qubits=num_w + additional * 2)(inp)
