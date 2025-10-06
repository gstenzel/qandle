import torch
import qandle
from qandle.test import check_qasm
import numpy as np

op = qandle.operators


def get_circuit_cnot10():
    return qandle.Circuit(
        num_qubits=10,
        layers=[
            op.CNOT(0, 1),
            op.CNOT(0, 2),
            op.CNOT(1, 2),
            op.CNOT(2, 3),
            op.CNOT(3, 4),
            op.CNOT(4, 3),
            op.CNOT(3, 4),
            op.CNOT(4, 3),
            op.CNOT(3, 4),
            op.CNOT(4, 3),
            op.CNOT(3, 4),
            op.CNOT(4, 3),
            op.CNOT(3, 5),
            op.CNOT(5, 6),
            op.CNOT(6, 7),
            op.CNOT(6, 1),
            op.CNOT(7, 8),
            op.CNOT(8, 9),
        ],
    )


def get_circuit_ccnot_num_qubits(num_qubits=10):
    assert num_qubits >=8, "num_qubits must be >= 8"
    return qandle.Circuit(
        [op.CCNOT(i, (i + np.random.randint(1, 2)) % num_qubits, (i + np.random.randint(3, 5)) % num_qubits) for i in
         range(num_qubits)], num_qubits=num_qubits
    )


def get_circuit_ccnot3():
    return qandle.Circuit(
        num_qubits=3,
        layers=[
            op.CCNOT(0, 1, 2),
            op.CCNOT(1, 2, 0),
            op.CCNOT(2, 0, 1),
        ])

def test_circuit_ccnot():

    c = get_circuit_ccnot3()
    split_c = c.split(max_qubits=2)
    inp = torch.rand(2 ** 3, dtype=torch.cfloat)
    inp = inp / torch.linalg.norm(inp, dim=-1, keepdim=True)
    res_c = c(inp)
    res_split = split_c(inp)
    assert torch.allclose(res_c, res_split, rtol=1e-6, atol=1e-6), f"Results in splitted CCNOT (3 qubits) do not match: {res_c} vs {res_split}"

    c_large = get_circuit_ccnot_num_qubits()
    split_c_large = c_large.split(max_qubits=5)
    inp_large = torch.rand(2 ** c_large.num_qubits, dtype=torch.cfloat)
    inp_large = inp_large / torch.linalg.norm(inp_large, dim=-1, keepdim=True)
    res_c_large = c_large(inp_large)
    res_split_large = split_c_large(inp_large)
    assert torch.allclose(res_c_large, res_split_large, rtol=1e-6, atol=1e-6), f"Results in splitted CCNOT (10 qubits) do not match: {res_c_large} vs {res_split_large}"


def test_nested_circuits():
    c1 = qandle.Circuit(
        layers=[
            qandle.RX(0),
            qandle.CNOT(0, 1),
        ],
        num_qubits=3,
    )
    c2 = qandle.Circuit(
        layers=[qandle.RY(1), qandle.StronglyEntanglingLayer(qubits=[1, 2])],
        num_qubits=3,
    )
    c1c2 = qandle.Circuit(layers=[c1, qandle.RZ(0), c2], num_qubits=3)
    split = c1c2.split(max_qubits=3)
    inp = torch.rand(2 ** 3, dtype=torch.cfloat)
    inp = inp / torch.linalg.norm(inp, dim=-1, keepdim=True)
    res_c1c2 = c1c2(inp)
    res_split = split(inp)
    assert torch.allclose(res_c1c2, res_split, rtol=1e-6, atol=1e-6)


def test_splitter_1():
    orig_c = get_circuit_cnot10()
    split_c = orig_c.split(max_qubits=5)
    inp = torch.rand(2 ** orig_c.num_qubits, dtype=torch.cfloat)
    inp = inp / torch.linalg.norm(inp, dim=-1, keepdim=True)
    orig_res = orig_c(inp)
    split_res = split_c(inp)
    assert torch.allclose(orig_res, split_res, rtol=1e-6, atol=1e-6)
    assert torch.allclose(orig_res, orig_c @ inp, rtol=1e-6, atol=1e-6)
    assert torch.allclose(split_res, split_c @ inp, rtol=1e-6, atol=1e-6)
    check_qasm.check(orig_c)
    check_qasm.check(split_c)


def test_splitter_2():
    su = qandle.SU(reps=2, rotations=["rx", "ry"], qubits=list(range(10)))
    orig_c_big = qandle.Circuit(layers=[su])
    orig_c = orig_c_big.decompose()
    split_c = orig_c.split(max_qubits=5)
    orig_gates = list(orig_c.circuit.layers)
    split_gates = [sc.layers for sc in split_c.circuit.subcircuits]
    assert len(orig_gates) == sum([len(g) for g in split_gates])
    assert isinstance(split_c.draw(), str)
    check_qasm.check(orig_c)
    check_qasm.check(split_c)


def test_splitter_small():
    su = qandle.SU(qubits=list(range(3)), reps=2, rotations=["rx", "ry"])
    orig_c_big = qandle.Circuit(layers=su.decompose()).decompose()
    inp = torch.rand(2 ** orig_c_big.num_qubits, dtype=torch.cfloat)
    inp = inp / torch.linalg.norm(inp, dim=-1, keepdim=True)
    split_t2 = orig_c_big.split(max_qubits=2)
    split_t3 = orig_c_big.split(max_qubits=3)
    split_t4 = orig_c_big.split(max_qubits=4)

    out_orig = orig_c_big(inp)
    out_t2 = split_t2(inp)
    out_t3 = split_t3(inp)
    out_t4 = split_t4(inp)

    assert torch.allclose(out_orig, out_t2, rtol=1e-6, atol=1e-6)
    assert torch.allclose(out_orig, out_t3, rtol=1e-6, atol=1e-6)
    assert torch.allclose(out_orig, out_t4, rtol=1e-6, atol=1e-6)
    check_qasm.check(orig_c_big)
    check_qasm.check(split_t2)
    check_qasm.check(split_t3)


def test_circuit_dummy():
    c = qandle.Circuit(
        split_max_qubits=2,
        layers=[qandle.RX(0), qandle.CNOT(0, 2), qandle.RY(1), qandle.CNOT(1, 2)],
    )
    assert isinstance(c.circuit, qandle.qcircuit.SplittedCircuit)
    assert len(c.circuit.subcircuits) == 2

