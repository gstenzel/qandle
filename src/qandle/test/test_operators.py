import pennylane as qml
import torch
import qandle


def test_reset_unbatched():
    for num_qubits in range(1, 5):
        probs = qandle.MeasureProbabilityBuilt(num_qubits=num_qubits)
        for qubit in range(num_qubits):
            reset = qandle.Reset(qubit=qubit)
            assert isinstance(reset.__str__(), str)
            reset.to_qasm()
            resetb = reset.build(num_qubits=num_qubits)
            assert isinstance(resetb.__str__(), str)
            resetb.to_qasm()
            inp = torch.arange(2**num_qubits).to(torch.cfloat) + 0.5 - 0.1j  # non-zero
            inp = inp / torch.linalg.norm(inp)
            inp.requires_grad = True
            probs_before = probs(inp)
            out = resetb(inp)
            probs_after = probs(out)
            if num_qubits == 1:
                assert torch.allclose(probs_after, torch.tensor(1.0))
            else:
                err = f"qubit: {qubit}, num_qubits: {num_qubits}, probs_before: {probs_before}, probs_after: {probs_after}"
                assert torch.allclose(probs_before[:qubit], probs_after[:qubit]), err
                assert torch.allclose(probs_before[qubit + 1 :], probs_after[qubit + 1 :]), err
                assert torch.allclose(probs_after[qubit], torch.tensor(1.0)), err
            assert out.requires_grad
            assert torch.allclose(torch.norm(out), torch.tensor(1.0))


def test_reset_batched():
    for num_qubits in range(1, 5):
        probs = qandle.MeasureProbabilityBuilt(num_qubits=num_qubits)
        for qubit in range(num_qubits):
            reset = qandle.Reset(qubit=qubit)
            resetb = reset.build(num_qubits=num_qubits)
            inp = torch.rand(7, 2**num_qubits, dtype=torch.cfloat)
            inp = inp / torch.linalg.norm(inp, dim=-1, keepdim=True)
            inp.requires_grad = True
            probs_before = probs(inp)
            out = resetb(inp)
            probs_after = probs(out)
            if num_qubits == 1:
                assert torch.allclose(probs_after, torch.tensor(1.0))
            else:
                err = f"qubit: {qubit}, num_qubits: {num_qubits}, probs_before: {probs_before}, probs_after: {probs_after}"
                assert torch.allclose(probs_before[..., :qubit], probs_after[..., :qubit]), err
                assert torch.allclose(
                    probs_before[..., qubit + 1 :], probs_after[..., qubit + 1 :]
                ), err
                assert torch.allclose(probs_after[..., qubit], torch.tensor(1.0)), err
            assert out.requires_grad
            assert torch.allclose(torch.norm(out, dim=-1), torch.tensor(1.0))


def forward_ground_truth(num_w: int, pl_op, inp):
    def pl_circuit():
        qml.QubitStateVector(inp, wires=range(num_w))
        for w in range(num_w):
            qml.RX(phi=0, wires=w)
        pl_op()
        return qml.state()

    dev = qml.device("default.qubit", wires=num_w)
    qnode = qml.QNode(pl_circuit, dev, interface="torch")
    return qnode().to(torch.cfloat)


def test_custom():
    torch.manual_seed(42)
    u = qandle.operators.U(qubit=1, matrix=torch.tensor([[1, 0], [0, 1]]))
    u_b = u.build(num_qubits=3)
    inp = torch.rand(2**3, dtype=torch.cfloat)
    inp = inp / torch.linalg.norm(inp)
    inp.requires_grad = True
    u_b(inp)
    assert isinstance(u.__str__(), str)
    assert isinstance(u_b.__str__(), str)
    u.to_qasm()
    u_b.to_qasm()


def test_unbatched():
    torch.manual_seed(42)
    v = 0.543321
    errors = []
    for num_w in range(6):
        inp = torch.rand(2**num_w, dtype=torch.cfloat)
        inp = inp / torch.linalg.norm(inp)
        inp.requires_grad = True
        for w in range(num_w):
            for pl_op, own_op in (
                (qml.RX, qandle.operators.RX),
                (qml.RY, qandle.operators.RY),
                (qml.RZ, qandle.operators.RZ),
            ):
                gt = forward_ground_truth(num_w, lambda: pl_op(v, wires=w), inp)
                own_rx_u = own_op(qubit=w, theta=v, remapping=None)
                own_rx = own_rx_u.build(num_qubits=num_w)
                own = own_rx(inp)
                assert torch.allclose(gt, own)
                errors.append((gt - own).abs().sum())
                assert isinstance(own_rx.__str__(), str)
                assert isinstance(own_rx_u.__str__(), str)
                own_rx.to_qasm()
                own_rx_u.to_qasm()
            for w2 in range(num_w):
                if w2 != w:
                    gt = forward_ground_truth(num_w, lambda: qml.CNOT(wires=[w, w2]), inp)
                    own_cnot_u = qandle.operators.CNOT(control=w, target=w2)
                    own_cnot = own_cnot_u.build(num_qubits=num_w)
                    own = own_cnot(inp)
                    assert torch.allclose(gt, own)
                    assert isinstance(own_cnot.__str__(), str)
                    assert isinstance(own_cnot_u.__str__(), str)

                    gt = forward_ground_truth(num_w, lambda: qml.CZ(wires=[w, w2]), inp)
                    own_cz_u = qandle.operators.CZ(control=w, target=w2)
                    own_cz = own_cz_u.build(num_qubits=num_w)
                    own = own_cz(inp)
                    assert torch.allclose(gt, own)
                    assert isinstance(own_cz.__str__(), str)
                    assert isinstance(own_cz_u.__str__(), str)

    info = f"errors max: {max(errors)}, avg: {sum(errors)/len(errors)}"
    assert max(errors) < 1e-5, f"Errors too high, {info}"


def test_reuploading():
    q = 3
    v = torch.tensor(0.123)
    torch.manual_seed(42)
    circuit1 = qandle.Circuit(layers=[qandle.RX(qubit=1, theta=v, remapping=None)], num_qubits=q)
    torch.manual_seed(42)
    circuit2 = qandle.Circuit(
        layers=[qandle.RX(qubit=1, name="reupload", remapping=None)], num_qubits=q
    )
    inp_unbatched = torch.rand(2**q, dtype=torch.cfloat)
    inp_unbatched = inp_unbatched / torch.linalg.norm(inp_unbatched)
    unb_1 = circuit1(inp_unbatched)
    unb_2 = circuit2(inp_unbatched, reupload=v)
    assert torch.allclose(unb_1, unb_2), f"unbatched: {unb_1}, {unb_2}, diff {unb_1-unb_2}"

    inp_batched = torch.rand(7, 2**q, dtype=torch.cfloat)
    inp_batched = inp_batched / torch.linalg.norm(inp_batched, dim=-1, keepdim=True)
    bat_1 = circuit1(inp_batched)
    bat_2 = circuit2(inp_batched, reupload=v)
    assert torch.allclose(bat_1, bat_2), f"batched: {bat_1}, {bat_2}, diff {bat_1-bat_2}"
    v_batched = torch.tensor([v, v, v, v, v, v, v])
    bat_3 = circuit2(inp_batched, reupload=v_batched)
    assert torch.allclose(bat_1, bat_3), f"batched: {bat_1}, {bat_3}, diff {bat_1-bat_3}"
    circuit2(inp_unbatched, reupload=v_batched)


def test_batched():
    torch.manual_seed(42)
    v = -2.3456
    errors = []
    for num_w in range(6):
        inp = torch.rand(13, 2**num_w, dtype=torch.cfloat)
        inp = inp / torch.linalg.norm(inp, dim=-1, keepdim=True)
        inp.requires_grad = True
        for w in range(num_w):
            for pl_op, own_op in (
                (qml.RX, qandle.operators.RX),
                (qml.RY, qandle.operators.RY),
                (qml.RZ, qandle.operators.RZ),
            ):
                gt = forward_ground_truth(num_w, lambda: pl_op(v, wires=w), inp)
                own_rx = own_op(qubit=w, theta=v, remapping=None).build(num_qubits=num_w)
                own = own_rx(inp)
                assert torch.allclose(
                    gt, own
                ), f"num_w: {num_w}, qubit: {w}, pl_op: {pl_op}, gt: {gt}, own: {own}, diff: {gt-own}"
                errors.append((gt - own).abs().sum())
            for w2 in range(num_w):
                if w2 != w:
                    gt = forward_ground_truth(num_w, lambda: qml.CNOT(wires=[w, w2]), inp)
                    own_cnot = qandle.operators.CNOT(control=w, target=w2).build(num_qubits=num_w)
                    own = own_cnot(inp)
                    assert torch.allclose(gt, own)
                    errors.append((gt - own).abs().sum())
                if w2 != w:
                    gt = forward_ground_truth(num_w, lambda: qml.SWAP(wires=[w, w2]), inp)
                    own_swap_ = qandle.operators.SWAP(a=w, b=w2)
                    assert isinstance(own_swap_.__str__(), str)
                    own_swap_.to_qasm()
                    own_swap = own_swap_.build(num_qubits=num_w)
                    assert isinstance(own_swap.__str__(), str)
                    own_swap.to_qasm()
                    own = own_swap(inp)
                    assert torch.allclose(gt, own)
                    errors.append((gt - own).abs().sum())
                if w2 != w:
                    gt = forward_ground_truth(num_w, lambda: qml.CZ(wires=[w, w2]), inp)
                    own_cz = qandle.operators.CZ(control=w, target=w2).build(num_qubits=num_w)
                    own = own_cz(inp)
                    assert torch.allclose(gt, own)
                    errors.append((gt - own).abs().sum())
    info = f"errors max: {max(errors)}, avg: {sum(errors)/len(errors)}"
    assert max(errors) < 1e-5, f"Errors too high, {info}"
