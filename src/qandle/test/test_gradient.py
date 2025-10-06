import torch
import pytest
import qandle


def build_single_qubit_circuit(gate_cls):
    """Simple circuit with a single parametrized gate and probability measurement."""
    return qandle.Circuit(
        layers=[
            gate_cls(qubit=0, remapping=lambda x: x),
            qandle.MeasureProbability(),
        ],
        num_qubits=1,
    )


def trainable_params(circuit):
    """Return all parameters of *circuit* that require gradients."""
    return [p for p in circuit.parameters() if p.requires_grad]


@pytest.mark.parametrize("gate_cls", [qandle.RX, qandle.RY, qandle.RZ])
def test_parameter_shift_matches_autograd(gate_cls):
    """Single-parameter gradient: compare parameter-shift against autograd."""
    torch.manual_seed(0)

    circuit = build_single_qubit_circuit(gate_cls)
    [param] = trainable_params(circuit)

    ref_out = circuit()
    if ref_out.ndim == 0:
        ref_out_scalar = ref_out
    else:
        ref_out_scalar = ref_out[0]
    (grad_ref,) = torch.autograd.grad(ref_out_scalar, param, retain_graph=True)

    ps_out = circuit(diff_method="parameter_shift")
    if ps_out.ndim == 0:
        ps_out_scalar = ps_out
    else:
        ps_out_scalar = ps_out[0]
    (grad_ps,) = torch.autograd.grad(ps_out_scalar, param)

    assert torch.allclose(grad_ps, grad_ref, rtol=1e-5, atol=1e-6)


def test_parameter_shift_multiple_parameters():
    """Multiple-parameter gradient: compare parameter-shift against autograd."""
    torch.manual_seed(42)

    circuit = qandle.Circuit(
        layers=[
            qandle.RX(qubit=0, remapping=lambda x: x),
            qandle.RY(qubit=0, remapping=lambda x: x),
            qandle.MeasureProbability(),
        ],
        num_qubits=1,
    )
    params = trainable_params(circuit)
    assert len(params) == 2

    ref_out = circuit()
    if ref_out.ndim == 0:
        ref_out_scalar = ref_out
    else:
        ref_out_scalar = ref_out[0]
    grads_ref = torch.autograd.grad(ref_out_scalar, params, retain_graph=True)

    ps_out = circuit(diff_method="parameter_shift")
    if ps_out.ndim == 0:
        ps_out_scalar = ps_out
    else:
        ps_out_scalar = ps_out[0]
    grads_ps = torch.autograd.grad(ps_out_scalar, params)

    for g_ref, g_ps in zip(grads_ref, grads_ps):
        assert torch.allclose(g_ps, g_ref, rtol=1e-5, atol=1e-6)