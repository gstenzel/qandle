import torch
import qandle


def test_bitflip_channel_deterministic():

    state0 = torch.tensor([1.0, 0.0], dtype=torch.cfloat)
    flip = qandle.BitFlip(p=1.0, qubit=0).build(num_qubits=1)
    out = flip(state0)
    assert torch.allclose(out, torch.tensor([0.0, 1.0], dtype=torch.cfloat))

    noflip = qandle.BitFlip(p=0.0, qubit=0).build(num_qubits=1)
    out = noflip(state0)
    assert torch.allclose(out, state0)


def test_bitflip_channel_probability():
    state0 = torch.tensor([1.0, 0.0], dtype=torch.cfloat)
    flip = qandle.BitFlip(p=0.5, qubit=0).build(num_qubits=1)
    out = flip(state0)
    meas = qandle.MeasureProbabilityBuilt(num_qubits=1)
    probs = meas(out)
    assert torch.allclose(probs, torch.tensor([0.5]))

