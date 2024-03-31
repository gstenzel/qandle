import torch
import qandle
import pytest


def test_shapes():
    h, w = 17, 18
    for c_in in [1, 3, 10]:
        for c_out in [1, 3, 10, 15]:
            conv1 = qandle.QConv(in_channels=c_in, out_channels=c_out, padding=1)
            for batch_size in [1, 10]:
                inp = torch.rand(batch_size, c_in, h, w, dtype=torch.float)
                out = conv1(inp)
                assert out.shape == (batch_size, c_out, h, w)
                assert out.dtype == torch.float

            conv2 = qandle.QConv(in_channels=c_in, out_channels=c_out, padding=0)
            for batch_size in [1, 10]:
                inp = torch.rand(batch_size, c_in, h, w, dtype=torch.float)
                out = conv2(inp)
                assert out.shape == (batch_size, c_out, h - 2, w - 2)
                assert out.dtype == torch.float


def test_errors():
    conv = qandle.QConv(in_channels=3, out_channels=10, kernel_size=3, padding=1)
    with pytest.raises(ValueError):
        inp = torch.rand(10, 4, 17, 18)  # wrong number of input channels
        conv(inp)
    with pytest.raises(ValueError):
        inp = torch.rand(10, 3, 17)  # wrong number of dimensions 1
        conv(inp)
    with pytest.raises(ValueError):
        inp = torch.rand(10, 3, 17, 18, 19)  # wrong number of dimensions 2
        conv(inp)
