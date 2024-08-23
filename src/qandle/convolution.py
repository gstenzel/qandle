import torch
import typing
import math

import qandle
import einops


class QConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: typing.Union[typing.Tuple[int, int], int] = (3, 3),
        padding: typing.Union[typing.Tuple[int, int], int] = (1, 1),
        qdepth: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.qdepth = qdepth
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, padding=self.padding)
        qubits_for_inp = math.ceil(
            math.log2(self.kernel_size[0] * self.kernel_size[1] * in_channels)
        )
        qubits_for_out = math.ceil(math.log2(out_channels))
        self.qubits = max(qubits_for_inp, qubits_for_out, 1)
        amp = qandle.AmplitudeEmbedding(qubits=list(range(self.qubits)), pad_with=0, name="emb")
        sel = qandle.StronglyEntanglingLayer(qubits=list(range(self.qubits)), depth=self.qdepth)
        mes = qandle.MeasureJointProbability()

        self.qcircuit = qandle.Circuit(
            num_qubits=self.qubits,
            layers=[amp, sel, mes],
        )
        self.ein1 = "batch channel feat -> (batch feat) channel"
        self.ein2 = "(batch h_out w_out) channel -> batch channel h_out w_out"
        self.almost_zero = 0.001

    def _post_process(self, x):
        x = x[:, : self.out_channels, :, :]  # remove padding
        x = x * self.out_channels / 2  # rescale
        return x

    def forward(self, x):
        b, c_in, h_in, w_in = x.shape
        h_out = h_in + 2 * self.padding[0] - self.kernel_size[0] + 1  # output height
        w_out = w_in + 2 * self.padding[1] - self.kernel_size[1] + 1  # output width
        if c_in != self.in_channels:
            raise ValueError(f"Input channels {c_in} does not match in_channels {self.in_channels}")
        x = self.unfold(x)
        x = einops.rearrange(x, self.ein1)
        x = x + self.almost_zero  # avoid zero input
        x = self.qcircuit(emb=x)
        x = einops.rearrange(x, self.ein2, batch=b, h_out=h_out, w_out=w_out)
        x = self._post_process(x)
        return x
