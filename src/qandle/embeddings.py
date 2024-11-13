import torch
import abc
import typing
import random

import qandle.operators as op
import qandle.utils as utils
import qandle.qasm as qasm


class InputOperator(op.UnbuiltOperator, abc.ABC): ...


class InputOperatorBuilt(op.BuiltOperator, abc.ABC):
    @abc.abstractmethod
    def to_qasm(self) -> qasm.QasmRepresentation: ...

    def decompose(self):
        raise NotImplementedError(f"Decomposing {self.__class__} is not yet supported")

    def to_matrix(self):
        raise ValueError("Input operators do not have a matrix representation")


def _identity(input, *args, **kwargs):
    return input


class AmplitudeEmbeddingBuilt(InputOperatorBuilt):
    """
    Will ignore the state of the circuit and set the features as the state.
    """

    def __init__(
        self,
        name: str,
        num_qubits: int,
        normalize: bool,
        pad_with: typing.Union[float, None],
        padding,
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.normalize = normalize
        self.pad_with = pad_with
        self.padding = padding
        self.named = True
        self.name = name

    def forward(self, _=None, **kwargs):
        x = kwargs[self.name]
        pad_len = 2**self.num_qubits - x.shape[-1]
        x = self.padding(
            input=x,
            pad=(0, pad_len),
            mode="constant",
            value=self.pad_with,  # type: ignore
        )
        x = self.normalize(x, p=2, dim=-1)  # type: ignore
        x = torch.complex(real=x, imag=torch.zeros_like(x))
        return x

    def __str__(self) -> str:
        return "AmplitudeEmbedding"

    def to_qasm(self) -> qasm.QasmRepresentation:
        raise NotImplementedError("AmplitudeEmbedding is not yet supported")


class AmplitudeEmbedding(InputOperator):
    """
    Amplitude Embedding
    """

    def __init__(
        self,
        name: str,
        qubits: list,
        normalize: bool = False,
        pad_with: typing.Union[float, None] = None,
    ):
        self.name = name
        self.named = True
        self.qubits = qubits
        self.pad_with = pad_with
        if pad_with is not None:
            self.padding = torch.nn.functional.pad
        else:
            self.padding = _identity
        if normalize:
            self.normalize = torch.nn.functional.normalize
        else:
            self.normalize = _identity

    def build(self, num_qubits: int) -> AmplitudeEmbeddingBuilt:
        assert num_qubits == len(
            self.qubits
        ), "Current Implementation requires all qubits to be used."
        return AmplitudeEmbeddingBuilt(
            num_qubits=num_qubits,
            normalize=self.normalize,
            pad_with=self.pad_with,
            padding=self.padding,
            name=self.name,
        )

    def __str__(self) -> str:
        return "AmplitudeEmbedding"

    def to_qasm(self) -> qasm.QasmRepresentation:
        raise NotImplementedError("AmplitudeEmbedding is not yet supported")


class AngleEmbeddingBuilt(InputOperatorBuilt):
    """
    Will ignore the state of the circuit and set the features as the state.
    """

    def __init__(self, name: str, num_qubits: int, qubits: typing.List[int], rotation: str):
        super().__init__()
        self.num_qubits = num_qubits
        self.qubits = qubits
        self.named = True
        self.name = name

        _0 = torch.tensor((0 + 0j,), requires_grad=False)
        _i = torch.tensor((0 + 1j,), requires_grad=False)
        self.register_buffer("_0", _0, persistent=False)
        self.register_buffer("_i", _i, persistent=False)
        self.rotation = rotation
        self.rots = utils.parse_rot(rotation)
        a, b, self.a_op, self.b_op = self._get_rot_matrices(self.rots, num_qubits, qubits)
        self.register_buffer("_a", a.contiguous(), persistent=False)
        self.register_buffer("_b", b.contiguous(), persistent=False)
        z = torch.zeros(2**num_qubits, dtype=torch.cfloat)
        z[0] = 1
        self.register_buffer("_z", z, persistent=False)

    @staticmethod
    def _get_rot_matrices(rot, num_qubits, qubits):
        gates = [rot(qubit=w, remapping=None).build(num_qubits) for w in qubits]
        a = [g._a.T for g in gates]
        b = [g._b.T for g in gates]
        a = torch.stack(a, dim=0)
        b = torch.stack(b, dim=0)
        return a, b, gates[0].a_op, gates[0].b_op

    def forward(self, state=None, **kwargs):
        inp = kwargs[self.name]
        a_matrix = torch.einsum("dab,...d->...dab", self._a, self.a_op(inp / 2))
        b_matrix = torch.einsum("dab,...d->...dab", self._b, self.b_op(inp / 2))
        matrix = a_matrix + b_matrix
        prod = self.prod(matrix)
        return prod @ self._z

    def prod(self, inp):
        """
        Return the matrix product along the third-to-last dimension.
        TODO: Optimize this.
        """
        res = inp[..., 0, :, :]
        for i in range(1, inp.shape[-3]):
            res = res @ inp[..., i, :, :]
        return res

    def to_qasm(self) -> qasm.QasmRepresentation:
        reps = []
        r = hash(random.random())
        for w in self.qubits:
            reps.append(self.rots(qubit=w, name=f"angle_{r}_{w}").to_qasm())
        return reps  # type: ignore

    def __str__(self) -> str:
        return f"AngleEmbedding_{self.rotation}"

    def decompose(self):
        reps = []
        r = hash(random.random())
        for w in self.qubits:
            reps.append(self.rots(qubit=w, name=f"angle_{r}_{w}"))
        return reps


class AngleEmbedding(InputOperator):
    """
    Angle Embedding
    """

    def __init__(
        self, name: str, qubits: typing.Union[typing.List[int], None] = None, rotation="rx"
    ):
        self.name = name
        self.named = True
        self.qubits = qubits
        self.rotation = rotation

    def build(self, num_qubits: int) -> AngleEmbeddingBuilt:
        if self.qubits is None:
            qubits = list(range(num_qubits))
        else:
            qubits = self.qubits
        return AngleEmbeddingBuilt(
            name=self.name, qubits=qubits, num_qubits=num_qubits, rotation=self.rotation
        )

    def to_qasm(self) -> qasm.QasmRepresentation:
        reps = []
        r = hash(random.random())
        for w in self.qubits:
            reps.append(self.rots(qubit=w, name=f"angle_{r}_{w}").to_qasm())
        return reps  # type: ignore

    def __str__(self) -> str:
        return f"AngleEmbedding_{self.rotation}"
