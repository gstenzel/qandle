import torch
import abc
import typing
import qw_map
import einops.layers.torch as einl

import qandle.utils as utils
import qandle.config as config
import qandle.errors as errors
import qandle.qasm as qasm

__all__ = [
    "Operator",
    "RX",
    "RY",
    "RZ",
    "CNOT",
    "Reset",
    "SWAP",
    "U",
    "CustomGate",
    "BUILT_CLASS_RELATION",
]

matrixbuilder = typing.Tuple[
    torch.Tensor, torch.Tensor, typing.Callable, typing.Callable
]


class AbstractNoForward(abc.ABCMeta, utils.do_not_implement("forward", "backward")):
    pass


class Operator(abc.ABC):
    """Everything that can be applied to a state."""

    named = False

    @abc.abstractmethod
    def __str__(self) -> str:
        """Returns a string representation of the operator."""

    def __repr__(self) -> str:
        return self.__str__()

    @abc.abstractmethod
    def to_qasm(self) -> qasm.QasmRepresentation:
        """Returns the OpenQASM2 representation of the operator."""


class UnbuiltOperator(Operator, abc.ABC):
    """Container class for operators that have not been built yet."""

    @abc.abstractmethod
    def build(self, num_qubits, **kwargs) -> "BuiltOperator":
        """Builds the operator, i.e. converts it to a torch.nn.Module."""


class BuiltOperator(Operator, torch.nn.Module, abc.ABC):
    """Container class for operators that have been built."""

    @abc.abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Applies the operator to the state."""


class U(UnbuiltOperator):
    def __init__(self, qubit: int, matrix: torch.Tensor):
        self.qubit = qubit
        self.matrix = matrix

    def __str__(self) -> str:
        return f"U_{self.qubit}"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return qasm.QasmRepresentation(gate_str="U", qubit=self.qubit)

    def build(self, num_qubits, **kwargs) -> "BuiltU":
        return BuiltU(qubit=self.qubit, matrix=self.matrix, num_qubits=num_qubits)


class BuiltU(BuiltOperator):
    def __init__(
        self,
        qubit: int,
        matrix: torch.Tensor,
        num_qubits: int,
        self_description: str = "U",
    ):
        super().__init__()
        self.qubit = qubit
        self.num_qubits = num_qubits
        self.description = self_description

        self.original_matrix = matrix
        m = torch.eye(1)
        for i in range(self.num_qubits):
            m = torch.kron(m, matrix if i == self.qubit else torch.eye(2))
        self.matrix = m.to(torch.cfloat).contiguous()

    def __str__(self) -> str:
        return f"{self.description}_{self.qubit}"

    def to_qasm(self) -> qasm.QasmRepresentation:
        u = f"{self.original_matrix[0, 0]:.2f}, {self.original_matrix[0, 1]:.2f}, {self.original_matrix[1, 0]:.2f}, {self.original_matrix[1, 1]:.2f}"
        definition = (
            "gate "
            + self.description
            + "_"
            + str(self.qubit)
            + " q["
            + str(self.qubit)
            + "] {{ "
            + "U("
            + u
            + ") }}"
        )
        return qasm.QasmRepresentation(gate_str=definition, qubit=self.qubit)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state @ self.matrix


CustomGate = BuiltU
"""
Define a custom single qubit gate. 
This module is always built, and therefore might have a big memory footprint. 

Attributes:
    qubit : int
        The index of the qubit this gate operates on.
    matrix : torch.Tensor
        The matrix representation of the gate. Must be a 2x2 unitary matrix.
    num_qubits : int
        The number of qubits in the circuit.
    self_description : str
        The description of the gate. Default is "U". This name will be used in the string representation of the gate and in the OpenQASM2 representation.

"""


class UnbuiltParametrizedOperator(UnbuiltOperator, metaclass=AbstractNoForward):
    """Container class for parametrized operators that have not been built yet."""

    def __init__(
        self,
        qubit: int,
        theta: typing.Union[float, torch.Tensor, None] = None,
        name: typing.Union[str, None] = None,
        **kwargs,
    ):
        """
        Creates a parametrized operator.

        """
        assert isinstance(qubit, int), "qubit must be an integer"
        assert qubit >= 0, "qubit must be >= 0"
        if isinstance(theta, (float, int)):
            theta = torch.tensor(theta, requires_grad=True, dtype=torch.float)
        remapping = kwargs.get("remapping", config.DEFAULT_MAPPING)
        if remapping is None:
            remapping = qw_map.none
        self.qubit = qubit
        self.name = name
        self.named = name is not None
        self.theta = theta
        self.remapping = remapping

    def __str__(self) -> str:
        if self.name is None:  # named
            return f"{self.__class__.__name__}{self.qubit} ({self.name})"
        else:  # unnamed
            if self.theta is None:
                return f"{self.__class__.__name__}{self.qubit}"
            else:
                return f"{self.__class__.__name__}{self.qubit} ({self.remapping(self.theta).item():.2f})"

    def to_qasm(self) -> qasm.QasmRepresentation:
        if self.named:
            return qasm.QasmRepresentation(
                gate_str=self.__class__.__name__.lower(),
                qubit=self.qubit,
                qasm3_inputs=self.name,  # type: ignore # if name is None, named would be False
            )

        if self.theta is not None:
            return qasm.QasmRepresentation(
                gate_str=self.__class__.__name__.lower(),
                qubit=self.qubit,
                gate_value=self.remapping(self.theta).item(),
            )

        raise errors.UnbuiltGateError(
            "This gate has no parameter. Set parameter or build or set name before converting to OpenQASM2."
        )

    def build(self, num_qubits, **kwargs) -> "BuiltParametrizedOperator":
        return BUILT_CLASS_RELATION[self.__class__](
            qubit=self.qubit,
            initialtheta=self.theta,
            name=self.name,
            remapping=self.remapping,
            num_qubits=num_qubits,
        )


class BuiltParametrizedOperator(BuiltOperator, abc.ABC):
    def __init__(
        self,
        qubit: int,
        remapping: typing.Callable,
        num_qubits: int,
        initialtheta: typing.Union[torch.Tensor, None] = None,
        name: typing.Union[str, None] = None,
    ):
        super().__init__()
        self.qubit = qubit
        if initialtheta is None:
            initialtheta = torch.rand(1)
        self.name = name  # currently unused
        self.named = name is not None  # faster than checking if name is None every time
        self.theta = torch.nn.Parameter(initialtheta, requires_grad=True)
        self.remapping = remapping
        self.num_qubits = num_qubits
        self.unbuilt_class = BUILT_CLASS_RELATION.T[self.__class__]

        self.register_buffer(
            "_i", torch.tensor(1j, dtype=torch.cfloat), persistent=False
        )
        a, b, self.a_op, self.b_op = self.matrix_builder()
        self.register_buffer("_a", a.T.contiguous(), persistent=False)
        self.register_buffer("_b", b.T.contiguous(), persistent=False)

    def __str__(self) -> str:
        base = f"{self.unbuilt_class.__name__}_{self.qubit}"
        if self.named:
            return f"{base} ({self.name})"
        else:
            if self.theta is None:
                return f"{base}"
            else:
                return f"{base} ({self.remapping(self.theta).item():.2f})"

    def to_qasm(self) -> qasm.QasmRepresentation:
        if self.named:
            return qasm.QasmRepresentation(
                gate_str=self.unbuilt_class.__name__.lower(),
                qubit=self.qubit,
                qasm3_inputs=self.name,  # type: ignore # if name is None, named would be False
            )

        else:
            return qasm.QasmRepresentation(
                gate_str=self.unbuilt_class.__name__.lower(),
                qubit=self.qubit,
                gate_value=self.remapping(self.theta).item(),
            )

    def get_matrix(self, **kwargs) -> torch.Tensor:
        if self.named:
            t = kwargs[self.name] / 2  # type: ignore # if name is None, named would be False
            if t.dim() == 1:
                t = t.unsqueeze(-1).unsqueeze(-1)
        else:
            t = self.remapping(self.theta) / 2
        a_matrix = self._a * self.a_op(t)
        b_matrix = self._b * self.b_op(t)
        matrix = a_matrix + b_matrix
        return matrix

    def hydrated(
        self, special: typing.Union[torch.Tensor, None, typing.Tuple] = None
    ) -> torch.Tensor:
        if special is None:
            special = torch.eye(2)
        if not isinstance(special, torch.Tensor):
            special = torch.tensor(special)
        matrix = torch.eye(1)
        for i in range(self.num_qubits):
            matrix = torch.kron(matrix, special if i == self.qubit else torch.eye(2))
        return matrix.to(torch.cfloat)

    def forward(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        mat = self.get_matrix(**kwargs)
        if mat.dim() == 2:
            res = state @ mat
        else:
            if state.dim() == 1:
                state = state.unsqueeze(0)
                # raise ValueError("One of the named matrices received batched input, but the state is not batched. Please batch the state and the named parameters or neither.")
            res = (state.unsqueeze(1) @ mat).squeeze(1)
        return res

    @abc.abstractmethod
    def matrix_builder(self) -> matrixbuilder:
        """Returns the matrix builder for the gate."""


class RX(UnbuiltParametrizedOperator):
    """
    Parametrized RX gate, i.e. a rotation around the x-axis of the Bloch sphere.

    This class represents a parametrized RX gate in a quantum circuit.

    Used by :class:`qandle.qcircuit.QCircuit` to build the circuit.

    Attributes:
        qubit : int
            The index of the qubit this gate operates on.
        theta : float, torch.Tensor, optional
            The parameter of the RX gate, by default None. If None, a random parameter :math:`[0, 1]` is chosen.
        name : str, optional
            The name of the operator, by default None. If None, the operator is not named and does not accept named inputs.
        remapping : Callable, optional
            A function that remaps the parameter theta, by default config.DEFAULT_MAPPING. To disable remapping, pass :code:`qw_map.none` or :code:`lambda x: x`.
    """


class RY(UnbuiltParametrizedOperator):
    """
    Parametrized RY gate, i.e. a rotation around the y-axis of the Bloch sphere.

    This class represents a parametrized RY gate in a quantum circuit.

    Used by :class:`qandle.qcircuit.QCircuit` to build the circuit.

    Attributes
        qubit : int
            The index of the qubit this gate operates on.
        theta : float, torch.Tensor, optional
            The parameter of the RY gate, by default None. If None, a random parameter :math:`[0, 1]` is chosen.
        name : str, optional
            The name of the operator, by default None. If None, the operator is not named and does not accept named inputs.
        remapping : Callable, optional
            A function that remaps the parameter theta, by default config.DEFAULT_MAPPING. To disable remapping, pass :code:`qw_map.none` or :code:`lambda x: x`.
    """


class RZ(UnbuiltParametrizedOperator):
    """
    Parametrized RZ gate, i.e. a rotation around the z-axis of the Bloch sphere.

    This class represents a parametrized RZ gate in a quantum circuit.

    Used by :class:`qandle.qcircuit.QCircuit` to build the circuit.

    Attributes
        qubit : int
            The index of the qubit this gate operates on.
        theta : float, torch.Tensor, optional
            The parameter of the RZ gate, by default None. If None, a random parameter :math:`[0, 1]` is chosen.
        name : str, optional
            The name of the operator, by default None. If None, the operator is not named and does not accept named inputs.
        remapping : Callable, optional
            A function that remaps the parameter theta, by default config.DEFAULT_MAPPING. To disable remapping, pass :code:`qw_map.none` or :code:`lambda x: x`.
    """


class BuiltRX(BuiltParametrizedOperator):
    def matrix_builder(self) -> matrixbuilder:
        return (
            -self.hydrated(((0, 1), (1, 0))) * self._i,
            self.hydrated(),
            torch.sin,
            torch.cos,
        )


class BuiltRY(BuiltParametrizedOperator):
    def matrix_builder(self) -> matrixbuilder:
        return (
            self.hydrated(((0, -1), (1, 0))),
            self.hydrated(),
            torch.sin,
            torch.cos,
        )


class BuiltRZ(BuiltParametrizedOperator):
    def matrix_builder(self) -> matrixbuilder:
        return (
            self.hydrated(((1, 0), (0, 0))),
            self.hydrated(((0, 0), (0, 1))),
            lambda theta: torch.exp(-self._i * theta),
            lambda theta: torch.exp(self._i * theta),
        )


class CNOT(UnbuiltOperator):
    """CNOT gate."""

    def __init__(self, control: int, target: int):
        assert control != target, "Control and target must be different"
        self.c = control
        self.t = target

    def __str__(self) -> str:
        return f"CNOT {self.c}|{self.t}"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return qasm.QasmRepresentation(gate_str=f"cx q[{self.c}], q[{self.t}]")

    def build(self, num_qubits, **kwargs) -> "BuiltCNOT":
        return BuiltCNOT(control=self.c, target=self.t, num_qubits=num_qubits)


class BuiltCNOT(BuiltOperator):
    def __init__(self, control: int, target: int, num_qubits: int):
        super().__init__()
        self.c = control
        self.t = target
        self.num_qubits = num_qubits
        self.register_buffer(
            "_M", self._calculate_matrix(control, target, num_qubits), persistent=False
        )

    @staticmethod
    def _calculate_matrix(c: int, t: int, num_qubits: int):
        M = torch.zeros(2**num_qubits, 2**num_qubits) * 0j
        c2, t2 = num_qubits - c - 1, num_qubits - t - 1
        for i in range(2**num_qubits):
            if i & (1 << c2):
                M[i, i ^ (1 << t2)] = 1
            else:
                M[i, i] = 1
        return M

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state @ self._M

    def __str__(self) -> str:
        return CNOT(self.c, self.t).__str__()

    def to_qasm(self) -> qasm.QasmRepresentation:
        return CNOT(self.c, self.t).to_qasm()


class BuiltReset(BuiltOperator):
    def __init__(self, qubit: int, num_qubits: int):
        super().__init__()
        self.qubit = qubit
        self.num_qubits = num_qubits
        self.to_matrix = einl.Rearrange(
            "batch (a ax b) -> batch (a b) ax", ax=2, a=2**qubit
        )
        self.to_state = einl.Rearrange(
            "batch (a b) ax -> batch (a ax b)", ax=2, a=2**qubit
        )

    def forward(self, state: torch.Tensor):
        unbatched = state.dim() == 1
        if unbatched:
            state = state.unsqueeze(0)
        state = self.to_matrix(state)
        new_state = torch.zeros_like(state)
        new_state[..., 0] = torch.linalg.norm(state, dim=-1)
        state = self.to_state(new_state)
        if unbatched:
            state = state.squeeze(0)
        return state

    def __str__(self) -> str:
        return Reset(self.qubit).__str__()

    def to_qasm(self) -> qasm.QasmRepresentation:
        return Reset(self.qubit).to_qasm()


class Reset(UnbuiltOperator):
    def __init__(self, qubit: int):
        self.qubit = qubit

    def __str__(self) -> str:
        return f"Reset {self.qubit}"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return qasm.QasmRepresentation(gate_str="reset", qubit=self.qubit)

    def build(self, num_qubits, **kwargs) -> "BuiltReset":
        return BUILT_CLASS_RELATION[self.__class__](
            qubit=self.qubit, num_qubits=num_qubits
        )


class BuiltSWAP(BuiltOperator):
    def __init__(self, a: int, b: int, num_qubits: int):
        super().__init__()
        self.a = a
        self.b = b
        self.num_qubits = num_qubits
        self.register_buffer(
            "_M", self._calculate_matrix(a, b, num_qubits), persistent=False
        )

    @staticmethod
    def _calculate_matrix(a: int, b: int, num_qubits: int):
        swap_matrix = torch.eye(2**num_qubits)
        a, b = num_qubits - a - 1, num_qubits - b - 1
        for i in range(2**num_qubits):
            swapped_i = i
            if ((i >> a) & 1) != ((i >> b) & 1):
                swapped_i = i ^ ((1 << a) | (1 << b))
            swap_matrix[i, i] = 0
            swap_matrix[i, swapped_i] = 1
        return swap_matrix.to(torch.cfloat)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state @ self._M

    def __str__(self) -> str:
        return f"SWAP {self.a}|{self.b}"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return qasm.QasmRepresentation(gate_str=f"swap q[{self.a}], q[{self.b}]")


class SWAP(UnbuiltOperator):
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b

    def __str__(self) -> str:
        return f"Swap {self.a}|{self.b}"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return qasm.QasmRepresentation(gate_str=f"swap q[{self.a}], q[{self.b}]")

    def build(self, num_qubits, **kwargs) -> "BuiltSWAP":
        return BuiltSWAP(a=self.a, b=self.b, num_qubits=num_qubits)


class rdict(dict):
    """reversible dict"""

    @property
    def T(self):
        return {v: k for k, v in self.items()}


BUILT_CLASS_RELATION = rdict(
    {
        UnbuiltOperator: BuiltOperator,
        UnbuiltParametrizedOperator: BuiltParametrizedOperator,
        RX: BuiltRX,
        RY: BuiltRY,
        RZ: BuiltRZ,
        CNOT: BuiltCNOT,
        Reset: BuiltReset,
        U: BuiltU,
    }
)
