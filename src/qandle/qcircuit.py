import torch
from qandle import drawer
from qandle import splitter
from qandle import operators
from qandle import qasm
from qandle import utils
from qandle.backends import (
    MPSBackend,
    QuantumBackend,
    StateVectorBackend,
)
import warnings
import typing

_BACKENDS = {
    "statevector": StateVectorBackend,
    "mps": MPSBackend,
}


def _make_backend(backend: str | QuantumBackend, n_qubits: int, **kwargs) -> QuantumBackend:
    if isinstance(backend, QuantumBackend):
        return backend
    return _BACKENDS[backend](n_qubits=n_qubits, **kwargs)


def _apply_backend_layer(layer: torch.nn.Module, backend: QuantumBackend, **kwargs) -> None:
    """Apply a built layer on a :class:`QuantumBackend`."""
    if hasattr(layer, "qubit"):
        backend.apply_1q(layer.to_matrix(**kwargs), layer.qubit)
    elif hasattr(layer, "c") and hasattr(layer, "t") and not hasattr(layer, "c2"):
        backend.apply_2q(layer.to_matrix(**kwargs), layer.c, layer.t)
    elif hasattr(layer, "a") and hasattr(layer, "b"):
        backend.apply_2q(layer.to_matrix(**kwargs), layer.a, layer.b)
    else:
        raise NotImplementedError("Only 1- and 2-qubit gates are supported.")

__all__ = ["Circuit"]


class Circuit(torch.nn.Module):
    class DummyUnbuiltCircuit:
        """Helper class to decompose a circuit with subcircuits without building it."""

        def __init__(self, num_qubits: int, layers: typing.List[operators.Operator]):
            self.num_qubits = num_qubits
            self.layers = []
            for layer in layers:
                if hasattr(layer, "decompose"):
                    self.layers.extend(layer.decompose())  # type: ignore
                else:
                    self.layers.append(layer)

        def decompose(self):
            return self

    def __init__(
        self,
        layers: typing.List[operators.Operator],
        num_qubits=None,
        split_max_qubits=0,
        circuit=None,
    ):
        super().__init__()
        self.name = ""
        self.named = True
        if not hasattr(layers, "__iter__"):
            layers = [layers]
        if num_qubits is not None:
            self.num_qubits = num_qubits
        else:
            self.num_qubits = len(self._used_qubits(layers))
        if circuit is not None:
            self.circuit = circuit
        else:
            if split_max_qubits > 0 and self.num_qubits > split_max_qubits:
                self.circuit = SplittedCircuit(
                    num_qubits=self.num_qubits,
                    subcircuitdict=splitter.split(
                        self.DummyUnbuiltCircuit(self.num_qubits, layers),
                        max_qubits=split_max_qubits,
                    ),
                )
            else:
                self.circuit = UnsplittedCircuit(self.num_qubits, layers)

    def forward(
        self,
        state=None,
        *,
        backend: str | QuantumBackend | None = None,
        backend_kwargs: dict | None = None,
        diff_method: str | None = None,
        **kwargs,
    ):
        if backend is not None:
            be = _make_backend(backend, self.num_qubits, **(backend_kwargs or {}))
            circ = (
                self.circuit
                if isinstance(self.circuit, UnsplittedCircuit)
                else self.circuit.decompose()
            )
            be.allocate(self.num_qubits)
            for layer in circ.layers:
                _apply_backend_layer(layer, be, **kwargs)
            return be

        if diff_method == "parameter_shift":
            from .gradients import parameter_shift_forward

            return parameter_shift_forward(self.circuit, state, **kwargs)

        return self.circuit.forward(state, **kwargs)

    def to_matrix(self, **kwargs):
        return self.circuit.to_matrix(**kwargs)

    def __matmul__(self, x):
        """Allows the circuit to be multiplied with a state tensor, directly calling forward. Not compatible with named inputs."""
        return self.circuit.forward(x)

    @staticmethod
    def _used_qubits(layers: typing.List[operators.Operator]) -> typing.Set[int]:
        import qandle.measurements

        qubits = set()
        for layer in layers:
            if isinstance(layer, operators.CNOT):
                qubits.add(layer.c)
                qubits.add(layer.t)
            elif isinstance(layer, operators.SWAP):
                qubits.add(layer.a)
                qubits.add(layer.b)
            elif isinstance(layer, operators.Controlled):
                qubits.add(layer.c)
                qubits.add(layer.t.qubits)
            elif hasattr(layer, "qubit"):
                qubits.add(layer.qubit)  # type: ignore
            elif hasattr(layer, "num_qubits"):
                qubits.update(range(layer.num_qubits))  # type: ignore
            elif hasattr(layer, "qubits"):
                qubits.update(layer.qubits)
            elif isinstance(
                layer,
                (
                    qandle.measurements.UnbuiltMeasurement,
                    qandle.measurements.BuiltMeasurement,
                ),
            ):
                pass  # ignore measurements, as they act on all qubits anyway
            else:
                raise ValueError(
                    f"Unknown layer type {type(layer)}, number of qubits could not be inferred. Pass :code:`num_qubits` to the circuit."
                )
        if len(qubits) == 0:
            raise ValueError(
                "Number of qubits could not be inferred from layers. Please provide num_qubits to the circuit directly."
            )
        return qubits

    def draw(self):
        """Shorthand for drawer.draw(self.circuit)"""
        return drawer.draw(self.circuit)

    def split(self, max_qubits):
        if isinstance(self.circuit, UnsplittedCircuit):
            splitdict = splitter.split(self.circuit.decompose(), max_qubits=max_qubits)
            return Circuit(
                layers=[],
                num_qubits=self.num_qubits,
                circuit=SplittedCircuit(self.num_qubits, splitdict),
            )
        else:
            warnings.warn("Circuit already split, returning original circuit.")
            return self

    def decompose(self):
        return Circuit(layers=[], num_qubits=self.num_qubits, circuit=self.circuit.decompose())

    def to_qasm(self):
        return self.circuit.to_qasm()

    def to_openqasm2(self) -> str:
        return qasm.convert_to_qasm(self, qasm_version=2, include_header=True)

    def to_openqasm3(self) -> str:
        return qasm.convert_to_qasm(self, qasm_version=3, include_header=True)


class UnsplittedCircuit(torch.nn.Module):
    """
    Main class for quantum circuits. This class is a torch.nn.Module, so it can be used like any other PyTorch module.
    """

    def __init__(self, num_qubits: int, layers: list):
        """
        Create a new quantum circuit, building all layers.
        """
        super().__init__()
        self.num_qubits = num_qubits
        self.state = torch.zeros(2**num_qubits, dtype=torch.complex64)
        self.state[0] = 1
        layers_built = self._build_layers(layers, num_qubits)
        self.layers = torch.nn.ModuleList(layers_built)

    @staticmethod
    def _build_layers(layers: list, num_qubits: int) -> typing.List[torch.nn.Module]:
        layers_built = []
        for la in layers:
            if hasattr(la, "build"):
                layers_built.append(la.build(num_qubits=num_qubits))
            else:
                layers_built.append(la)
        return layers_built

    def forward(self, state=None, **kwargs):
        """
        Run the circuit. If state is None, the initial state is used. Supply specific inputs to specific gates using the kwargs, of the form name:torch.Tensor
        """
        if state is None:
            state = self.state
        for mod in self.layers:
            if mod.named:
                state = mod(state, **kwargs)
            else:
                state = mod(state)
        return state

    def to_qasm(self) -> qasm.QasmRepresentation:
        reps = []
        for layer in self.layers:
            if hasattr(layer, "to_qasm"):
                reps.extend(layer.to_qasm())
        return reps  # type: ignore

    def decompose(self) -> "UnsplittedCircuit":
        """
        Decompose the circuit into a circuit without any subcircuits, returning a new, flat Circuit.
        """
        new_layers = []
        for layer in self.layers:
            if hasattr(layer, "decompose"):
                if isinstance(layer, (Circuit)):
                    new_layers.extend(layer.decompose().circuit.layers)
                else:
                    new_layers.extend(layer.decompose())
            else:
                new_layers.append(layer)
        return UnsplittedCircuit(self.num_qubits, new_layers)

    def split(self, **kwargs) -> typing.Union["SplittedCircuit", "UnsplittedCircuit"]:
        """
        Split the circuit into subcircuits, returning a CuttedCircuit if the circuit was split, or the original circuit if not.
        """
        if self.num_qubits == 1:
            warnings.warn("Can't split circuit with only 1 qubit")
            return self
        splitted = splitter.split(self, **kwargs)
        return SplittedCircuit(self.num_qubits, splitted)

    def to_matrix(self, **kwargs):
        """
        Get the matrix representation of the circuit.
        """
        return utils.reduce_dot(*[mod.to_matrix(**kwargs) for mod in self.layers])


class SplittedCircuit(UnsplittedCircuit):
    """
    Circuit with subcircuits, created by splitting a Circuit.
    """

    def __init__(
        self,
        num_qubits: int,
        subcircuitdict: typing.Dict[int, splitter.SubcircuitContainer],
    ):
        torch.nn.Module.__init__(self)
        self.num_qubits = num_qubits
        self.state = torch.zeros(2**num_qubits, dtype=torch.complex64)
        self.state[0] = 1

        self._check_qubit_order(subcircuitdict)

        self.subcircuits = torch.nn.ModuleList(
            [
                Subcircuit(num_qubits, subcircuitdict[i].mapping, subcircuitdict[i].layers)
                for i in subcircuitdict
            ]
        )

    @staticmethod
    def _check_qubit_order(subcircuitdict):
        for sc in subcircuitdict.values():
            keys_sorted = sorted(sc.mapping.keys())
            values = [sc.mapping[k] for k in keys_sorted]
            assert values == sorted(values), f"Mapping {sc.mapping} is not ordered."

    def forward(self, inp=None, **kwargs):
        if inp is None:
            inp = self.state
        for sc in self.subcircuits:
            inp = sc(inp, **kwargs)
        return inp

    def to_qasm(self) -> qasm.QasmRepresentation:
        reps = [sc.to_qasm() for sc in self.subcircuits]
        return reps  # type: ignore

    def to_matrix(self, **kwargs):
        raise NotImplementedError("Can't get matrix representation of a splitted circuit.")


class Subcircuit(torch.nn.Module):
    """
    Submodule for CuttedCircuit, contains a single subcircuits
    """

    def __init__(self, num_qubits_parent: int, mapping: typing.Dict[int, int], layers: list):
        """
        num_qubits_parent: number of qubits in the parent circuit
        mapping: mapping of qubits from parent circuit to subcircuit
        layers: layers of the subcircuit
        """
        super().__init__()
        self.num_qubits_parent = num_qubits_parent
        self.mapping = mapping
        layers_built = UnsplittedCircuit._build_layers(layers, num_qubits=len(mapping))
        self.to_matrix, self.to_state = utils.get_matrix_transforms(
            num_qubits_parent, list(mapping.keys())
        )

        self.layers = torch.nn.ModuleList(layers_built)

    def forward(self, x, **kwargs):
        batched = x.ndim == 2
        if not batched:
            x = x.unsqueeze(0)
        x = self.to_matrix(x)
        for mod in self.layers:
            if mod.named:
                x = mod(x, **kwargs)
            else:
                x = mod(x)
        x = self.to_state(x)
        if not batched:
            x = x.squeeze(0)
        return x

    def to_qasm(self) -> qasm.QasmRepresentation:
        import re

        reps = [mod.to_qasm() for mod in self.layers if hasattr(mod, "to_qasm")]
        for r in reps:
            if r.qubit not in ("", None):
                r.qubit = [k for k, v in self.mapping.items() if v == r.qubit][0]
            else:
                rstr = r.gate_str

                def new_qubit(x):
                    old_q = int(x.group(1))
                    new_q = [k for k, v in self.mapping.items() if v == old_q][0]
                    return f"q[{new_q}]"

                rstr = re.sub(r"q\[(\d+)\]", new_qubit, rstr)
                r.gate_str = rstr
        return reps  # type: ignore
