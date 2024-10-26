import typing
import qw_map
import torch


import qandle.config as config
import qandle.operators as op
import qandle.utils as utils
import qandle.qasm as qasm
from qandle.ansaetze.ansatz import BuiltAnsatz


__all__ = ["StronglyEntanglingLayerBudget"]


class StronglyEntanglingLayerBudget(BuiltAnsatz):
    """
    A strongly entangling layer, inspired by `this paper <https://arxiv.org/abs/1804.00633>`_.
    Consists of a series of single-qubit rotations on each qubit specified in the :code:`qubits` parameter, followed by a series of control gates. Instead of specifying a depth, a :code:`param_budget` is specified, which determines the number of parameters in the circuit. The control gates are inserted every :code:`control_gate_spacing` layers, quickly allowing to change the entanglement structure of the circuit.
    """

    def __init__(
        self,
        num_qubits_total: int,
        qubits: typing.Union[typing.List[int], None] = None,
        param_budget: int = 1,
        control_gate=op.CNOT,
        control_gate_spacing=1,
        rotations=["rz", "ry"],
        remapping: typing.Union[typing.Callable, None] = config.DEFAULT_MAPPING,
    ):
        super().__init__()
        assert control_gate in [op.CNOT, op.CZ], f"Control gate {control_gate} not supported"
        if qubits is None:
            qubits = list(range(num_qubits_total))
        self.qubits = qubits
        self.num_qubits = num_qubits_total
        self.rots = [utils.parse_rot(r) for r in rotations]
        if remapping is None:
            remapping = qw_map.none
        self.remapping = remapping
        self.param_budget = param_budget
        self.control_gate = control_gate
        self.control_gate_spacing = control_gate_spacing
        self.layers_ub = self._allocate_params()
        self.layers = torch.nn.Sequential(
            *[layer.build(num_qubits=num_qubits_total) for layer in self.layers_ub]
        )

    def _allocate_params(self):
        def q_yielder():
            def rot_yielder():
                while True:
                    for r in self.rots:
                        yield r

            rots = rot_yielder()
            current_params = 0
            current_depth = 0
            control_iteration = -1
            while True:
                r = next(rots)
                if current_depth % self.control_gate_spacing == 0 and current_depth > 0:
                    control_iteration = (control_iteration + 1) % len(self.qubits)
                    for ci in range(len(self.qubits)):
                        it = control_iteration % (len(self.qubits) - 1)
                        ti = (ci + 1 + it) % len(self.qubits)
                        ci_q, ti_q = self.qubits[ci], self.qubits[ti]
                        yield self.control_gate(control=ci_q, target=ti_q)

                current_depth += 1
                for q in self.qubits:
                    current_params += 1
                    if current_params > self.param_budget:
                        return
                    yield r(qubit=q, remapping=self.remapping)

        return list(q_yielder())

    def __str__(self) -> str:
        return "SEL"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return [layer.to_qasm() for layer in self.layers_ub]

    def decompose(self) -> typing.List[op.UnbuiltOperator]:
        return self.layers_ub

    def forward(self, x):
        return self.layers(x)
