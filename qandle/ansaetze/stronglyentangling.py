import torch
import typing
import qw_map
import copy


import qandle.config as config
import qandle.operators as op
import qandle.utils as utils
import qandle.qasm as qasm
from qandle.ansaetze.ansatz import UnbuiltAnsatz, BuiltAnsatz


__all__ = ["StronglyEntanglingLayer"]


class StronglyEntanglingLayer(UnbuiltAnsatz):
    def __init__(
        self,
        num_qubits: int,
        depth: int = 1,
        rotations=["rz", "ry", "rz"],
        q_params=None,
        remapping: typing.Union[typing.Callable, None] = config.DEFAULT_MAPPING,
    ):
        super().__init__()
        self.depth = depth
        self.num_qubits = num_qubits
        self.rots = [utils.parse_rot(r) for r in rotations]
        if q_params is None:
            q_params = torch.rand(depth, num_qubits, len(rotations))
        self.q_params = q_params
        if remapping is None:
            remapping = qw_map.none
        self.remapping = remapping

    def build(self, *args, **kwargs) -> BuiltAnsatz:
        return StronglyEntanglingLayerBuilt(
            num_qubits=self.num_qubits,
            depth=self.depth,
            rotations=self.rots,
            q_params=self.q_params,
            remapping=self.remapping,
        )

    def __str__(self) -> str:
        return "SEL"

    def to_qasm(self) -> qasm.QasmRepresentation:
        raise NotImplementedError("TODO")

    def decompose(self) -> typing.List[op.UnbuiltOperator]:
        layers = []
        for d in range(self.depth):
            for w in range(self.num_qubits):
                for r in range(len(self.rots)):
                    layers.append(
                        self.rots[r](
                            qubit=w,
                            theta=self.q_params[d, w, r],
                            remapping=self.remapping,
                        )
                    )
            layers.extend(
                StronglyEntanglingLayerBuilt._get_cnots(
                    self.num_qubits, d % (self.num_qubits - 1)
                )
            )
        return layers


class StronglyEntanglingLayerBuilt(BuiltAnsatz):
    def __init__(
        self,
        num_qubits: int,
        depth: int,
        rotations: typing.List[op.UnbuiltParametrizedOperator],
        q_params: torch.Tensor,
        remapping: typing.Union[typing.Callable, None],
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.rots = rotations
        layers = []
        for d in range(depth):
            for w in range(self.num_qubits):
                for r in range(len(self.rots)):
                    layers.append(
                        self.rots[r](
                            qubit=w, theta=q_params[d, w, r], remapping=remapping # type: ignore
                        ).build(num_qubits)
                    )
            layers.extend(self._get_cnots(self.num_qubits, d % (self.num_qubits - 1)))
        self.mods = torch.nn.Sequential(*layers)

    def forward(self, state: torch.Tensor):
        return self.mods(state)

    @staticmethod
    def _get_cnots(num_qubits: int, iteration: int) -> typing.List[op.UnbuiltOperator]:
        assert iteration < num_qubits - 1
        cnots = []
        for c in range(num_qubits):
            t = (c + iteration + 1) % num_qubits
            cnots.append(op.CNOT(c, t).build(num_qubits))
        return cnots

    def __str__(self) -> str:
        return "SEL"

    def to_qasm(self) -> str:
        return [g.to_qasm() for g in self.decompose()] # type: ignore

    def decompose(self) -> typing.List[op.UnbuiltOperator]:
        layers = []
        for mod in self.mods:
            layers.append(copy.deepcopy(mod))
        return layers
