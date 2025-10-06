from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .channels import BitFlip


@dataclass
class NoiseModel:
    """Simple noise model mapping gate names to channels."""

    channels: Dict[str, BitFlip] | None = None
    global_noise: List[BitFlip] = field(default_factory=list)

    def apply(self, circuit: "Circuit") -> "Circuit":  # noqa: F401 - forward ref
        """Return a new circuit with noise channels inserted.

        TODO: implement the actual insertion logic.
        """
        # TODO: insert noise into circuit
        return circuit
