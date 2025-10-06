"""Density-matrix simulator backend with Kraus noise support."""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..noise.model import NoiseModel


class DensityMatrixBackend:
    """Exact density matrix simulator.

    This is a skeleton implementation; rest is marked as TODO
    blocks.
    """

    def __init__(self, num_qubits: int, noise_model: Optional[NoiseModel] = None):
        self.num_qubits = num_qubits
        self.noise_model = noise_model
        self.state = torch.eye(2**num_qubits, dtype=torch.cfloat)

    def run(self, circuit: Any) -> torch.Tensor:
        """Simulate ``circuit`` and return the final density matrix.

        TODO: apply gates and noise channels correctly.
        """
        # TODO: implement circuit execution
        return self.state
