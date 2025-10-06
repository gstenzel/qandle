"""Monte-Carlo trajectory simulator using Pauli sampling."""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..noise.model import NoiseModel


class TrajectoryBackend:
    """Stochastic trajectory simulator skeleton."""

    def __init__(self, num_qubits: int, noise_model: Optional[NoiseModel] = None, shots: int = 1024):
        self.num_qubits = num_qubits
        self.noise_model = noise_model
        self.shots = shots
        self.state = torch.zeros(2 ** num_qubits, dtype=torch.cfloat)
        self.state[0] = 1.0

    def run(self, circuit: Any) -> torch.Tensor:
        """Run ``shots`` trajectories of ``circuit``.

        TODO: implement sampling of Pauli noise and circuit evolution.
        """
        # TODO: implement circuit execution
        return self.state
