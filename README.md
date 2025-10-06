# Disclaimer: THIS REPO IS CURRENTLY EXPECTING SOME DIFFICULTIES WHEN SPLITTING CIRCUIT. 
WE ARE LOOKING FORWARD TO IMPROVE THE SITUATION IN THE UPCOMING RELEASE. PLEASE STAY TUNED AND LEAVE A STAR :star: TO BE NOTIFIED ONCE OUR FIXES GO LIVE.

# QANDLE
**QANDLE** is a fast and simple quantum state-vector simulator for hybrid machine learning using the PyTorch library.
Documentation and examples can be found in the [QANDLE documentation](https://gstenzel.github.io/qandle/), the code resides on [GitHub](https://github.com/gstenzel/qandle).
The paper can be found on [arXiv](https://arxiv.org/abs/2404.09213).

## Installation
```bash
pip install qandle
```

## Usage
```python
import torch
import qandle

# Create a quantum circuit
circuit = qandle.Circuit(
    layers=[
        # embedding layer
        qandle.AngleEmbedding(num_qubits=2),
        # trainable layer, with random initialization
        qandle.RX(qubit=1),
        # trainable layer, with fixed initialization
        qandle.RY(qubit=0, theta=torch.tensor(0.2)),
        # data reuploading
        qandle.RX(qubit=0, name="data_reuploading"),
        # disable quantum weight remapping
        qandle.RY(qubit=1, remapping=None),
        qandle.CNOT(control=0, target=1),
        qandle.MeasureProbability(),
    ]
)


input_state = torch.rand(circuit.num_qubits, dtype=torch.float) # random input
data_reuploading = torch.rand(1, dtype=torch.float) # random data reuploading input

# Run the circuit
circuit(input_state, data_reuploading=data_reuploading)
```	

## New version 
The latest version of the package (still debugging) can be downloaded as follows:
```bash
git clone https://github.com/MKorp7/qandle.git --branch current_version_30_09
uv sync
```

This release (branch current_version_30_09) introduces numerous features and improvements:
- Multiple Simulation Backends: In addition to the default state-vector simulator, QANDLE now includes a** Matrix Product State** (MPS) backend for more scalable simulations. You can execute any circuit on the MPS backend (with a configurable bond dimension) for lower memory usage on larger circuits. QANDLE’s backend interface (QuantumBackend) makes it easy to plug in new simulation engines. (The state-vector backend remains available for exact simulation.)
- **CCNOT** (Toffoli gate)

  The Toffoli gate (known also CCNOT) is a universal reversible logic gate that was missing from the QANDLE quantum simulator,but is essential for universal quantum computation and widely used in quantum algorithms and error correction. Splitting of circuits with Toffoli gate is supported

**- Pairwise rotations
   Creating $2^n $ matrices and multipling them is a time and space consuming opertiation. 
  <img width="605" height="332" alt="obraz" src="https://github.com/user-attachments/assets/008a42d0-2b5d-47d0-a3f8-a7d311d344f6" />

  The correctness and time was tested on CPU by 1000 repetisions for each number of qubits. 

****- Parameter shift
**
   The parameter-shift rule provides a quantum-native gradient approach that is compatible with real devices, as it avoids the need to introspect or differentiate through the quantum state evolution. In other words, it enables gradient computation “externally” by querying the circuit as a function, which is ideal for hardware where the quantum process is a black box. By implementing this in Qandle, we allow users to explore training algorithms that could later be deployed on real quantum processors, bridging the gap between simulation and experiment.
  
  **Limitations**: The current implementation of parameter-shift differentiation in Qandle supports the most common parameterized gates (rotations about Pauli axes, which have two eigenvalues). These gates all obey the standard two-term shift rule. For more complex gates or multi-parameter operations, the situation is more complicated. The basic two-term rule is not directly applicable to gates with more than two distinct eigenvalues.


- **Noise** 

  Another major improvement in this release is the introduction of a noise simulation framework in QANDLE. The library now defines common quantum noise channels (such as bit-flip, phase-flip, depolarizing, etc.) following a similar pattern to how gates are defined. To actually simulate noisy circuits, the release provides new backend engines: an exact density-matrix simulator and a Monte Carlo trajectory simulator. The density matrix backend evolves the full $2^n \times 2^n$ density matrix of the system and can apply Kraus operators for noise, allowing accurate noisy simulations (at higher memory cost).

- **Utils**

  Since Qandle supports building own matrices, user can independently define gates such as Hadamard etc. However, for practical reasons, ready-made implementations have been added.

- **Benchmarking**

  <img width="1729" height="566" alt="obraz" src="https://github.com/user-attachments/assets/9e224675-6b03-4f62-8d2d-0ce0e1992393" />
<img width="1732" height="572" alt="obraz" src="https://github.com/user-attachments/assets/529f8335-e502-4654-adab-9664556d1183" />


Example:

```python
    circuit = qandle.Circuit(
        layers=[
            qandle.AngleEmbedding(num_qubits=2),
            qandle.RX(qubit=0),
            qandle.CNOT(control=0, target=1),
            qandle.BitFlipNoise(qubit=0, p=0.01),
            qandle.PhaseFlipNoise(qubit=1, p=0.02),
            qandle.DepolarizingNoise(qubit=1, p=0.05),
            qandle.MeasureProbability(),
        ],
        backend=qandle.backends.MPSBackend(max_bond_dim=32),
    )

```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use QANDLE in your research, please cite the following paper:
```bbl
@misc{qandle2024,
      title={Qandle: Accelerating State Vector Simulation Using Gate-Matrix Caching and Circuit Splitting}, 
      author={Gerhard Stenzel and Sebastian Zielinski and Michael Kölle and Philipp Altmann and Jonas Nüßlein and Thomas Gabor},
      year={2024},
      eprint={2404.09213},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
