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
