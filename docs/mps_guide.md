# Matrix Product State (MPS) Simulation

Qandle now supports Matrix Product State (MPS) simulation as an alternative to full state vector simulation. MPS can be more memory-efficient for certain quantum states, particularly those with limited entanglement.

## Basic Usage

To use MPS simulation, simply set `use_mps=True` when creating a circuit:

```python
import qandle

# Create gates
rx = qandle.RX(qubit=0, theta=1.0)
ry = qandle.RY(qubit=1, theta=0.5)
cnot = qandle.CNOT(control=0, target=1)

# Create circuit with MPS simulation
circuit = qandle.Circuit([rx, ry, cnot], num_qubits=2, use_mps=True)

# Run the circuit - returns a state vector for compatibility
result = circuit()
print(f"Result: {result}")
```

## Backward Compatibility

MPS simulation is fully backward compatible. Existing code works unchanged:

```python
# This still works exactly as before (uses state vectors)
circuit = qandle.Circuit([rx, ry, cnot], num_qubits=2)
result = circuit()
```

## Performance Characteristics

### Memory Usage
- **State Vector**: O(2^n) memory for n qubits
- **MPS**: O(n × d^2) memory, where d is the bond dimension (typically much smaller than 2^n)

### Time Complexity
- **Single-qubit gates**: O(d^2) for MPS vs O(2^n) for state vectors
- **Two-qubit gates**: O(d^3) for adjacent qubits, O(2^n) for non-adjacent qubits
- **General operations**: MPS may fall back to state vector simulation when needed

### When to Use MPS
MPS is most beneficial for:
- Large numbers of qubits (n > 10)
- Circuits with limited entanglement
- States that can be efficiently represented with small bond dimensions

MPS may be less efficient for:
- Highly entangled states
- Small numbers of qubits (n < 5)
- Circuits with many non-adjacent two-qubit gates

## Supported Operations

### Fully Optimized for MPS
- Single-qubit gates: RX, RY, RZ, custom U gates
- Two-qubit gates: CNOT, CZ (for adjacent qubits)

### Fallback to State Vector
- Two-qubit gates on non-adjacent qubits
- Named/parametrized operations with dynamic inputs
- Complex composite operations

## Examples

### Basic Circuit Comparison
```python
import torch
import qandle

# Create a 3-qubit circuit
gates = [
    qandle.RX(qubit=0, theta=0.5),
    qandle.RY(qubit=1, theta=0.3),
    qandle.CNOT(control=0, target=1),
    qandle.CNOT(control=1, target=2)
]

# Compare both methods
circuit_sv = qandle.Circuit(gates, num_qubits=3, use_mps=False)
circuit_mps = qandle.Circuit(gates, num_qubits=3, use_mps=True)

result_sv = circuit_sv()
result_mps = circuit_mps()

print(f"Results match: {torch.allclose(result_sv, result_mps, atol=1e-6)}")
```

### Custom Input States
```python
# Create custom input state
input_state = torch.rand(8, dtype=torch.cfloat)  # 3-qubit state
input_state = input_state / torch.norm(input_state)

# Both methods accept custom input states
result_sv = circuit_sv(input_state)
result_mps = circuit_mps(input_state)
```

### Performance Timing
```python
import time

# Time both approaches
start = time.time()
result_sv = circuit_sv()
time_sv = time.time() - start

start = time.time()
result_mps = circuit_mps()
time_mps = time.time() - start

print(f"State vector time: {time_sv:.4f}s")
print(f"MPS time: {time_mps:.4f}s")
```

## Implementation Details

The MPS implementation:
- Uses left-canonical form for efficient operations
- Automatically handles bond dimension growth during operations
- Provides seamless conversion between MPS and state vector representations
- Maintains numerical precision comparable to state vector simulation

## Future Enhancements

Planned improvements include:
- Bond dimension compression and truncation options
- Optimized implementations for specific gate sequences
- MPS-native measurement operations
- Enhanced support for non-adjacent two-qubit gates