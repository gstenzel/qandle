import dataclasses
import typing


@dataclasses.dataclass
class QasmRepresentation:
    """
    A class representing a single gate in a quantum circuit
    """

    gate_str: str
    gate_value: typing.Union[str, float] = ""
    qubit: typing.Union[str, int] = ""
    qasm3_inputs: str = ""
    qasm3_outputs: str = ""

    def __iter__(self):
        yield self

    def to_string(self) -> str:
        s = self.gate_str
        if self.qasm3_inputs != "":
            s += f"({self.qasm3_inputs})"
        else:
            if self.gate_value != "":
                s += f"({self.gate_value})"
        if self.qubit != "" and self.qubit is not None:
            s += f" q[{self.qubit}]"
        s += ";"
        return s


def convert_to_qasm(circuit, qasm_version: int = 2, include_header: bool = True):
    """
    Convert a circuit to qasm
    """

    num_qubits = circuit.num_qubits

    if qasm_version == 2:
        qasm_header = f"""\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
creg c[{num_qubits}];
"""
    else:
        qasm_header = f"""\
OPENQASM 3.0;
qubit[{num_qubits}] q;
bit[{num_qubits}] c;
"""

    reps = circuit.to_qasm()
    outp = ""
    if include_header:
        outp += qasm_header
    if qasm_version == 3:
        for rep in reps:
            for r in rep:
                if r.qasm3_inputs != "":
                    outp += f"input float {r.qasm3_inputs};\n"
                if r.qasm3_outputs != "":
                    outp += f"output float {r.qasm3_outputs};\n"

    for rep in reps:
        for r in rep:
            outp += r.to_string() + "\n"

    return outp
