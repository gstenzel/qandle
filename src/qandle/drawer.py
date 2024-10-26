from . import qcircuit
from . import config
from . import operators as op
import typing
import einops
import re

__all__ = ["draw"]


def draw(circuit) -> str:
    """
    Draw a circuit diagram. This is just a simple text-based drawing, for more advanced drawing, export to OpenQASM and use more advanced tools, like PennyLane's or Qiskit's drawer. For more options, see :class:`qandle.config`.
    """

    def str_(gate):
        if config.DRAW_SHOW_VALUES:
            return str(gate)
        else:
            """remove all digits, dots, underscores, blanks and brackets"""
            return re.sub(r"[\d\._\s\(\)]", "", str(gate))

    if isinstance(circuit, qcircuit.SplittedCircuit):
        """For CuttedQCircuit, draw the subcircuits separately"""
        return _draw_text_splitted(circuit)
    num_qubits = circuit.num_qubits
    qubits = [f"q{i}" for i in range(num_qubits)]
    for gate in circuit.layers:
        _all_w = set(range(num_qubits))
        try:
            qubits[gate.qubit] += config.DRAW_DASH + str_(gate)
            _all_w.remove(gate.qubit)
            for w in _all_w:
                if not config.DRAW_SHIFT_LEFT:
                    qubits[w] += config.DRAW_DASH * (len(str_(gate)) + 1)
        except AttributeError:
            if isinstance(gate, (op.CNOT, op.BuiltCNOT, op.CZ, op.BuiltCZ)):
                _all_w.remove(gate.c)
                _all_w.remove(gate.t)

                # longest line between control and target (including control/target)
                longest = max(
                    len(qubits[li]) for li in range(min(gate.c, gate.t), max(gate.c, gate.t) + 1)
                )

                if config.DRAW_SHIFT_LEFT:
                    qubits[gate.c] += config.DRAW_DASH * (longest - len(qubits[gate.c]))
                    qubits[gate.t] += config.DRAW_DASH * (longest - len(qubits[gate.t]))

                qubits[gate.c] += config.DRAW_DASH + "⬤"
                if isinstance(gate, (op.CNOT, op.BuiltCNOT)):
                    qubits[gate.t] += config.DRAW_DASH + "⊕"
                elif isinstance(gate, (op.CZ, op.BuiltCZ)):
                    qubits[gate.t] += config.DRAW_DASH + "⬤"
                else:
                    raise ValueError("Unknown gate type")
                for w in range(min(gate.c, gate.t) + 1, max(gate.c, gate.t)):
                    if config.DRAW_SHIFT_LEFT:
                        qubits[w] += config.DRAW_DASH * (longest - len(qubits[w]))
                    if config.DRAW_CROSS_BETWEEN_CNOT:
                        qubits[w] += config.DRAW_DASH + "┼"
                    else:
                        qubits[w] += config.DRAW_DASH * 2
                    _all_w.remove(w)
                for w in _all_w:
                    if not config.DRAW_SHIFT_LEFT:
                        qubits[w] += config.DRAW_DASH * 2
            elif isinstance(gate, einops.layers.torch.Rearrange):
                pass  # ignore einops layers
            else:
                for w in _all_w:
                    qubits[w] += config.DRAW_DASH + str_(gate)

    longest = max([len(w) for w in qubits]) + 1
    for w in range(num_qubits):
        qubits[w] += config.DRAW_DASH * (longest - len(qubits[w]))
    return "\n".join(qubits)


def _draw_text_splitted(circuit) -> str:
    num_qubits = circuit.num_qubits
    placeholder = "SUB{}"
    qubits = [f"q{i}" for i in range(num_qubits)]
    sub_drawings = {}
    for subkey, sub in enumerate(circuit.subcircuits):
        stretch = list(sub.mapping.keys())  # list of qubits the subcircuit is applied to
        _all_w = set(range(num_qubits))
        for w in stretch:
            qubits[w] += "-" + placeholder.format(subkey)
            _all_w.remove(w)
        for w in _all_w:
            qubits[w] += "-" + "-" * len(placeholder.format(subkey))
        sub_drawings[subkey] = (
            placeholder.format(subkey)
            + "\n"
            + draw(qcircuit.UnsplittedCircuit(num_qubits=len(stretch), layers=sub.layers))
        )
    for w in range(num_qubits):
        qubits[w] += "-"
    qubits = "\n".join(qubits)
    formatted_sub_drawings = _format_splitted(sub_drawings, how="column")  # type: ignore
    return qubits + "\nSubcircuits:\n" + formatted_sub_drawings


def _format_splitted(sub_drawing: dict, how=typing.Literal["column", "row"]) -> str:
    """
    returns a formatted string of the subcircuits in sub_drawing
    if how == "column", the subcircuits are drawn next to each other
    if how == "row", the subcircuits are drawn below each other
    """
    if how == "row":
        return "\n".join(sub_drawing.values())
    elif how == "column":
        format_str = ""
        result = ""
        longest_block = max(  # Most lines in a sub_drawing
            [len(subv.split("\n")) for subv in sub_drawing.values()]
        )
        sub_drawing = {
            k: v + "\n" * (longest_block - len(v.split("\n"))) for k, v in sub_drawing.items()
        }
        for subkey in sub_drawing:
            longest_line = max([len(la) for la in sub_drawing[subkey].split("\n")])
            format_str += (
                "{" + str(subkey) + ":<" + str(longest_line + config.DRAW_SPLITTED_PAD) + "}"
            )
        for line in zip(*[s.split("\n") for s in sub_drawing.values()]):
            result += format_str.format(*line) + "\n"
        return result
    else:
        raise ValueError("how must be 'column' or 'row'.")
