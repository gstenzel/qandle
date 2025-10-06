import typing
import networkx as nx
import numpy as np
import dataclasses
import torch
import qw_map
import qandle
import qandle.splitter.grouping as grouping
import qandle.operators as op

cnot_types = typing.Union[op.CNOT, op.BuiltCNOT]
ccnot_types = typing.Union[op.CCNOT, op.BuiltCCNOT]

def split(
    circuit,
    max_qubits: int = 5,
) -> typing.Dict[int, "SubcircuitContainer"]:
    assert circuit.num_qubits > 1, "Can't split circuits with only 1 qubit"
    assert all(
        isinstance(layer, op.Operator) for layer in circuit.layers
    ), f"Unknown layer type in circuit: {[type(layer) for layer in circuit.layers if not isinstance(layer, op.Operator)]}"
    has_ccnot = any(isinstance(g, ccnot_types) for g in circuit.layers)
    decomposed_layers = []
    for gate in circuit.layers:
        if isinstance(gate, ccnot_types):
            decomposed_layers.extend(_decompose_ccnot(gate.c1, gate.c2, gate.t))
        else:
            decomposed_layers.append(gate)

    if has_ccnot:
        # fallback: keep ordering strictly and create one subcircuit per gate
        def remap_gate(g, mapping):
            if isinstance(g, cnot_types):
                return op.CNOT(control=mapping[g.c], target=mapping[g.t])
            elif isinstance(g, op.U):
                return op.U(qubit=mapping[g.qubit], matrix=g.matrix)
            elif isinstance(g, op.CustomGate):
                return op.CustomGate(
                    qubit=mapping[g.qubit],
                    matrix=g.original_matrix,
                    num_qubits=len(mapping),
                    self_description=g.description,
                )
            else:
                c = (
                    op.BUILT_CLASS_RELATION.T[type(g)]
                    if isinstance(g, op.BuiltOperator)
                    else type(g)
                )
                other = {}
                if hasattr(g, "theta") and g.theta is not None:
                    other["theta"] = g.theta
                if hasattr(g, "name") and g.name is not None:
                    other["name"] = g.name
                if hasattr(g, "qubit"):
                    return c(qubit=mapping[g.qubit], **other)
                elif hasattr(g, "qubits"):
                    other["qubits"] = [mapping[q] for q in g.qubits]
                    return c(**other)
                elif hasattr(g, "a") and hasattr(g, "b"):
                    return c(a=mapping[g.a], b=mapping[g.b], **other)
                elif hasattr(g, "c") and hasattr(g, "t"):
                    return c(control=mapping[g.c], target=mapping[g.t], **other)
                elif hasattr(g, "c1") and hasattr(g, "c2") and hasattr(g, "t"):
                    return c(
                        control1=mapping[g.c1],
                        control2=mapping[g.c2],
                        target=mapping[g.t],
                        **other,
                    )
                else:
                    raise ValueError(f"Unsupported gate type {type(g)}")

        subcircuits = {}
        idx = 0
        for g in decomposed_layers:
            qubits = set()
            if hasattr(g, "c"):
                qubits.update([g.c, g.t])
            if hasattr(g, "qubit"):
                qubits.add(g.qubit)
            if hasattr(g, "c1"):
                qubits.update([g.c1, g.c2, g.t])
            mapping = {q: i for i, q in enumerate(sorted(qubits))}
            subcircuits[idx] = SubcircuitContainer([remap_gate(g, mapping)], mapping)
            idx += 1
        return subcircuits

    temp_circuit = qandle.Circuit(num_qubits=circuit.num_qubits, layers=decomposed_layers)
    G = _construct_graph(temp_circuit)
    G = grouping.groupnodes(G, max_qubits)
    assigned = _assign_to_subcircuits(G.nodes(), decomposed_layers)
    normalized = _normalize_subcircuits(assigned)
    return normalized

def _decompose_ccnot(control1, control2, target):
    """
    decomposition idea from: Shende, V.V., & Markov, I.L. (2008). On the CNOT-cost of TOFFOLI gates. Quantum Inf. Comput., 9, 461-486.
    """
    no_mapping = qw_map.none
    hmat = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex64)
    tmat = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex64)
    tdgmat = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex64)

    gates = [
        # Initial Hadamard on target
        op.U(target, torch.tensor(hmat)),

        op.CNOT(control2, target),
        op.U(target, torch.tensor(tdgmat)),  # T†
        op.CNOT(control1, target),
        op.U(target, torch.tensor(tmat)),   # T
        op.CNOT(control2, target),
        op.U(control2, torch.tensor(tmat)),  # T on control2
        op.U(target, torch.tensor(tdgmat)),   # T†
        op.CNOT(control1, target),

        op.CNOT(control1, control2),
        op.U(control1, torch.tensor(tmat)),  # T on control1
        op.U(control2, torch.tensor(tdgmat)), # T† on control2
        op.CNOT(control1, control2),

        op.U(target, torch.tensor(tmat)),    # T on target
        op.U(target, torch.tensor(hmat)),
    ]
    return gates



def _construct_graph(circuit) -> nx.DiGraph:
    nodes = []
    circuit = circuit.decompose().circuit
    for i, layer in enumerate(circuit.layers):
        if isinstance(layer, cnot_types):
            nodes.append(grouping.Node(layer.c, layer.t, i))
    _txt = (
        "No CNOTs in the circuit, instead of splitting, just use separate circuits for each qubit"
    )
    assert len(nodes) > 0, _txt
    used_qubits = set(n.c for n in nodes) | set(n.t for n in nodes)
    _txt = f"Circuit has qubits not used in CNOTs. Instead of splitting, just use separate circuits for separate qubits. Only used qubits are: {used_qubits}"
    assert len(used_qubits) == circuit.num_qubits, _txt
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n)  # All CNOTs are nodes

    nodes_per_qubit = {w: [] for w in range(circuit.num_qubits)}
    for n in nodes:
        nodes_per_qubit[n.c].append(n)
        nodes_per_qubit[n.t].append(n)

    for w in range(circuit.num_qubits):
        for i in range(len(nodes_per_qubit[w]) - 1):
            # CNOTs are dependent on the previous CNOT they share a qubit with
            G.add_edge(nodes_per_qubit[w][i], nodes_per_qubit[w][i + 1])
    return G


def _assign_to_subcircuits(
    nodes: typing.List[grouping.Node], original_circuit_layers: list
) -> typing.Dict[int, typing.List[op.Operator]]:
    """
    From the list of nodes (which have their new group assigned), sort the original layers into subcircuits
    """
    # create new subcircuits, one for each group
    subcircuits = {i: dict() for i in sorted({n.group for n in nodes})}
    node_per_index = {n.original_index: n for n in nodes}
    for i, layer in enumerate(original_circuit_layers):
        # insert CNOTS into the subcircuits
        if isinstance(layer, cnot_types):
            n = node_per_index[i]
            subcircuits[n.group][i] = layer

    def get_close_cnot(original_index) -> grouping.Node:
        w = original_circuit_layers[original_index].qubit
        previous_cnode_index, next_cnode_index = None, None
        for i, layer in enumerate(original_circuit_layers):
            if isinstance(layer, cnot_types) and (layer.c == w or layer.t == w):
                if i < original_index:
                    previous_cnode_index = i
                elif i > original_index:
                    next_cnode_index = i
                    break
        nearest_cnot_index = (
            previous_cnode_index if previous_cnode_index is not None else next_cnode_index
        )
        assert nearest_cnot_index is not None, f"Error. \n\
            original circuit {original_circuit_layers}\n\
            index {original_index},\
            previous_cnode_index {previous_cnode_index},\
            next_cnode_index {next_cnode_index},"
        return node_per_index[nearest_cnot_index]

    for i, layer in enumerate(original_circuit_layers):
        if not isinstance(layer, cnot_types): #and hasattr(layer, 'qubit'):
            # insert non-CNOTs into the subcircuits
            close_n = get_close_cnot(i)
            subcircuits[close_n.group][i] = layer

    # flatten the subcircuits, removing empty dict entries
    flat_subc = dict()
    ordered_groups = sorted(
        subcircuits.keys(),
        key=lambda g: min(subcircuits[g].keys()) if len(subcircuits[g]) > 0 else float("inf"),
    )
    for subc in ordered_groups:
        flat_subc[subc] = []
        for i in sorted(subcircuits[subc].keys()):
            flat_subc[subc].append(subcircuits[subc][i])
    return flat_subc


@dataclasses.dataclass
class SubcircuitContainer:
    layers: list  # list of gates
    mapping: dict  # original index -> normalised index


def _normalize_subcircuits(assigned: dict):
    """
    subcircuits may have gaps in the qubit indices, remove those e.g.
    [CNOT(0,1), CNOT(4,5)] becomes [CNOT(0,1), CNOT(2,3)].
    Also returns a mapping from the original index to the normalised index
    (so {0:0, 1:1, 4:2, 5:3} in the example above)
    """

    def normalize_subcircuit(subcircuit: list) -> SubcircuitContainer:
        all_qubits = (
            set(g.c for g in subcircuit if hasattr(g, "c"))
            | set(g.t for g in subcircuit if hasattr(g, "t"))
            | set(g.c1 for g in subcircuit if hasattr(g, "c1"))
            | set(g.c2 for g in subcircuit if hasattr(g, "c2"))
            | set(g.a for g in subcircuit if hasattr(g, "a"))
            | set(g.b for g in subcircuit if hasattr(g, "b"))
            | set(g.qubit for g in subcircuit if hasattr(g, "qubit"))
            | set(q for g in subcircuit if hasattr(g, "qubits") for q in g.qubits)
        )
        all_qubits = sorted(list(all_qubits))
        mapping = {q: i for i, q in enumerate(all_qubits)}
        new_layers = []
        for g in subcircuit:
            if isinstance(g, cnot_types):
                new_layers.append(
                    op.CNOT(control=mapping[g.c], target=mapping[g.t])
                )
            elif isinstance(g, op.CustomGate):
                gnew = op.CustomGate(
                    qubit=mapping[g.qubit],
                    matrix=g.original_matrix,
                    num_qubits=len(all_qubits),
                    self_description=g.description,
                )
                new_layers.append(gnew)
            elif isinstance(g, op.U):
                new_layers.append(
                    op.U(
                        qubit=mapping[g.qubit],
                        matrix=g.matrix,
                    )
                )
            else:
                c = (
                    op.BUILT_CLASS_RELATION.T[type(g)]
                    if isinstance(g, op.BuiltOperator)
                    else type(g)
                )
                otherargs = {}
                if hasattr(g, "theta") and g.theta is not None:
                    otherargs["theta"] = g.theta
                if hasattr(g, "name") and g.name is not None:
                    otherargs["name"] = g.name
                if hasattr(g, "remapping"):
                    otherargs["remapping"] = g.remapping

                if hasattr(g, "qubit"):
                    new_layers.append(c(qubit=mapping[g.qubit], **otherargs))
                elif hasattr(g, "qubits"):
                    otherargs["qubits"] = [mapping[q] for q in g.qubits]
                    new_layers.append(c(**otherargs))
                elif hasattr(g, "a") and hasattr(g, "b"):
                    new_layers.append(
                        c(a=mapping[g.a], b=mapping[g.b], **otherargs)
                    )
                elif hasattr(g, "c") and hasattr(g, "t") and not isinstance(g, cnot_types):
                    new_layers.append(
                        c(control=mapping[g.c], target=mapping[g.t], **otherargs)
                    )
                elif (
                    hasattr(g, "c1")
                    and hasattr(g, "c2")
                    and hasattr(g, "t")
                ):
                    new_layers.append(
                        c(
                            control1=mapping[g.c1],
                            control2=mapping[g.c2],
                            target=mapping[g.t],
                            **otherargs,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported gate type {type(g)} in splitter normalization"
                    )
        return SubcircuitContainer(new_layers, mapping)

    normalized = {i: normalize_subcircuit(assigned[i]) for i in assigned.keys()}
    return normalized
