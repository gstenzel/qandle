import typing
import networkx as nx

import dataclasses

import qandle.splitter.grouping as grouping
import qandle.operators as op

cnot_types = typing.Union[op.CNOT, op.BuiltCNOT]


def split(
    circuit,
    max_qubits: int = 5,
) -> typing.Dict[int, "SubcircuitContainer"]:
    assert circuit.num_qubits > 1, "Can't split circuits with only 1 qubit"
    assert all(
        isinstance(layer, op.Operator) for layer in circuit.layers
    ), f"Unknown layer type in circuit: {[type(layer) for layer in circuit.layers if not isinstance(layer, op.Operator)]}"

    G = _construct_graph(circuit)
    G = grouping.groupnodes(G, max_qubits)
    assigned = _assign_to_subcircuits(G.nodes(), circuit.layers)
    normalized = _normalize_subcircuits(assigned)
    return normalized


def _construct_graph(circuit) -> nx.DiGraph:
    nodes = []
    circuit = circuit.decompose()
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
    subcircuits = {i: dict() for i in set(n.group for n in nodes)}
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
        if not isinstance(layer, cnot_types):
            # insert non-CNOTs into the subcircuits
            close_n = get_close_cnot(i)
            subcircuits[close_n.group][i] = layer

    # flatten the subcircuits, removing empty dict entries
    flat_subc = dict()
    for subc in subcircuits.keys():
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
            set(g.c for g in subcircuit if isinstance(g, cnot_types))
            | set(g.t for g in subcircuit if isinstance(g, cnot_types))
            | set(g.qubit for g in subcircuit if not isinstance(g, cnot_types))
        )
        all_qubits = sorted(list(all_qubits))
        mapping = {q: i for i, q in enumerate(all_qubits)}
        new_layers = []
        for g in subcircuit:
            if isinstance(g, cnot_types):
                new_layers.append(op.CNOT(control=mapping[g.c], target=mapping[g.t]))
            elif isinstance(g, op.CustomGate):
                gnew = op.CustomGate(
                    qubit=mapping[g.qubit],
                    matrix=g.original_matrix,
                    num_qubits=len(all_qubits),
                    self_description=g.description,
                )
                new_layers.append(gnew)
            else:
                c = (
                    op.BUILT_CLASS_RELATION.T[type(g)]
                    if isinstance(g, op.BuiltOperator)
                    else type(g)
                )
                otherargs = dict()
                if hasattr(g, "theta") and g.theta is not None:
                    otherargs["theta"] = g.theta
                if hasattr(g, "name") and g.name is not None:
                    otherargs["name"] = g.name
                new_layers.append(c(qubit=mapping[g.qubit], **otherargs))
        return SubcircuitContainer(new_layers, mapping)

    normalized = {i: normalize_subcircuit(assigned[i]) for i in assigned.keys()}
    return normalized
