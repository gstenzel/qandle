import networkx as nx
import random


class Node:
    def __init__(self, control: int, target: int, original_index: int):
        self.c = control
        self.t = target
        self.original_index = original_index
        self.group: int = -1
        self.__hash = hash(f"{self.c}|{self.t}|{self.original_index}|{random.random()}")

    def __repr__(self) -> str:
        return f"{self.c}|{self.t}"

    def __hash__(self) -> int:
        return self.__hash


def groupnodes(G: nx.DiGraph, max_qubits):
    for sub_g in nx.weakly_connected_components(G):
        roots = [n for n in sub_g if G.in_degree(n) == 0]
        if len(roots) > 1:
            G = groupnodes_max_qubits_per_subcircuit(G, max_qubits)  # FIXME
        else:
            G = groupnodes_max_qubits_per_subcircuit(G, max_qubits)
    return G


def groupnodes_max_qubits_per_subcircuit(sub_g: nx.DiGraph, max_qubits: int) -> nx.DiGraph:
    assert max_qubits >= 2, "max_qubits must be >= 2"

    def get_start():
        """Get node with the lowest in_degree, that has group -1"""
        nodes_by_deg = sorted(sub_g.nodes(), key=lambda n: sub_g.in_degree(n))
        for n in nodes_by_deg:
            if n.group == -1:
                return n
        raise ValueError("No node with group -1 found")

    def highest_group() -> int:
        """Get the highest group number"""
        return max(n.group for n in sub_g.nodes())

    def all_parents_grouped(node: Node) -> bool:
        """Check if all parents of node are in a group"""
        return all(pre.group != -1 for pre in sub_g.predecessors(node))

    def add_to_group(start, used_qubits: set):
        used_qubits.add(start.c)
        used_qubits.add(start.t)
        for n in sub_g[start]:  # for all successors
            if n.group == -1:
                if all_parents_grouped(n):
                    if len(used_qubits.union(set([n.c, n.t]))) <= max_qubits:
                        n.group = start.group
                        add_to_group(n, used_qubits)

    while any(n.group == -1 for n in sub_g.nodes()):
        first = get_start()
        first.group = highest_group() + 1
        add_to_group(first, set((first.c, first.t)))

    return sub_g
