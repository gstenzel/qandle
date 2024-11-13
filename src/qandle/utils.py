import typing
import einops.layers.torch as einl
import torch


def reduce_dot(*args):
    """
    Compute the dot product of a series of matrices.
    """
    # matrix = args[0]
    # for sub_matrix in args[1:]:
    #     matrix = torch.matmul(sub_matrix, matrix)
    # return matrix
    return torch.linalg.multi_dot(args).T


def parse_rot(rot: str):
    from . import operators as op

    rot = rot.lower().replace("r", "")
    if rot == "x":
        return op.RX
    elif rot == "y":
        return op.RY
    elif rot == "z":
        return op.RZ
    else:
        raise ValueError(f"Unknown rotation {rot}")


def do_not_implement(*protected, reason="") -> type:
    class LimitedClass(type):
        def __new__(cls, name, bases, attrs):
            for attribute in attrs:
                if attribute in protected:
                    if reason:
                        raise AttributeError(reason)
                    else:
                        raise AttributeError(f'Overiding of attribute "{attribute}" not allowed')
            return super().__new__(cls, name, bases, attrs)

    return LimitedClass


def get_matrix_transforms(
    num_qubits: int, in_circuit: typing.List[int]
) -> typing.Tuple[einl.Rearrange, einl.Rearrange]:
    """
    Get einops layers for transforming between state and matrix representation of a subcircuit.
    """
    qubits_in_subc = [f"a{i}" for i in in_circuit]
    qubits_not_in_subc = [f"a{i}" for i in range(num_qubits) if i not in in_circuit]
    all_qubits = [f"a{i}" for i in range(num_qubits)]
    state_pattern = f"batch ({' '.join(all_qubits)})"
    matrix_pattern = f"(batch {' '.join(qubits_not_in_subc)}) ({' '.join(qubits_in_subc)})"
    to_matrix_pattern = f"{state_pattern} -> {matrix_pattern}"
    to_state_pattern = f"{matrix_pattern} -> {state_pattern}"
    args = {f"a{i}": 2 for i in range(num_qubits)}
    to_matrix = einl.Rearrange(to_matrix_pattern, **args)
    to_state = einl.Rearrange(to_state_pattern, **args)
    return to_matrix, to_state


def __analyze_barren_plateu(
    circuit, loss_f=lambda x: x[0], num_points=30, other_params=0.1
):  # pragma: no cover
    """
    Analyze, whether a circuit has a barren plateau for a given loss function,
    by checking the gradient of the loss function for num_points parameters (while keeping the rest fixed).
    """
    # sel = qandle.StronglyEntanglingLayer(num_qubits=3, depth=8, rotations=["rz", "ry", "rz"], remapping=None)
    # qc =  qandle.QCircuit(3, layers=[*sel.decompose(), ])
    # grads, losses, params, names = analyze_barren_plateu(qc)
    # fig, ax = plt.subplots(ncols=2, sharey=True, sharex=True, figsize=(15, 7))
    # ax[0].imshow(grads,)
    # ax[1].imshow(losses,)
    # ax[0].set_xticks(np.arange(len(params))[::3], labels=[f"{p.item():.2f}" for p in params[::3]])
    # ax[0].set_yticks(np.arange(len(names)), labels=names)
    # fig.tight_layout()

    import torch

    possible_params = torch.linspace(-torch.pi, torch.pi, num_points, requires_grad=True)
    zerosd = {
        k: torch.ones_like(v, requires_grad=True) + other_params
        for k, v in circuit.state_dict().items()
    }
    grads = torch.zeros(len(zerosd), num_points)
    losses = torch.zeros(len(zerosd), num_points)
    for ipname in range(len(zerosd)):
        pname = list(zerosd.keys())[ipname]
        for thetai, theta in enumerate(possible_params):
            sd = zerosd.copy()
            sd[pname] = theta
            circuit.load_state_dict(sd)
            for p in circuit.parameters():
                p.requires_grad = True
            circuit.zero_grad()
            loss = loss_f(circuit())
            loss.backward()
            grad = list(circuit.parameters())[ipname].grad
            grads[ipname, thetai] = grad.abs().mean()
            losses[ipname, thetai] = loss.detach()
    return grads, losses, possible_params.detach(), list(circuit.state_dict().keys())
