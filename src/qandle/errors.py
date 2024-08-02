class UnbuiltGateError(ValueError):
    """
    Raised when a gate is used in a circuit before it is built. Either build the gate manually using :code:`gate.build()` or use the gate in a circuit, which will build it automatically.
    """

    pass
