import qw_map

DEFAULT_MAPPING = qw_map.tanh
r"""
Default weight remapping function for quantum gates. This function limits the weights of the variational quantum circuit's gates. 
:code:`qw_map.tanh` is the default value, yielding better results in practice. For higher performance or naive usage, set :code:`DEFAULT_MAPPING = lambda x: x` or :code:`DEFAULT_MAPPING = qw_map.none`. For more information, see the `arxiv paper 1 <https://arxiv.org/abs/2212.14807>`_ and `arxiv paper 2 <https://arxiv.org/abs/2306.05776>`_.
"""

DRAW_SPLITTED_PAD = 7
"""Number of spaces between columns in the drawer for splitted circuits."""

DRAW_DASH = "─"
r"""
Character used for drawing qubits, can be changed to the normal dash '-', depending on font, where :code:`───` often looks better than :code:`---`

.. code-block:: python

    >>> qandle.config.DRAW_DASH = "─"
    >>> print(circuit.draw())
    q0─RZ─RY─RZ─■─────X─RZ─RY─RZ─■───X───
    q1─RZ─RY─RZ─X─■───┼─RZ─RY─RZ─┼─■─┼─X─
    q2─RZ─RY─RZ───X─■─┼─RZ─RY─RZ─X─┼─■─┼─
    q3─RZ─RY─RZ─────X─■─RZ─RY─RZ───X───■─
    
    >>> qandle.config.DRAW_DASH = "-"
    >>> print(circuit.draw())
    q0-RZ-RY-RZ-■-----X-RZ-RY-RZ-■---X---
    q1-RZ-RY-RZ-X-■---┼-RZ-RY-RZ-┼-■-┼-X-
    q2-RZ-RY-RZ---X-■-┼-RZ-RY-RZ-X-┼-■-┼-
    q3-RZ-RY-RZ-----X-■-RZ-RY-RZ---X---■-

    
"""

DRAW_SHOW_VALUES = True
"""Whether to show the values of the gates in the drawer. If not, only the gate positions are shown, reducing clutter."""

DRAW_SHIFT_LEFT = False
"""
Whether to shift the gates to the left if possible for a more compact drawing.

.. code-block:: python

    >>> qandle.config.DRAW_SHIFT_LEFT = False
    >>> print(circuit.draw())
    q0─RZ─RY─RZ───────────────────■───X─RZ─RY─RZ───────────────────■─X───
    q1──────────RZ─RY─RZ──────────X─■─┼──────────RZ─RY─RZ──────────┼─■─X─
    q2───────────────────RZ─RY─RZ───X─■───────────────────RZ─RY─RZ─X───■─

    >>> qandle.config.DRAW_SHIFT_LEFT = True
    >>> print(circuit.draw())
    q0─RZ─RY─RZ─■───X─RZ─RY─RZ─■─X───
    q1─RZ─RY─RZ─X─■─┼─RZ─RY─RZ─┼─■─X─
    q2─RZ─RY─RZ───X─■─RZ─RY─RZ─X───■─


"""

DRAW_CROSS_BETWEEN_CNOT = True
"""
Whether to draw a cross :code:`┼` on the qubits between the CNOT control and target

.. code-block:: python

    >>> qandle.config.DRAW_CROSS_BETWEEN_CNOT = True
    >>> print(circuit.draw())
    q0─RZ─RY─RZ─■─────X─RZ─RY─RZ─■───X───
    q1─RZ─RY─RZ─X─■───┼─RZ─RY─RZ─┼─■─┼─X─
    q2─RZ─RY─RZ───X─■─┼─RZ─RY─RZ─X─┼─■─┼─
    q3─RZ─RY─RZ─────X─■─RZ─RY─RZ───X───■─

    >>> qandle.config.DRAW_CROSS_BETWEEN_CNOT = False
    >>> print(circuit.draw())
    q0─RZ─RY─RZ─■─────X─RZ─RY─RZ─■───X───
    q1─RZ─RY─RZ─X─■─────RZ─RY─RZ───■───X─
    q2─RZ─RY─RZ───X─■───RZ─RY─RZ─X───■───
    q3─RZ─RY─RZ─────X─■─RZ─RY─RZ───X───■─


"""
