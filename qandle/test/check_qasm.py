def check(circuit):
    import openqasm3
    import qandle

    q2 = qandle.convert_to_qasm(circuit, qasm_version=2, include_header=True)
    q3 = qandle.convert_to_qasm(circuit, qasm_version=3, include_header=True)
    assert openqasm3.parser.parse(q2).version == "2.0"
    assert openqasm3.parser.parse(q3).version == "3.0"
