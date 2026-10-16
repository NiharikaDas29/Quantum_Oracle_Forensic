"""Local reversible-oracle helpers for the public constant and linear examples."""

from collections.abc import Callable

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Statevector

from oracle_functions import constant_public, linear_public


def append_oracle_placeholder(
    circuit: QuantumCircuit, n: int, m: int
) -> QuantumCircuit:
    """Append the judge-replaced oracle to input qubits followed by output qubits."""
    if n < 0 or m < 0 or n + m > circuit.num_qubits:
        raise ValueError("the circuit must contain at least n + m qubits")

    placeholder = Gate("oracle", n + m, [])
    circuit.append(placeholder, list(range(n + m)))
    return circuit


def build_reversible_oracle(
    function: Callable[[int], int], n: int = 8, m: int = 1
) -> Gate:
    """Build a local gate implementing |x>|y> -> |x>|y XOR f(x)>."""
    if n != 8 or m != 1:
        raise ValueError("the public examples require n=8 and m=1")

    oracle_circuit = QuantumCircuit(n + m)
    input_qubits = list(range(n))
    output_qubit = n

    for x in range(1 << n):
        value = function(x)
        if value not in (0, 1):
            raise ValueError("a one-bit oracle function must return 0 or 1")
        if value == 1:
            zero_controls = [qubit for qubit in input_qubits if not (x >> qubit) & 1]
            if zero_controls:
                oracle_circuit.x(zero_controls)
            oracle_circuit.mcx(input_qubits, output_qubit)
            if zero_controls:
                oracle_circuit.x(zero_controls)

    return oracle_circuit.to_gate(label="local_oracle")


def _basis_state(n: int, x: int, y: int) -> Statevector:
    """Create |x>|y> using qubit k as bit k."""
    return Statevector.from_int(x | (y << n), 2 ** (n + 1))


def _assert_basis_action(function: Callable[[int], int], x: int, y: int) -> None:
    """Verify the local gate preserves x and XORs f(x) into y."""
    n = 8
    oracle = build_reversible_oracle(function, n=n, m=1)
    actual = _basis_state(n, x, y).evolve(oracle)
    expected = _basis_state(n, x, y ^ function(x))
    if actual != expected:
        raise AssertionError(
            f"{function.__name__} failed for x={x}, y={y}: "
            f"expected {expected}, got {actual}"
        )


def run_focused_tests() -> None:
    """Test both public one-bit examples, including nonzero output inputs."""
    for function in (constant_public, linear_public):
        for x in (0, 1, 2, 5, 178, 255):
            for y in (0, 1):
                _assert_basis_action(function, x, y)

    circuit = QuantumCircuit(9)
    append_oracle_placeholder(circuit, n=8, m=1)
    instruction = circuit.data[0].operation
    if instruction.name != "oracle" or instruction.num_qubits != 9:
        raise AssertionError("placeholder oracle gate has the wrong shape")


if __name__ == "__main__":
    run_focused_tests()
    print("Focused reversible-oracle tests passed.")
