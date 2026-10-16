"""Superposition and phase-kickback test for the public linear oracle."""

from math import isclose, sqrt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from oracle_functions import linear_public
from oracle_simulator import build_reversible_oracle


def test_linear_superposition_phase_kickback() -> None:
    """Verify |+>^8|-> becomes the phase-encoded input state times |->."""
    n = 8
    circuit = QuantumCircuit(n + 1)
    circuit.h(range(n))
    circuit.x(n)
    circuit.h(n)
    circuit.append(build_reversible_oracle(linear_public, n=n, m=1), range(n + 1))

    state = Statevector.from_instruction(circuit)
    amplitude = 1 / sqrt(2 ** (n + 1))

    for x in range(2**n):
        phase = -1 if linear_public(x) else 1
        zero_output_amplitude = state.data[x]
        one_output_amplitude = state.data[x | (1 << n)]

        assert isclose(zero_output_amplitude.real, phase * amplitude, abs_tol=1e-10)
        assert isclose(zero_output_amplitude.imag, 0.0, abs_tol=1e-10)
        assert isclose(one_output_amplitude.real, -phase * amplitude, abs_tol=1e-10)
        assert isclose(one_output_amplitude.imag, 0.0, abs_tol=1e-10)

        # The opposite amplitudes are the computational-basis signature of |->.
        assert abs(zero_output_amplitude + one_output_amplitude) <= 1e-10


if __name__ == "__main__":
    test_linear_superposition_phase_kickback()
    print("Linear-oracle superposition test passed.")
