"""Local coherent runner for the public constant and linear oracle examples."""

from collections import Counter
from collections.abc import Callable
import random

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Statevector

from oracle_functions import constant_public, linear_public
from oracle_simulator import build_reversible_oracle


DEFAULT_ORACLE_MISFIRE_PROBABILITY = 0.04
DEFAULT_READOUT_FLIP_PROBABILITY = 0.02
MAX_QUBITS = 20

ORACLE_FAMILIES: dict[str, Callable[[int], int]] = {
    "constant": constant_public,
    "linear": linear_public,
}


def mark_local_oracle(circuit: QuantumCircuit, family: str) -> QuantumCircuit:
    """Annotate a local test circuit with the public family to simulate."""
    if family not in ORACLE_FAMILIES:
        raise ValueError(f"unsupported local oracle family: {family!r}")
    metadata = dict(circuit.metadata or {})
    metadata["oracle_family"] = family
    circuit.metadata = metadata
    return circuit


def query_count(circuit: QuantumCircuit, shots: int) -> int:
    """Return the challenge query cost for a circuit and shot count."""
    if not isinstance(shots, int) or isinstance(shots, bool) or shots < 0:
        raise ValueError("shots must be a nonnegative integer")
    return sum(
        1
        for instruction in circuit.data
        if instruction.operation.name == "oracle"
    ) * shots


def ensure_within_budget(circuit: QuantumCircuit, shots: int, budget: int) -> None:
    """Check one circuit execution against its supplied query budget."""
    cost = query_count(circuit, shots)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if cost > budget:
        raise ValueError(f"query budget exceeded: {cost} queries > {budget}")


class QueryBudgetTracker:
    """Track cumulative query usage across multiple circuit executions."""

    def __init__(self, budget: int) -> None:
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        self.budget = budget
        self.used_queries = 0

    def reserve(self, circuit: QuantumCircuit, shots: int) -> int:
        """Reserve one execution and return its query cost."""
        cost = query_count(circuit, shots)
        if self.used_queries + cost > self.budget:
            raise ValueError(
                f"cumulative query budget exceeded: "
                f"{self.used_queries + cost} queries > {self.budget}"
            )
        self.used_queries += cost
        return cost

    @property
    def remaining_queries(self) -> int:
        """Return the unused portion of the total budget."""
        return self.budget - self.used_queries


def _validate_probability(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _measurement_map(circuit: QuantumCircuit) -> dict[int, int]:
    """Return classical-bit index to measured quantum-bit index."""
    if len(circuit.cregs) > 1:
        raise ValueError(
            "multiple classical registers are not supported by local_runner; "
            "use one classical register to preserve Qiskit count-key ordering"
        )
    measurements: dict[int, int] = {}
    for instruction in circuit.data:
        if instruction.operation.name != "measure":
            continue
        qubit = circuit.find_bit(instruction.qubits[0]).index
        clbit = circuit.find_bit(instruction.clbits[0]).index
        if clbit in measurements:
            raise ValueError("a classical bit is measured more than once")
        measurements[clbit] = qubit
    if not measurements:
        raise ValueError("the circuit must contain measurements")
    return measurements


def _sample_basis_state(
    state: Statevector, rng: random.Random
) -> int:
    probabilities = np.asarray(state.probabilities(), dtype=float)
    return rng.choices(range(len(probabilities)), weights=probabilities, k=1)[0]


def _run_one_shot(
    circuit: QuantumCircuit,
    local_oracle: Gate,
    misfire_probability: float,
    measurements: dict[int, int],
    readout_flip_probability: float,
    rng: random.Random,
) -> str:
    """Execute one shot, replacing each oracle coherently or skipping it."""
    executable = QuantumCircuit(circuit.num_qubits)
    n = 8
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name == "measure":
            continue
        if operation.name == "oracle":
            if operation.num_qubits != n + 1:
                raise ValueError("local public oracles require an 8+1 qubit oracle")
            if rng.random() < misfire_probability:
                continue
            operation = local_oracle
        elif operation.name in {"barrier"}:
            continue
        elif instruction.clbits:
            raise ValueError(
                f"unsupported operation with classical bits: {operation.name}"
            )

        qubits = [
            circuit.find_bit(qubit).index for qubit in instruction.qubits
        ]
        executable.append(operation, qubits)

    state = Statevector.from_instruction(executable)
    basis_state = _sample_basis_state(state, rng)
    classical_bits = [0] * circuit.num_clbits
    for clbit, qubit in measurements.items():
        classical_bits[clbit] = (basis_state >> qubit) & 1
        if rng.random() < readout_flip_probability:
            classical_bits[clbit] ^= 1

    return "".join(str(classical_bits[index]) for index in range(circuit.num_clbits - 1, -1, -1))


def run(
    circuit: QuantumCircuit,
    shots: int,
    *,
    oracle_misfire_probability: float = DEFAULT_ORACLE_MISFIRE_PROBABILITY,
    readout_flip_probability: float = DEFAULT_READOUT_FLIP_PROBABILITY,
    rng: random.Random | None = None,
    budget: int | None = None,
) -> dict[str, int]:
    """Run a locally annotated public-oracle circuit and return Qiskit-style counts."""
    if circuit.num_qubits > MAX_QUBITS:
        raise ValueError(f"circuits may contain at most {MAX_QUBITS} qubits")
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    _validate_probability(
        oracle_misfire_probability, "oracle_misfire_probability"
    )
    _validate_probability(readout_flip_probability, "readout_flip_probability")
    if budget is not None:
        ensure_within_budget(circuit, shots, budget)

    family = (circuit.metadata or {}).get("oracle_family")
    if family not in ORACLE_FAMILIES:
        raise ValueError(
            "local circuits must set metadata['oracle_family'] to "
            "'constant' or 'linear'"
        )
    measurements = _measurement_map(circuit)
    generator = rng or random.Random()
    local_oracle = build_reversible_oracle(ORACLE_FAMILIES[family], n=8, m=1)
    counts: Counter[str] = Counter()
    for _ in range(shots):
        key = _run_one_shot(
            circuit,
            local_oracle,
            oracle_misfire_probability,
            measurements,
            readout_flip_probability,
            generator,
        )
        counts[key] += 1
    return dict(counts)


def _measured_oracle_circuit(
    family: str, input_value: int = 0, output_value: int = 0, oracle_count: int = 1
) -> QuantumCircuit:
    """Build a fully measured 8-input, 1-output local test circuit."""
    circuit = QuantumCircuit(9, 9)
    for qubit in range(8):
        if (input_value >> qubit) & 1:
            circuit.x(qubit)
    if output_value:
        circuit.x(8)
    for _ in range(oracle_count):
        circuit.append(
            Gate("oracle", 9, []),
            range(9),
        )
    circuit.measure(range(9), range(9))
    return mark_local_oracle(circuit, family)


def run_focused_tests() -> None:
    """Exercise execution, noise, ordering, register validation, and accounting."""
    normal = _measured_oracle_circuit("linear", input_value=0)
    assert run(normal, 5, oracle_misfire_probability=0, readout_flip_probability=0) == {
        "100000000": 5
    }

    misfire = _measured_oracle_circuit("constant")
    assert run(misfire, 4, oracle_misfire_probability=1, readout_flip_probability=0) == {
        "000000000": 4
    }

    readout = _measured_oracle_circuit("constant")
    assert run(readout, 4, oracle_misfire_probability=0, readout_flip_probability=1) == {
        "011111111": 4
    }

    ordering = _measured_oracle_circuit("constant", input_value=1)
    assert run(ordering, 1, oracle_misfire_probability=0, readout_flip_probability=0) == {
        "100000001": 1
    }

    multiple_registers = QuantumCircuit(9)
    multiple_registers.add_register(ClassicalRegister(8, "input_bits"))
    multiple_registers.add_register(ClassicalRegister(1, "output_bit"))
    multiple_registers.append(Gate("oracle", 9, []), range(9))
    multiple_registers.measure(range(9), list(multiple_registers.clbits))
    mark_local_oracle(multiple_registers, "constant")
    try:
        run(
            multiple_registers,
            1,
            oracle_misfire_probability=0,
            readout_flip_probability=0,
        )
    except ValueError as error:
        assert "multiple classical registers" in str(error)
    else:
        raise AssertionError("multiple classical registers were not rejected")

    two_oracles = _measured_oracle_circuit("linear", oracle_count=2)
    assert query_count(two_oracles, 7) == 14
    ensure_within_budget(two_oracles, 7, 14)
    try:
        ensure_within_budget(two_oracles, 8, 14)
    except ValueError:
        pass
    else:
        raise AssertionError("budget overrun was not rejected")

    tracker = QueryBudgetTracker(20)
    assert tracker.reserve(two_oracles, 7) == 14
    assert tracker.used_queries == 14
    assert tracker.remaining_queries == 6
    try:
        tracker.reserve(two_oracles, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("cumulative budget overrun was not rejected")


if __name__ == "__main__":
    run_focused_tests()
    print("Local runner focused tests passed.")
