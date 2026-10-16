"""Judge-compatible classifier for the QryptoNox oracle challenge."""

from collections import Counter
from collections.abc import Callable
from itertools import combinations
import random

from qiskit import QuantumCircuit
from qiskit.circuit import Gate


FamilyRun = Callable[[QuantumCircuit, int], dict[str, int]]


class _Budget:
    def __init__(self, budget: int) -> None:
        self.limit = budget
        self.used = 0

    def reserve(self, circuit: QuantumCircuit, shots: int) -> bool:
        if shots <= 0:
            return False

        cost = sum(
            instruction.operation.name == "oracle"
            for instruction in circuit.data
        ) * shots

        if self.used + cost > self.limit:
            return False

        self.used += cost
        return True


def _basis_probe(n: int, m: int, x: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n + m, n + m)

    for bit in range(n):
        if (x >> bit) & 1:
            circuit.x(bit)

    circuit.append(Gate("oracle", n + m, []), range(n + m))
    circuit.measure(range(n + m), range(n + m))
    return circuit


def _phase_probe(n: int, m: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n + m, n)

    circuit.x(n)
    circuit.h(n)
    circuit.h(range(n))
    circuit.append(Gate("oracle", n + m, []), range(n + m))
    circuit.h(range(n))
    circuit.measure(range(n), range(n))
    return circuit


def _grover_probe(n: int, m: int, iterations: int = 8) -> QuantumCircuit:
    circuit = QuantumCircuit(n + m, n)
    circuit.x(n)
    circuit.h(n)
    circuit.h(range(n))

    for _ in range(iterations):
        circuit.append(Gate("oracle", n + m, []), range(n + m))
        circuit.h(range(n))
        circuit.x(range(n))
        circuit.h(n - 1)
        circuit.mcx(list(range(n - 1)), n - 1)
        circuit.h(n - 1)
        circuit.x(range(n))
        circuit.h(range(n))

    circuit.measure(range(n), range(n))
    return circuit


def _simon_probe(n: int, m: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n + m, n)
    circuit.h(range(n))
    circuit.append(Gate("oracle", n + m, []), range(n + m))
    circuit.h(range(n))
    circuit.measure(range(n), range(n))
    return circuit


def _counts_to_values(
    counts: dict[str, int], width: int
) -> list[tuple[int, int]]:
    values = []
    for key, count in counts.items():
        compact = key.replace(" ", "").zfill(width)
        bits = compact[-width:]
        value = sum(
            int(bit) << index
            for index, bit in enumerate(reversed(bits))
        )
        values.append((value, count))
    return values


def _run_once(
    runner: FamilyRun,
    budget: _Budget,
    circuit: QuantumCircuit,
    shots: int,
) -> dict[str, int]:
    if not budget.reserve(circuit, shots):
        return {}
    return runner(circuit, shots)


def _best_bit_function(
    observations: list[tuple[int, int]], a: int
) -> tuple[int, int]:
    candidates = []
    for b in (0, 1):
        score = sum(
            value == (((a & x).bit_count() & 1) ^ b)
            for x, value in observations
        )
        candidates.append((score, b))
    return max(candidates)


def _verify_input(
    n: int,
    budget: _Budget,
    runner: FamilyRun,
    x: int,
    shots: int = 3,
) -> tuple[int, int]:
    """Return the number of observed output ones and total observations."""
    actual_shots = min(shots, budget.limit - budget.used)
    if actual_shots <= 0:
        return 0, 0

    counts = _run_once(
        runner, budget, _basis_probe(n, 1, x), actual_shots
    )
    parsed = _counts_to_values(counts, n + 1)
    ones = sum(
        ((value >> n) & 1) * repetitions
        for value, repetitions in parsed
    )
    total = sum(repetitions for _, repetitions in parsed)
    return ones, total


def _solve_one_bit(
    n: int,
    budget: _Budget,
    families: set[str],
    runner: FamilyRun,
) -> tuple[str, object]:
    """Classify one-bit-output families using adaptive probes."""
    probes = [0, 1, 2, 4, 8, 16, 32, 64]
    observations: list[tuple[int, int]] = []

    for x in probes:
        counts = _run_once(runner, budget, _basis_probe(n, 1, x), 1)
        parsed = _counts_to_values(counts, n + 1)
        if parsed:
            observations.append((x, (parsed[0][0] >> n) & 1))

    phase_shots = min(8, budget.limit - budget.used)
    phase_counts = _run_once(
        runner, budget, _phase_probe(n, 1), phase_shots
    )
    phase_histogram = Counter()
    for value, count in _counts_to_values(phase_counts, n):
        phase_histogram[value] += count

    phase_mode, phase_mode_count = (
        phase_histogram.most_common(1)[0] if phase_histogram else (0, 0)
    )
    total_phase = sum(phase_histogram.values())
    zero_phase_count = phase_histogram.get(0, 0)
    phase_is_concentrated = (
        total_phase >= 6 and zero_phase_count >= total_phase - 1
    )

    if (
        "linear" in families
        and phase_mode != 0
        and phase_mode_count >= max(4, int(total_phase * 0.65))
    ):
        score, b = _best_bit_function(observations, phase_mode)
        if score >= max(5, len(observations) - 2):
            return "linear", [phase_mode, b]

    zero_count = sum(value == 0 for _, value in observations)
    one_count = len(observations) - zero_count
    observed_count = len(observations)

    constant_value = None
    if observed_count >= 6:
        if zero_count >= observed_count - 1:
            constant_value = 0
        elif one_count >= observed_count - 1:
            constant_value = 1

    if (
        "constant" in families
        and constant_value == 1
        and phase_is_concentrated
    ):
        return "constant", 1

    candidate_counts = Counter()

    # A single-marked-input oracle is phase-concentrated too, but amplitude
    # amplification should make its marked input dominate the output.
    if (
        "grover" in families
        and constant_value == 0
        and phase_is_concentrated
    ):
        remaining = budget.limit - budget.used
        iterations = 8 if remaining >= 24 else (6 if remaining >= 18 else 0)
        shots = min(3, remaining // iterations) if iterations else 0

        if shots:
            counts = _run_once(
                runner, budget, _grover_probe(n, 1, iterations), shots
            )
            for value, count in _counts_to_values(counts, n):
                candidate_counts[value] += count

        if candidate_counts:
            for candidate, _ in candidate_counts.most_common(3):
                if budget.limit - budget.used < 3:
                    break
                candidate_ones, candidate_total = _verify_input(
                    n, budget, runner, candidate, shots=3
                )
                # Two of three positive observations tolerate one readout/
                # oracle misfire while retaining a stronger test than 1/3.
                if candidate_total == 3 and candidate_ones >= 2:
                    return "grover", candidate

    seen = {x for x, _ in observations}
    while budget.limit - budget.used > 0 and len(observations) < 24:
        x = random.randrange(1 << n)
        if x in seen:
            continue
        seen.add(x)
        counts = _run_once(runner, budget, _basis_probe(n, 1, x), 1)
        parsed = _counts_to_values(counts, n + 1)
        if parsed:
            observations.append((x, (parsed[0][0] >> n) & 1))

    zero_count = sum(value == 0 for _, value in observations)
    one_count = len(observations) - zero_count
    observed_count = len(observations)

    if "constant" in families and observed_count >= 12:
        if one_count <= 1:
            return "constant", 0
        if zero_count <= 1:
            return "constant", 1

    if "balanced" in families and "randfun" in families:
        rate = one_count / observed_count if observed_count else 0.0
        return ("balanced", None) if rate >= 0.40 else ("randfun", None)

    for family in ("balanced", "randfun", "constant", "grover", "linear"):
        if family in families:
            if family in {"balanced", "randfun"}:
                return family, None
            if family == "constant":
                return "constant", int(one_count > zero_count)
            if family == "grover":
                candidate = (
                    candidate_counts.most_common(1)[0][0]
                    if candidate_counts else 0
                )
                return "grover", candidate
            if family == "linear":
                return "linear", [phase_mode, 0]

    raise ValueError("families does not contain a supported one-bit family")


def _parity(value: int, mask: int) -> int:
    return (value & mask).bit_count() & 1


def _rank(vectors: list[int], n: int) -> int:
    basis = [0] * n
    rank = 0
    for value in vectors:
        current = value
        for bit in range(n - 1, -1, -1):
            if not ((current >> bit) & 1):
                continue
            if basis[bit]:
                current ^= basis[bit]
            else:
                basis[bit] = current
                rank += 1
                break
    return rank


def _solve_eight_bit(
    n: int,
    m: int,
    budget: _Budget,
    families: set[str],
    runner: FamilyRun,
) -> tuple[str, object]:
    shots = min(50, budget.limit)
    counts = _run_once(runner, budget, _simon_probe(n, m), shots)
    observations = _counts_to_values(counts, n)

    if not observations:
        if "perm" in families:
            return "perm", None
        raise ValueError("Insufficient query budget for eight-bit classification")

    scores = {
        mask: sum(
            count for value, count in observations
            if _parity(value, mask) == 0
        )
        for mask in range(1, 1 << n)
    }

    total = sum(count for _, count in observations)
    best_mask = max(scores, key=scores.get)
    best_score = scores[best_mask]
    second_score = max(
        score for mask, score in scores.items() if mask != best_mask
    )
    threshold = max(1, int(total * 0.80))
    margin = max(2, int(total * 0.08))

    if "simon2" in families:
        best_triple = None
        best_triple_score = -1
        for first, second in combinations(range(1, 1 << n), 2):
            third = first ^ second
            if third <= second or third == 0:
                continue
            if _rank([first, second], n) != 2:
                continue
            if (
                scores[first] >= threshold
                and scores[second] >= threshold
                and scores[third] >= threshold
            ):
                triple_score = (
                    scores[first] + scores[second] + scores[third]
                )
                if triple_score > best_triple_score:
                    best_triple_score = triple_score
                    best_triple = (first, second, third)
        if best_triple is not None:
            return "simon2", list(sorted(best_triple))

    if (
        "simon" in families
        and best_score >= threshold
        and best_score - second_score >= margin
    ):
        return "simon", best_mask

    if "perm" in families:
        return "perm", None
    if "simon" in families:
        return "simon", best_mask
    if "simon2" in families:
        return "simon2", [best_mask, 0, best_mask]

    raise ValueError("families does not contain a supported eight-bit family")


def solve(
    n: int,
    m: int,
    budget: int,
    families: list[str],
    run: FamilyRun,
) -> tuple[str, object]:
    """Classify one sealed oracle and return its family and required secret."""
    if n != 8 or m not in (1, 8):
        raise ValueError("the challenge requires n=8 and m in {1, 8}")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("budget must be a positive integer")

    available = set(families)
    if not available:
        raise ValueError("families must not be empty")

    budget_state = _Budget(budget)
    if m == 1:
        return _solve_one_bit(n, budget_state, available, run)
    return _solve_eight_bit(n, m, budget_state, available, run)
