"""Seeded local evaluation harness for the QryptoNox solver."""

from collections import Counter
from dataclasses import dataclass
import argparse
import json
import random
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit

from solver import solve


@dataclass(frozen=True)
class OracleInstance:
    family: str
    n: int
    m: int
    secret: object
    function: Callable[[int], int]


def _random_function(rng: random.Random, values: list[int]) -> Callable[[int], int]:
    table = list(values)

    def function(x: int) -> int:
        return table[x]

    return function


def _make_instance(family: str, rng: random.Random) -> OracleInstance:
    if family == "constant":
        secret = rng.randrange(2)
        function = lambda x, value=secret: value
        return OracleInstance(family, 8, 1, secret, function)
    if family == "balanced":
        values = [0] * 128 + [1] * 128
        rng.shuffle(values)
        return OracleInstance(family, 8, 1, None, _random_function(rng, values))
    if family == "linear":
        a = rng.randrange(1, 256)
        b = rng.randrange(2)
        function = lambda x, a=a, b=b: ((a & x).bit_count() & 1) ^ b
        return OracleInstance(family, 8, 1, [a, b], function)
    if family == "grover":
        target = rng.randrange(256)
        function = lambda x, target=target: int(x == target)
        return OracleInstance(family, 8, 1, target, function)
    if family == "randfun":
        values = [int(rng.random() < 0.3) for _ in range(256)]
        return OracleInstance(family, 8, 1, None, _random_function(rng, values))
    if family == "simon":
        secret = rng.randrange(1, 256)
        labels: dict[int, int] = {}
        values = list(range(128))
        rng.shuffle(values)
        for x in range(256):
            representative = min(x, x ^ secret)
            if representative not in labels:
                labels[representative] = values[len(labels)]
        function = lambda x, labels=labels, secret=secret: labels[min(x, x ^ secret)]
        return OracleInstance(family, 8, 8, secret, function)
    if family == "simon2":
        first = rng.randrange(1, 256)
        second = rng.randrange(1, 256)
        while second == first:
            second = rng.randrange(1, 256)
        subspace = {0, first, second, first ^ second}
        labels: dict[int, int] = {}
        values = list(range(64))
        rng.shuffle(values)
        for x in range(256):
            representative = min(x ^ h for h in subspace)
            if representative not in labels:
                labels[representative] = values[len(labels)]
        secret = sorted(subspace - {0})
        function = lambda x, labels=labels, subspace=subspace: labels[
            min(x ^ h for h in subspace)
        ]
        return OracleInstance(family, 8, 8, secret, function)
    if family == "perm":
        values = list(range(256))
        rng.shuffle(values)
        return OracleInstance(family, 8, 8, None, _random_function(rng, values))
    raise ValueError(f"unknown family: {family}")


def _apply_oracle(
    state: np.ndarray,
    n_qubits: int,
    qargs: list[int],
    function: Callable[[int], int],
) -> np.ndarray:
    n = 8
    m = len(qargs) - n
    output = np.zeros_like(state)
    input_qubits = qargs[:n]
    output_qubits = qargs[n:]
    for basis, amplitude in enumerate(state):
        x = sum(((basis >> q) & 1) << bit for bit, q in enumerate(input_qubits))
        y = sum(((basis >> q) & 1) << bit for bit, q in enumerate(output_qubits))
        new_y = y ^ function(x)
        target = basis
        for bit, q in enumerate(output_qubits):
            if ((new_y >> bit) & 1) != ((target >> q) & 1):
                target ^= 1 << q
        output[target] += amplitude
    return output


def _counts_key(bits: list[int], registers: list[int]) -> str:
    if len(registers) == 1:
        return "".join(str(bits[index]) for index in range(len(bits) - 1, -1, -1))
    parts = []
    offset = 0
    for width in reversed(registers):
        part = bits[offset:offset + width]
        parts.append("".join(str(bit) for bit in reversed(part)))
        offset += width
    return " ".join(parts)


def make_run(
    instance: OracleInstance,
    *,
    seed: int,
    oracle_misfire_probability: float = 0.04,
    readout_flip_probability: float = 0.02,
) -> Callable[[QuantumCircuit, int], dict[str, int]]:
    """Return a judge-like runner closed over one hidden instance."""
    rng = random.Random(seed)
    usage = {"queries": 0}

    def run(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        oracle_count = sum(
            instruction.operation.name == "oracle"
            for instruction in circuit.data
        )
        usage["queries"] += oracle_count * shots
        if circuit.num_qubits > 20:
            raise ValueError("circuit exceeds the 20-qubit limit")
        measured: dict[int, int] = {}
        for instruction in circuit.data:
            if instruction.operation.name == "measure":
                q = circuit.find_bit(instruction.qubits[0]).index
                c = circuit.find_bit(instruction.clbits[0]).index
                measured[c] = q
        if not measured:
            raise ValueError("circuit has no measurements")
        counts: Counter[str] = Counter()
        for _ in range(shots):
            state = np.zeros(2 ** circuit.num_qubits, dtype=complex)
            state[0] = 1
            for instruction in circuit.data:
                operation = instruction.operation
                if operation.name == "measure" or operation.name == "barrier":
                    continue
                qargs = [circuit.find_bit(q).index for q in instruction.qubits]
                if operation.name == "oracle":
                    if rng.random() < oracle_misfire_probability:
                        continue
                    state = _apply_oracle(state, circuit.num_qubits, qargs, instance.function)
                    continue
                step = QuantumCircuit(circuit.num_qubits)
                step.append(operation, qargs)
                from qiskit.quantum_info import Statevector
                state = np.asarray(Statevector(state).evolve(step).data)
            probabilities = np.abs(state) ** 2
            basis = rng.choices(range(len(probabilities)), weights=probabilities, k=1)[0]
            bits = [0] * circuit.num_clbits
            for c, q in measured.items():
                bits[c] = (basis >> q) & 1
                if rng.random() < readout_flip_probability:
                    bits[c] ^= 1
            registers = [register.size for register in circuit.cregs]
            counts[_counts_key(bits, registers)] += 1
        return dict(counts)

    run.query_usage = usage
    return run


def evaluate(
    seed: int = 20261016,
    trials_per_family: int = 2,
    budget: int = 60,
    oracle_misfire_probability: float = 0.04,
    readout_flip_probability: float = 0.02,
) -> dict[str, object]:
    """Run a seeded benchmark and return aggregate accuracy evidence."""
    families = ["constant", "balanced", "linear", "grover", "randfun", "simon", "simon2", "perm"]
    results = []
    for family_index, family in enumerate(families):
        for trial in range(trials_per_family):
            rng = random.Random(seed + family_index * 1000 + trial)
            instance = _make_instance(family, rng)
            runner = make_run(
                instance,
                seed=seed + family_index * 1000 + trial + 100,
                oracle_misfire_probability=oracle_misfire_probability,
                readout_flip_probability=readout_flip_probability,
            )
            answer = solve(instance.n, instance.m, budget, families, runner)
            results.append(
                {
                    "family": family,
                    "reported": answer[0],
                    "secret_correct": answer[1] == instance.secret,
                    "correct": answer[0] == family and answer[1] == instance.secret,
                    "queries": runner.query_usage["queries"],
                }
            )
    confusion: dict[str, dict[str, int]] = {}
    for item in results:
        row = confusion.setdefault(item["family"], {})
        row[item["reported"]] = row.get(item["reported"], 0) + 1
    per_family = {}
    for family in families:
        group = [item for item in results if item["family"] == family]
        per_family[family] = {
            "accuracy": sum(item["correct"] for item in group) / len(group),
            "average_queries": sum(item["queries"] for item in group) / len(group),
        }
    return {
        "seed": seed,
        "trials_per_family": trials_per_family,
        "budget": budget,
        "oracle_misfire_probability": oracle_misfire_probability,
        "readout_flip_probability": readout_flip_probability,
        "accuracy": sum(item["correct"] for item in results) / len(results),
        "per_family": per_family,
        "confusion": confusion,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--misfire", type=float, default=0.04)
    parser.add_argument("--readout", type=float, default=0.02)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    report = evaluate(
        seed=args.seed,
        trials_per_family=args.trials,
        budget=args.budget,
        oracle_misfire_probability=args.misfire,
        readout_flip_probability=args.readout,
    )
    print(json.dumps(report, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
