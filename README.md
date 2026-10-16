# QryptoNox Quantum Oracle Forensics

**Identifying the Hidden Structure of Sealed Quantum Black Boxes**

A Qiskit 2.x solution for the *Quantum Oracle Forensics* challenge at IBM Qiskit Fall Fest 2026, organized by QC@IITI, IIT Indore, with QryptoNox.

The objective is to classify an unknown quantum oracle into one of eight families and recover its secret when required, using a limited and noisy query interface.

## 1. Project Overview

The solver implements an adaptive quantum-query strategy for the eight oracle families:

- `constant`
- `balanced`
- `linear`
- `grover`
- `randfun`
- `simon`
- `simon2`
- `perm`

The challenge uses \(n=8\) input bits, with \(m=1\) or \(m=8\) output bits. Each oracle has a maximum query budget of 60. Under the standard noise setting, oracle applications have a 4% misfire probability and measured bits have a 2% flip probability.

The solver combines classical basis-state probes with quantum circuits for phase kickback, Bernstein–Vazirani-style linear-mask recovery, amplitude amplification, and Simon-type hidden-structure detection.

## 2. Solver Interface

The judge-facing entry point in `solver.py` is:

```python
solve(n, m, budget, families, run) -> (family, secret)
```

- `n`: number of input qubits.
- `m`: number of output qubits.
- `budget`: maximum number of oracle queries.
- `families`: list of candidate family names.
- `run(circuit, shots)`: judge-supplied function that executes a circuit against the sealed oracle and returns measurement counts.

Oracle calls are represented using the required placeholder instruction:

```python
Gate("oracle", n + m, [])
```

The solver accesses the oracle through the supplied `run()` interface. It does not require access to the hidden oracle's internal table or metadata.

For `balanced`, `randfun`, and `perm`, the returned secret is `None`. For the other families, the secret must match the challenge's required format.

## 3. Repository Structure

The repository contains the following files, where applicable:

| File | Purpose |
|---|---|
| `solver.py` | Judge-compatible adaptive classifier |
| `requirements.txt` | Python package dependencies |
| `README.md` | Setup, execution, and reproducibility instructions |
| `evaluate.py` | Local seeded evaluation harness |
| `oracle_functions.py` | Public reference oracle functions |
| `oracle_simulator.py` | Local reversible oracle implementation |
| `local_runner.py` | Local circuit runner and noise experiments |
| `test_oracle_superposition.py` | Phase-kickback test |
| `evaluation_results.json` | Saved local evaluation output |
| `technical_report.pdf` | Technical report, limited to three pages |
| `AI_use_disclosure.pdf` | Separate AI-use disclosure, limited to one page |

Include only files that actually exist in the final repository. Test logs and report results should correspond to the exact submitted version of `solver.py`.

## 4. Environment and Installation

The implementation uses Python 3.10 or newer and Qiskit 2.x. The challenge permits the standard library, NumPy, and SciPy.

On Windows, activate the project environment or use the existing virtual environment directly.

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the required packages are already installed at the specified versions, no additional installation may be necessary.

## 5. Running Tests

Run the local component tests if the corresponding files are included:

```powershell
.\.venv\Scripts\python.exe .\oracle_functions.py
.\.venv\Scripts\python.exe .\oracle_simulator.py
.\.venv\Scripts\python.exe .\test_oracle_superposition.py
.\.venv\Scripts\python.exe .\local_runner.py
```

Run the local evaluation harness:

```powershell
.\.venv\Scripts\python.exe .\evaluate.py --seed 20261019 --trials 1 --budget 60 --output evaluation_results.json
```

The local evaluator is a development tool, not the official event judge. Its results depend on the generated oracle instances, random seed, query budget, and noise realization. Local results should therefore be treated as empirical evidence rather than a guarantee of performance on the hidden evaluation suites.

## 6. Current Evaluation Snapshot

One local evaluation run used:

- Seed: `20261019`
- Trials per family: `1`
- Query budget: `60`
- Oracle misfire probability: `0.04`
- Readout flip probability: `0.02`

Observed overall accuracy was **87.5% (7/8 instances)**.

| Family | Correct in this run | Queries used |
|---|---:|---:|
| `constant` | Yes | 32 |
| `balanced` | Yes | 32 |
| `linear` | Yes | 16 |
| `grover` | No | 32 |
| `randfun` | Yes | 32 |
| `simon` | Yes | 50 |
| `simon2` | Yes | 50 |
| `perm` | Yes | 50 |

The Grover instance was classified as `randfun`. Since this snapshot contains only one instance per family, it does not establish generalization, worst-family robustness, or stress-noise performance.

Additional evaluation results should be added only after running the corresponding experiments. In particular, budget-cap comparisons at 20, 30, 40, 50, and 60 queries, baseline comparisons, and doubled-noise tests should be reported separately when available.

## 7. Reproducibility and Limitations

For reproducible reporting:

1. Preserve the exact submitted `solver.py`.
2. Record the command, seed, trial count, query budget, and noise settings for each evaluation.
3. Keep the raw JSON output or equivalent test log in the repository.
4. Report both overall accuracy and per-family results.
5. Document misclassifications and avoid generalizing from a small test sample.

The solver uses heuristic decision thresholds under a limited query budget. Noise can affect both family classification and secret recovery. In particular, the current evaluation snapshot demonstrates that Grover classification remains a known failure case.

No claim of quantum advantage is made solely from this small local benchmark. Any comparison with classical or single-algorithm baselines must be supported by separately recorded experiments.

## 8. Submission Documents

The submission includes:

- **Technical report:** strategy, thresholds, query-budget allocation, baseline comparisons, diagnostics, and failure analysis; maximum three pages.
- **AI-use disclosure:** a separate one-page statement describing AI assistance and independent verification.
- **Test logs:** saved outputs from local experiments used to support claims in the report.

The final repository should contain the exact solver and documentation that are submitted for evaluation.