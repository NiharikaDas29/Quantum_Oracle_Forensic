# QryptoNox technical report

## Strategy

For one-bit outputs, the solver first spends eight single-shot computational
basis queries on fixed inputs. It then runs an eight-shot Bernstein--Vazirani
phase experiment. A concentrated nonzero phase result, together with the basis
probes, identifies `linear` and recovers `[a, b]`. A concentrated zero phase
result with constant basis observations identifies `constant`. A sparse-oracle
phase signature triggers an eight-iteration Grover experiment with five shots;
the repeated marked-input result identifies `grover`. The nominal path uses
56 queries (eight basis, eight phase, and forty Grover) and leaves a small
reserve. Remaining one-bit cases are separated as `balanced` versus `randfun`
using the observed one-rate.

For eight-bit outputs, the solver runs the Simon experiment: input Hadamards,
one oracle, input Hadamards, and input measurement. Each observed vector `y`
is scored against all nonzero masks using `parity(y & mask) == 0`. A single
strong mask identifies `simon`; three strong masks closed under XOR identify
`simon2`; otherwise the result is `perm`. The hidden subspace elements are
returned in ascending order.

All circuits use at most 17 qubits in the current strategy and all query
reservations are cumulative inside `solve()`. The solver never uses local
family metadata.

## Noise and limitations

The local harness models each oracle gate as either its coherent reversible
operation or the identity with the official 4% probability, and flips each
measured bit independently with probability 2%. The solver uses thresholds
chosen for the 60-query setting; sparse statistical families can fail on
unfavorable noise draws. In particular, distinguishing a constant function
from a Grover target at zero is intrinsically difficult with few queries.

## Evidence

The reproducible command is:

```powershell
.\.venv\Scripts\python.exe .\evaluate.py --seed 20261016 --trials 2 --output evaluation_results.json
```

The development run used two generated instances per family (16 total), fixed
seed `20261016`, 60-query budget, 4% oracle misfire, and 2% readout flips.
It achieved 15/16 correct tasks (93.75%); the only failure was one Grover
secret. The exact per-instance output is saved in `evaluation_results.json`.
One-trial-per-family budget-cap and doubled-noise results are saved in
`evaluation_caps_stress.json`. These are development samples, not claims about
the hidden event suite.
