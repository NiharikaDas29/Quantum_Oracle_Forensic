
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Create a 1-qubit quantum circuit
qc = QuantumCircuit(1)

# Put the qubit into a superposition
qc.h(0)

print("Quantum Circuit:")
print(qc.draw())

# Simulate the quantum state
state = Statevector.from_instruction(qc)

print("\nStatevector:")
print(state)

print("\nMeasurement probabilities:")
print(state.probabilities_dict())
