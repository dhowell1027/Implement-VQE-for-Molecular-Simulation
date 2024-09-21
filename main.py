#from qiskit.providers.aer import Aer

from qiskit import Aer, IBMQ
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.drivers import PySCFDriver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.algorithms import GroundStateEigensolver
from qiskit_nature.transformers import FreezeCoreTransformer
from qiskit.utils import QuantumInstance

# Load IBMQ account for real quantum hardware (optional, requires IBMQ credentials)
# Uncomment if you want to run on real quantum devices, requires an IBMQ account
# IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q')
# backend = provider.get_backend('ibmq_manila')  # Choose a real quantum device

# Define the molecular structure (H2 molecule)
driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735', unit='Angstrom', charge=0, spin=0, basis='sto3g')
molecule = driver.run()

# Convert the molecule to a qubit operator
qubit_converter = QubitConverter(mapper=ParityMapper())
problem = ElectronicStructureProblem(driver, transformers=[FreezeCoreTransformer()])
qubit_op = qubit_converter.convert(problem.second_q_ops()[0])

# Use a TwoLocal ansatz for the variational form
ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')

# Set the optimizer (COBYLA) and variational algorithm (VQE)
optimizer = COBYLA(maxiter=1000)

# Create a quantum instance with the simulator backend
quantum_instance_simulator = QuantumInstance(Aer.get_backend('statevector_simulator'))
vqe_simulator = VQE(ansatz, optimizer, quantum_instance=quantum_instance_simulator)

# Solve for the ground state energy using VQE (simulator)
ground_state_solver_simulator = GroundStateEigensolver(qubit_converter, vqe_simulator)
result_simulator = ground_state_solver_simulator.solve(problem)

# Print the calculated ground state energy (simulator)
print(f"Calculated Ground State Energy (Simulator): {result_simulator.total_energies[0].real} Hartree")

# Optional: Uncomment the following to run on a real quantum computer
# quantum_instance_real = QuantumInstance(backend=backend)
# vqe_real = VQE(ansatz, optimizer, quantum_instance=quantum_instance_real)
# ground_state_solver_real = GroundStateEigensolver(qubit_converter, vqe_real)
# result_real = ground_state_solver_real.solve(problem)
# print(f"Real Quantum Hardware Calculated Ground State Energy: {result_real.total_energies[0].real} Hartree")
