"""
Script for analyzing eigenstate composition in a chosen basis using state similarity.

Loads parameters from 'global_parameters.json', computes eigenvectors, and for the first N eigenvalues,
prints the list of basis vectors (labels) that contribute more than 0.5% to each eigenstate.
"""

import sys
import os
import json

# Add root directory to sys.path so 'src' can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from DQD21 import DQD21, BasesNames

# Load parameters
parameters_file = os.path.join(root_dir, 'global_parameters.json')
with open(parameters_file, 'r') as f:
    params = json.load(f)

# Create DQD system
dqd = DQD21(params=params)

# Choose basis for state similarity analysis
basis_name = dqd.basesNames[BasesNames.SINGLET_TRIPLET.value]

# Compute eigenvalues and eigenvectors
eigvals, eigvecs = dqd.calculate_eigenvalues_and_eigenvectors()

N = 10  # Number of eigenstates to analyze
threshold = 0.005  # 0.5%

print(f"Analyzing the first {N} eigenstates in basis: {basis_name}\n")
for i in range(min(N, eigvecs.shape[1])):
    eigenstate = eigvecs[:, i]
    classification = dqd.classify_eigenstate_in_basis(eigenstate, basis_name)
    significant = [f"{entry['label']} ({entry['probability']*100:.2f}%)"
                   for entry in classification if entry['probability'] > threshold]
    print(f"Eigenstate {i} (energy = {eigvals[i]:.4f}):")
    for s in significant:
        print(f"  {s}")
    print()