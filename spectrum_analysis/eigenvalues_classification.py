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

from DQD21 import DQD21, BasesNames, DQDParameters

# Load parameters
parameters_file = os.path.join(root_dir, 'global_parameters.json')
with open(parameters_file, 'r') as f:
    params = json.load(f)

# Create DQD system
dqd = DQD21(params=params)

# Choose basis for state similarity analysis
basis_name = dqd.basesNames[BasesNames.SINGLET_TRIPLET.value]

# Values of the parameter to iterate over
parameterToChange = DQDParameters.E_R.value
parameterValues = [4.4, 4.5, 4.6, 4.7]  # ejemplo con 3 valores
N = 5  # number of eigenstates to analyze
threshold = 0.005  # 0.5%

# Precompute results
results = {}  # dict: eigenstate index -> list of dicts per parameter
for value in parameterValues:
    eigvals, eigvecs = dqd.calculate_eigenvalues_and_eigenvectors(
        parameterToChange=parameterToChange, newValue=value
    )
    for i in range(min(N, eigvecs.shape[1])):
        eigenstate = eigvecs[:, i]
        classification = dqd.classify_eigenstate_in_basis(eigenstate, basis_name)
        significant = [f"{entry['label']} ({entry['probability']*100:.2f}%)"
                       for entry in classification if entry['probability'] > threshold]
        if i not in results:
            results[i] = []
        results[i].append({
            'parameter': value,
            'energy': eigvals[i],
            'significant': significant
        })

# --- Compute column widths for proper alignment ---
numCols = len(parameterValues)
colWidths = [0] * numCols

# Consider header, energy, and significant states
for col in range(numCols):
    # Header width
    colWidths[col] = max(colWidths[col], len(f"{parameterToChange} = {results[0][col]['parameter']}"))
    # Energy width
    colWidths[col] = max(colWidths[col], len(f"{results[0][col]['energy']:.4f}"))
    # Significant states
    for i in range(N):
        for entry in results[i][col]['significant']:
            colWidths[col] = max(colWidths[col], len(entry))

# --- Print results with aligned columns ---
for i in range(N):
    print(f"\nEigenstate {i}:")
    
    # Header row
    header = " | ".join(f"{f'{parameterToChange} = {res['parameter']}':<{colWidths[j]}}" 
                        for j, res in enumerate(results[i]))
    print(header)
    
    # Energy row
    if parameterToChange == DQDParameters.E_R.value:
        energies = " | ".join(f"{f'{res['energy']-res['parameter']:.4f}':<{colWidths[j]}}" 
                          for j, res in enumerate(results[i]))
    else:
        energies = " | ".join(f"{f'{res['energy']:.4f}':<{colWidths[j]}}" 
                          for j, res in enumerate(results[i]))
    print(energies)
    
    # Significant states
    max_len = max(len(res['significant']) for res in results[i])
    for j in range(max_len):
        row_entries = []
        for col, res in enumerate(results[i]):
            if j < len(res['significant']):
                row_entries.append(f"{res['significant'][j]:<{colWidths[col]}}")
            else:
                row_entries.append(" " * colWidths[col])
        print(" | ".join(row_entries))
