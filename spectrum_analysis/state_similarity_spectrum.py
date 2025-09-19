"""
Script for plotting eigenvalues colored by state similarity using plot_eigenvalues_state_similarity.

The system used is a double quantum dot (DQD) system modeled by the Extended Hubbard Model.
Parameters are loaded from 'global_parameters.json' in the same directory.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import logging

# Add root directory to sys.path so 'src' can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.plottingMethods import plot_eigenvalues_state_similarity, setupLogger
from DQD21 import DQD21, BasesNames, DQDParameters


setupLogger(current_dir)
logging.info("Computing spectrum with state similarity...")

# Load parameters
parameters_file = os.path.join(root_dir, 'global_parameters.json')
with open(parameters_file, 'r') as f:
    params = json.load(f)

# Create DQD system
dqd = DQD21(params=params)

# Define the parameter to sweep and its values
parameter_name = DQDParameters.E_R.value  # Example: detuning parameter
x_values = np.linspace(0.0, 8.0, 1000)
results_list = []

for val in x_values:
    eigvals, eigvecs = dqd.calculate_eigenvalues_and_eigenvectors(parameterToChange=parameter_name, newValue=val)
    results_list.append((eigvals - val, eigvecs)) # Shift eigenvalues by -val for better visualization in the case of detuning

# Choose basis for state similarity coloring
basis_name = dqd.basesNames[BasesNames.SINGLET_TRIPLET_QUBIT_4.value] # Any singlet triplet will work, but we are interested in lower states

# Plot and save
fig, ax = plot_eigenvalues_state_similarity(
    results_list=results_list,
    x_values=x_values,
    mbh=dqd,
    basis_name=basis_name,
    max_eigenvalues=10,
    figsize=(8, 6),
    scatter_kwargs={'s': 2},
    color_palette_name='tab20'
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
figures_dir = os.path.join(current_dir, "figures")
data_dir = os.path.join(current_dir, "data")
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

fig_filename = os.path.join(figures_dir, f"state_similarity_spectrum_{timestamp}.png")
fig.savefig(fig_filename, bbox_inches='tight', dpi=300)

params_filename = os.path.join(data_dir, f"state_similarity_spectrum_params_{timestamp}.json")
with open(params_filename, 'w') as f:
    json.dump(getattr(dqd, "params", params), f, indent=4)

print(f"Figure saved to: {fig_filename}")
print(f"Parameters saved to: {params_filename}")

logging.info("Computations ended.")
plt.show()