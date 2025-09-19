"""
Script for plotting a heatmap of a Hamiltonian matrix with optional sector sizes and labels.

The system used in the example is a double quantum dot (DQD) system modeled by the Extended Hubbard Model.
The parameters for the DQD are obtained from 'global_parameters.json' and the Hamiltonian is computed in 
the Fock basis. 

A projection onto other defined bases in DQD21.py is also possible.
"""

import sys
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime
import logging

# Add root directory to sys.path so 'src' can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.plottingMethods import plot_hamiltonian_heatmap, setupLogger
from DQD21 import DQD21, BasesNames

setupLogger(current_dir)
logging.info("Starting Hamiltonian heatmap...")

# Obtain parameters from JSON file in the same directory as this script
parameters_file = os.path.join(root_dir, 'global_parameters.json')
with open(parameters_file, 'r') as f:
    params = json.load(f)

# Create DQD system and obtain Hamiltonian in the desired basis
dqd = DQD21(params=params)

HProjected = dqd.getHamiltonianInBase(dqd.basesNames[BasesNames.SINGLET_TRIPLET.value])
sector_sizes = dqd.basesSectors[BasesNames.SINGLET_TRIPLET.value]

# Plot hamiltonian heatmap with sector sizes and labels
fig, ax = plot_hamiltonian_heatmap(
    HProjected,
    sector_sizes=sector_sizes,
    figsize=(8, 8),
    min_val=1e-5
)

# Generate timestamp for filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Prepare output directories
figures_dir = os.path.join(current_dir, "figures")
data_dir = os.path.join(current_dir, "data")
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Save figure
fig_filename = os.path.join(figures_dir, f"hamiltonian_heatmap_{timestamp}.png")
fig.savefig(fig_filename, bbox_inches='tight', dpi=300)

# Save parameters used by dqd
params_filename = os.path.join(data_dir, f"hamiltonian_heatmap_params_{timestamp}.json")
with open(params_filename, 'w') as f:
    json.dump(getattr(dqd, "params", params), f, indent=4)

print(f"Figure saved to: {fig_filename}")
print(f"Parameters saved to: {params_filename}")

logging.info("Computations ended.")
plt.show()

