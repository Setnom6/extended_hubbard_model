"""
Script for running and plotting a detuning protocol using DQD21 and ManyBodyHamiltonian.
Saves both the figure and the parameters used, with a timestamp, in subfolders 'figures' and 'data'.
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

from DQD21 import DQD21, DQDParameters, BasesNames
from src.plottingMethods import plot_protocol_results, setupLogger

setupLogger(current_dir)

logging.info("Starting dynamics protocol...")

# --- Parameters ---
parameters_file = os.path.join(root_dir, 'global_parameters.json')
with open(parameters_file, 'r') as f:
    params = json.load(f)

dqd = DQD21(params=params)


# --- Protocol definition ---

# General parameters

totalPoints = 300
T1 = 0.0  # Spin relaxation time in ns
T2star = 0.0  # Dephasing time in ns
cutOffN = None

# Characteristic values obtained from further analysis
interactionDetuning = 4.3502
expectedPeriod = 1.48  # ns
peakDetuningreadOut = dqd.params[DQDParameters.U0.value]


# Options for general protocol (comment if one wants simple evolution)

parameterToChange = DQDParameters.E_R.value
initialStateDict = {}
slopesShapes = [
    [peakDetuningreadOut, interactionDetuning, 2*expectedPeriod],  # Ramp down
    [interactionDetuning, interactionDetuning, 1.25*expectedPeriod],  # Ramp up
    [interactionDetuning, peakDetuningreadOut, 2*expectedPeriod],  # Ramp to phase point
    [peakDetuningreadOut, peakDetuningreadOut, 1*expectedPeriod],  # Phase accumulation
]

"""# Options for simple evolution (uncomment if wanted)

parameterToChange = None
initialStateDict = {DQDParameters.E_R.value: 0.0}
slopesShapes =[[interactionDetuning, interactionDetuning, 10*expectedPeriod]]
dqd.updateParams({DQDParameters.E_R.value: interactionDetuning})"""

# --- Build protocol ---
tlistNano, eiValues = dqd.build_protocol_timeline(slopesShapes, totalPoints)

# --- Decoherence rates ---

dephasing = dqd.gamma_from_time(T2star)
spinRelaxation = dqd.gamma_from_time(T1)

# --- Prepare options for time evolution ---
options = {
    'cutOffN': cutOffN,
    'parameterToChange': parameterToChange,
    'initialState': initialStateDict,
    'protocolDetails': slopesShapes,
    'totalPoints': totalPoints,
    'dephasingRate': dephasing,
    'spinRelaxationRate': spinRelaxation
}

# --- Run protocol ---
result = dqd.performTimeEvolution(options)

# --- Plotting (encapsulated) ---
if cutOffN is None:
    cutOffN = 4

labels = dqd.basesDict[BasesNames.SINGLET_TRIPLET_QUBIT_4.name]['labels']

singletIndices, tripletIndices = dqd.get_current_indices(BasesNames.SINGLET_TRIPLET_QUBIT_4.name, cutOffN)

fig, axes, populations = plot_protocol_results(
    result=result,
    cutoffN=cutOffN,
    labels=labels,
    sweepvalues=eiValues,
    subspace0Indices=singletIndices,
    subspace1Indices=tripletIndices,
    tlist=tlistNano
)

# Adjust some axes labels

axes[1].set_ylabel(r"$E_I$ (meV)")
axes[1].set_title("Detuning sweep")
axes[2].set_title("Subspace populations (|0>=symmetric, |1>=antisymmetric)")

# --- Save results ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
figures_dir = os.path.join(current_dir, "figures")
data_dir = os.path.join(current_dir, "data")
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

fig_filename = os.path.join(figures_dir, f"detuning_protocol_{timestamp}.png")
fig.savefig(fig_filename, bbox_inches="tight", dpi=300)

params_filename = os.path.join(data_dir, f"detuning_protocol_params_{timestamp}.json")
with open(params_filename, "w") as f:
    json.dump(getattr(dqd, "params", params), f, indent=4)

npz_filename = os.path.join(data_dir, f"detuning_protocol_data_{timestamp}.npz")
np.savez(
    npz_filename,
    tlistNano=tlistNano,
    populations=populations,
    labels=np.array(labels),
    eiValues=eiValues,
    singletIndices=np.array(singletIndices),
    tripletIndices=np.array(tripletIndices),
    slopesShapes=np.array(slopesShapes, dtype=object),
    params=getattr(dqd, "params", params),
)

print(f"Figure saved to: {fig_filename}")
print(f"Parameters saved to: {params_filename}")
print(f"Data saved to: {npz_filename}")

logging.info("All computations ended.")
plt.show()