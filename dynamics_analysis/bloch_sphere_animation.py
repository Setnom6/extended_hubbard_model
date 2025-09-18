
"""
Script for running and plotting a detuning protocol using DQD21 and ManyBodyHamiltonian and getting an animation with bloch vector.
Saves both the figure and the parameters used, with a timestamp, in subfolders 'figures' and 'data'.
"""

import sys
import os
import numpy as np
import json
from datetime import datetime
from matplotlib.animation import PillowWriter
import logging


# Add root directory to sys.path so 'src' can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from DQD21 import DQD21, DQDParameters, BasesNames
from src.plottingMethods import plot_bloch_sphere, setupLogger

setupLogger(current_dir)
logging.info("Starting sweep protocol...")


# --- Parameters ---
parameters_file = os.path.join(root_dir, 'global_parameters.json')
with open(parameters_file, 'r') as f:
    params = json.load(f)

dqd = DQD21(params=params)


# --- Protocol definition (example, adapt as needed) ---

totalPoints = 500
T1 = 0.0  # Spin relaxation time in ns
T2star = 0.0  # Dephasing time in ns
cutOffN = None

# Detuning protocol shape (adapt as needed)
interactionDetuning = 3.6102
expectedPeriod = 11.41  # ns
peakDetuningreadOut = dqd.params[DQDParameters.U0.value]


slopesShapes = [
    [peakDetuningreadOut, interactionDetuning, 2*expectedPeriod],  # Ramp down
    [interactionDetuning, interactionDetuning, 1.25*expectedPeriod],  # Ramp up
    [interactionDetuning, peakDetuningreadOut, 1.0*expectedPeriod],
    [peakDetuningreadOut, peakDetuningreadOut, 1.0*expectedPeriod],
    [peakDetuningreadOut, interactionDetuning, 1.75*expectedPeriod],
    [interactionDetuning, peakDetuningreadOut, 2*expectedPeriod],  # Ramp to phase point
    [peakDetuningreadOut, peakDetuningreadOut, 1*expectedPeriod],  # Phase accumulation
]


# --- Build protocol ---
tlistNano, eiValues = dqd.build_protocol_timeline(slopesShapes, totalPoints)

# --- Decoherence rates ---

dephasing = dqd.gamma_from_time(T2star)
spinRelaxation = dqd.gamma_from_time(T1)

# --- Prepare options for time evolution ---
options = {
    'cutOffN': cutOffN,
    'parameterToChange': DQDParameters.E_R.value,
    'initialState': {},
    'protocolDetails': slopesShapes,
    'totalPoints': totalPoints,
    'dephasingRate': dephasing,
    'spinRelaxationRate': spinRelaxation
}

# --- Run protocol ---
result = dqd.performTimeEvolution(options)

logging.info("Dynamics computed. Starting Bloch sphere animation...")

# --- Plotting (encapsulated) ---
if cutOffN is None:
    cutOffN = 4

labels = dqd.basesDict[BasesNames.SINGLET_TRIPLET_QUBIT_4.name]['labels']
singletIndices, tripletIndices = dqd.get_current_indices(BasesNames.SINGLET_TRIPLET_QUBIT_4.name, cutOffN)

fig, ani, populations, blochVectors = plot_bloch_sphere(
    result=result,
    tlist=tlistNano,
    labels=labels,
    iSym=singletIndices,
    iAnti=tripletIndices,
    sweepValues=eiValues,
    actualDetuning=eiValues,
    detThreshold=1.05*interactionDetuning,
    cutOffN=cutOffN
)

# --- Save results ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
figures_dir = os.path.join(current_dir, "figures")
data_dir = os.path.join(current_dir, "data")
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

mp4_filename = os.path.join(figures_dir, f"st_qubit_{timestamp}.mp4")
ani.save(mp4_filename.replace(".mp4", ".gif"), writer=PillowWriter(fps=25))

params_filename = os.path.join(data_dir, f"detuning_protocol_params_{timestamp}.json")
with open(params_filename, "w") as f:
    json.dump(getattr(dqd, "params", params), f, indent=4)

npz_filename = os.path.join(data_dir, f"st_qubit_data_{timestamp}.npz")
np.savez(
    npz_filename,
    tlistNano=tlistNano,
    populations=populations,
    labels=np.array(labels),
    eiValues=eiValues,
    singletIndices=np.array(singletIndices),
    tripletIndices=np.array(tripletIndices),
    blochVectors=blochVectors
)

print(f"Animation saved to: {mp4_filename}")
print(f"Parameters saved to: {params_filename}")
print(f"Data saved to: {npz_filename}")

logging.info("All computations ended.")