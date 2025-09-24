"""
Script for running detuning × time maps at different parameter values,
analyzing Rabi frequencies, and saving results.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import logging
from joblib import Parallel, delayed, cpu_count
from copy import deepcopy
from time import sleep
from scipy.fft import fft, fftfreq
from copy import deepcopy

# Add root directory to sys.path so 'src' can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from DQD21 import DQD21, DQDParameters, BasesNames
from src.plottingMethods import setupLogger, phaseVsTime, plot_combined_rabi_results


# ---------------- dynamics ----------------
def runDynamics(parameters, tlistNano, totalPoints, interactionDetuning, expectedPeriod, lowDetuning, cutOffN, T1, T2star):
    dqd = DQD21(parameters)
    dephasing = dqd.gamma_from_time(T2star)
    spinRelaxation = dqd.gamma_from_time(T1)
    parameterToChange = DQDParameters.E_R.value
    peakDetuning = 2.0*interactionDetuning
    initialStateDict = {}
    currents = []

    for time in tlistNano:
        slopesShapes = [
            [peakDetuning, interactionDetuning, 2.0*expectedPeriod],
            [interactionDetuning, interactionDetuning, 1.25*expectedPeriod],
            [interactionDetuning, lowDetuning, 1.0*expectedPeriod],
            [lowDetuning, lowDetuning, time],
            [lowDetuning, interactionDetuning, 1.0*expectedPeriod],
            [interactionDetuning, interactionDetuning, 1.75*expectedPeriod],
            [interactionDetuning, peakDetuning, 2.0*expectedPeriod]
            ]


        options = {
        'cutOffN': cutOffN,
        'parameterToChange': parameterToChange,
        'initialState': initialStateDict,
        'protocolDetails': slopesShapes,
        'totalPoints': totalPoints,
        'dephasingRate': dephasing,
        'spinRelaxationRate': spinRelaxation
        }

        result = dqd.performTimeEvolution(options)
        populations = np.array([state.diag() for state in result.states])

        subspace0Indices, _ = dqd.get_current_indices(BasesNames.SINGLET_TRIPLET_QUBIT_4.name, 4)
        sum0Subspace = np.sum(populations[:, subspace0Indices], axis=1)
        currents.append(np.abs(sum0Subspace[-1]))

    return np.array(currents)

    

def run_repetitive_detuning_protocol(params, interactionDetuning, expectedPeriod, plateauDetuningList,
                          tlistNano, totalPoints, cutOffN, T1, T2star, numCores):
    """
    Runs the detuning protocol for each value in arrayOfParameters, returning:
    symmetryAxes, rabiFreqs_sym, rabiPeriods_sym, listOfDetuningsFound
    """


    currents = Parallel(n_jobs=numCores)(
            delayed(runDynamics)(params, tlistNano, totalPoints, interactionDetuning, expectedPeriod, lowDetuning, cutOffN, T1, T2star)
            for lowDetuning in plateauDetuningList
        )
    currents = np.array(currents)

    # Plotting each current map
    fig, axes = phaseVsTime(currents, tlistNano, plateauDetuningList)

    # --- Save results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figures_dir = os.path.join(current_dir, "figures")
    data_dir = os.path.join(current_dir, "data")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    fig_filename = os.path.join(figures_dir, f"phase_frequency_fixed_cutoff_{cutOffN if cutOffN is not None else "SWT"}_{timestamp}.png")
    fig.savefig(fig_filename, bbox_inches="tight", dpi=300)

    params_to_save = deepcopy(params)
    params_to_save["coherentPeriod"] = expectedPeriod
    params_to_save["interactionDetuning"] = interactionDetuning
    params_to_save["slopesTotalTime"] = 2*expectedPeriod
    params_to_save["plateauDetuningList"] = plateauDetuningList 
    paramsFilename = os.path.join(data_dir, f"phase_frequency_fixed_params_cutoff_{cutOffN if cutOffN is not None else "SWT"}_{timestamp}.json")
    with open(paramsFilename, "w") as f:
            json.dump(params_to_save, f, indent=4)

    npz_filename = os.path.join(data_dir, f"phase_frequency_fixed_data_cutoff_{cutOffN if cutOffN is not None else "SWT"}_{timestamp}.npz")
    np.savez(
            npz_filename,
            tlistNano=tlistNano,
            currents=currents,
            eiValues=plateauDetuningList,
            params=params_to_save,
    )

    print(f"Figure saved to: {fig_filename}")
    print(f"Parameters saved to: {paramsFilename}")
    print(f"Data saved to: {npz_filename}")

    plt.close(fig)

    return None



# ---------------- main ----------------
if __name__ == "__main__":
    setupLogger(current_dir)
    logging.info("Starting dynamics protocol...")

    parameters_file = os.path.join(root_dir, "global_parameters.json")
    with open(parameters_file, "r") as f:
        params = json.load(f)

    gvLlist = [14.66, 22.0, 33.0, 44.0]

    for i, gvL in enumerate(gvLlist):

        params[DQDParameters.GV_L.value] = gvL
        # --- Simulation parameters ---
        interactionDetuning = - 0.043397*gvL + 5.4498 # Obtained numerically
        expectedPeriod = 1.0/0.6098 
        plateauDetuningList =[0.5*interactionDetuning]
        cutOffN = None
        totalPoints = 2000
        totalTimes = 50
        maxTime = 2.0*expectedPeriod
        T1 = 0.0
        T2star = 0.0
        tlistNano = np.linspace(0, maxTime, totalTimes)
        numCores = min(24, cpu_count())
        logging.info(f"Using {numCores} cores with joblib.")

        

        # Run the protocol
        run_repetitive_detuning_protocol(
            params, interactionDetuning, expectedPeriod, plateauDetuningList,
            tlistNano, totalPoints, cutOffN, T1, T2star, numCores
        )

        logging.info(f"Simulation {i+1} of {len(gvLlist)} completed.")

    logging.info("All computations ended.")
