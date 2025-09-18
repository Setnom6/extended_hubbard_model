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

# Add root directory to sys.path so 'src' can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from DQD21 import DQD21, DQDParameters, BasesNames
from src.plottingMethods import setupLogger, plotCurrentMap


# ---------------- dynamics ----------------
def runDynamics(parameters, maxTime, totalPoints, eiValue, cutOffN, T1, T2star):
    dqd = DQD21(parameters)
    slopesShapes = [[eiValue, eiValue, maxTime]]
    dephasing = dqd.gamma_from_time(T2star)
    spinRelaxation = dqd.gamma_from_time(T1)
    dqd.updateParams({DQDParameters.E_R.value: eiValue})
    parameterToChange = None
    initialStateDict = {DQDParameters.E_R.value: 0.0}

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
    return [np.abs(current) for current in sum0Subspace]


# ---------------- freq estimation ----------------
def estimateRabiFrequency(signal, times, paddingFactor=4):
    signal = np.array(signal) - np.mean(signal)
    N = len(signal)
    dt = times[1] - times[0]

    N_padded = paddingFactor * N
    yf = fft(signal, n=N_padded)
    xf = fftfreq(N_padded, dt)[:N_padded // 2]
    spectrum = 2.0 / N * np.abs(yf[:N_padded // 2])
    spectrum[0] = 0.0

    i = np.argmax(spectrum)
    if 1 <= i < len(spectrum) - 1:
        y0, y1, y2 = spectrum[i - 1], spectrum[i], spectrum[i + 1]
        dx = (y2 - y0) / (2 * (2 * y1 - y2 - y0))
        dominantFreq = xf[i] + dx * (xf[1] - xf[0])
    else:
        dominantFreq = xf[i]

    return dominantFreq


def computeFreqsForDetuningSweep(currents, detuningList, times):
    n = len(detuningList)
    freqs = np.zeros(n)
    maxCurrents = np.zeros(n)
    for i in range(n):
        signal = currents[i, :]
        freqs[i] = estimateRabiFrequency(signal, times)
        maxCurrents[i] = np.max(np.abs(signal))
    return freqs, maxCurrents


# ---------------- detuning analysis ----------------
def findCentralDetuning(currents, detuningList, times, minCurrentFraction=0.05):
    freqsNs, maxCurrents = computeFreqsForDetuningSweep(currents, detuningList, times)
    currentThreshold = minCurrentFraction * np.max(maxCurrents)
    sortedIndices = np.argsort(freqsNs)

    for idx in sortedIndices:
        if maxCurrents[idx] >= currentThreshold:
            bestIndex = idx
            break
    else:
        bestIndex = sortedIndices[0]

    return detuningList[bestIndex], freqsNs[bestIndex], freqsNs


def findSweetSpotsFromFreqs(currents, detuningList, times, minCurrentFraction=0.05, relGradThreshold=0.1):
    freqs, maxCurrents = computeFreqsForDetuningSweep(currents, detuningList, times)
    grads = np.gradient(freqs, detuningList)
    absGrads = np.abs(grads)
    maxAbsGrad = np.max(absGrads) if np.max(absGrads) > 0 else 1.0
    gradMask = absGrads <= relGradThreshold * maxAbsGrad
    currentThreshold = minCurrentFraction * np.max(maxCurrents)
    currentMask = maxCurrents >= currentThreshold
    combinedMask = gradMask & currentMask
    sweetIndices = np.where(combinedMask)[0]
    if len(sweetIndices) == 0:
        minIdx = np.argmin(absGrads)
        sweetIndices = np.array([minIdx])
    return sweetIndices, grads


def formatFrequencies(freqsGHz):
    maxFreq = np.max(freqsGHz)
    if maxFreq >= 1:
        return freqsGHz, "GHz"
    elif maxFreq >= 1e-3:
        return freqsGHz * 1e3, "MHz"
    elif maxFreq >= 1e-6:
        return freqsGHz * 1e6, "kHz"
    else:
        return freqsGHz * 1e9, "Hz"



# ---------------- main ----------------
if __name__ == "__main__":
    setupLogger(current_dir)
    logging.info("Starting dynamics protocol...")

    parameters_file = os.path.join(root_dir, "global_parameters.json")
    with open(parameters_file, "r") as f:
        params = json.load(f)

    interactionDetuningList = np.linspace(4.2, 4.6, 500)
    cutOffN =None
    totalPoints = 500
    maxTime = 3.0
    T1 = 0.0
    T2star = 0.0
    parameterToChange = DQDParameters.B_X.value
    arrayOfParameters = np.arange(0.05, 0.75, 0.05)
    tlistNano = np.linspace(0, maxTime, totalPoints)

    numCores = min(24, cpu_count())
    logging.info(f"Using {numCores} cores with joblib.")

    symmetryAxes, rabiFreqs_sym, rabiPeriods_sym, listOfDetuningsFound = [], [], [], []

    for idx, value in enumerate(arrayOfParameters):
        new_params = deepcopy(params)
        new_params[parameterToChange] = value

        currents = Parallel(n_jobs=numCores)(
            delayed(runDynamics)(new_params, maxTime, totalPoints, eiValue, cutOffN, T1, T2star)
            for eiValue in interactionDetuningList
        )
        currents = np.array(currents)

        symDetuning, rabiFreq, freqsNs = findCentralDetuning(currents, interactionDetuningList, tlistNano)
        sweetIndices, grads = findSweetSpotsFromFreqs(currents, interactionDetuningList, tlistNano)
        symmetryAxes.append(symDetuning)
        rabiFreqs_sym.append(rabiFreq)
        rabiPeriods_sym.append(1.0 / rabiFreq)
        listOfDetuningsFound.append(symDetuning)

        logging.info(f"[{idx+1}/{len(arrayOfParameters)}] Param={value:.4f}, CentralDet={symDetuning:.4f}, RabiFreq={rabiFreq:.4f}")

        fig, ax = plotCurrentMap(currents, tlistNano, interactionDetuningList, parameterToChange, value, symDetuning)

        # --- Save results ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        figures_dir = os.path.join(current_dir, "figures")
        data_dir = os.path.join(current_dir, "data")
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        fig_filename = os.path.join(figures_dir, f"detuning_protocol_{timestamp}_{value:.3f}.png")
        fig.savefig(fig_filename, bbox_inches="tight", dpi=300)

        paramsFilename = os.path.join(data_dir, f"detuning_protocol_params_{timestamp}.json")
        with open(paramsFilename, "w") as f:
            json.dump(params, f, indent=4)

        npz_filename = os.path.join(data_dir, f"detuning_protocol_data_{timestamp}.npz")
        np.savez(
            npz_filename,
            tlistNano=tlistNano,
            currents=currents,
            eiValues=interactionDetuningList,
            params=new_params,
        )

        print(f"Figure saved to: {fig_filename}")
        print(f"Parameters saved to: {paramsFilename}")
        print(f"Data saved to: {npz_filename}")

        plt.close(fig)
        sleep(0.1)

    logging.info("All computations ended.")


    # ------------------- Combined final figure -------------------
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    freqsScaled, unit = formatFrequencies(rabiFreqs_sym)

    # Rabi frequency
    axes[0].plot(arrayOfParameters, freqsScaled, "o-", label="Rabi frequency central detuning")
    axes[0].set_ylabel(f"Frequency ({unit})")
    axes[0].set_title(f"Rabi frequency vs {parameterToChange}")
    axes[0].legend()
    axes[0].grid(True)

    # Rabi period
    axes[1].plot(arrayOfParameters, rabiPeriods_sym, "o-", label="Rabi period central detuning")
    axes[1].set_ylabel("Period (ns)")
    axes[1].set_title(f"Rabi period vs {parameterToChange}")
    axes[1].legend()
    axes[1].grid(True)

    # Central detuning
    axes[2].plot(arrayOfParameters, symmetryAxes, "o-", label="Central detuning")
    axes[2].set_xlabel(f"{parameterToChange}")
    axes[2].set_ylabel("Detuning (meV)")
    axes[2].set_title(f"Central detuning vs {parameterToChange}")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    # --- Save combined figure ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figures_dir = os.path.join(current_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    combined_fig_filename = os.path.join(figures_dir, f"detuning_protocol_combined_{timestamp}.png")
    fig.savefig(combined_fig_filename, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Combined figure saved to: {combined_fig_filename}")
