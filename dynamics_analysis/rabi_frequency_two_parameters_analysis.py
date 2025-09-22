"""
Script for running detuning × time maps at different parameter values,
analyzing Rabi frequencies in a 2D map.
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
from matplotlib.cm import get_cmap
from scipy.fft import fft, fftfreq

# Add root directory to sys.path so 'src' can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from DQD21 import DQD21, DQDParameters, BasesNames
from src.plottingMethods import setupLogger, plot_2D_rabi_scan


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
def computeRabiIfCurrent(signal, times, currentThreshold=1e-3, paddingFactor=4):
    maxCurrent = np.max(np.abs(signal))
    if maxCurrent < currentThreshold:
        return np.nan, maxCurrent, np.nan
    else:
        freq, dominance = estimateRabiFrequency(signal, times, paddingFactor)
        return freq, maxCurrent, dominance
    
def estimateRabiFrequency(signal, times, paddingFactor=4):
    """
    Estima la frecuencia dominante de un señal y su dominancia física.
    Dominancia = altura del pico / anchura a mitad de altura (FWHM)
    """
    signal = np.array(signal) - np.mean(signal)
    N = len(signal)
    dt = times[1] - times[0]

    # FFT con padding
    N_padded = paddingFactor * N
    yf = fft(signal, n=N_padded)
    xf = fftfreq(N_padded, dt)[:N_padded // 2]
    spectrum = 2.0 / N * np.abs(yf[:N_padded // 2])
    spectrum[0] = 0.0  # ignorar DC

    # Pico máximo
    peakIdx = np.argmax(spectrum)
    dominantFreq = xf[peakIdx]

    # --- Dominancia física: altura / FWHM ---
    peakHeight = spectrum[peakIdx]
    halfHeight = peakHeight / 2

    # Buscar los índices a izquierda y derecha donde cruza la mitad
    left = peakIdx
    while left > 0 and spectrum[left] > halfHeight:
        left -= 1
    right = peakIdx
    while right < len(spectrum)-1 and spectrum[right] > halfHeight:
        right += 1

    fwhm = (right - left) * (xf[1] - xf[0])  # ancho en GHz
    dominance = peakHeight / fwhm if fwhm > 0 else np.nan

    return dominantFreq, dominance # Dominant frequency in GHz, dominance as ratio



# ---------------- main computation ----------------
def computeRabiFrequencyMap(params):
    setupLogger(current_dir)

    cutOffN =None
    totalPoints = 500
    totalDetunings = 300


    maxTime = 10.0
    T1 = 0.0
    T2star = 0.0
    
    # Range for 2D space
    detuningList = np.linspace(3.0, 6.0, totalDetunings)

    # for bx (change for other parameter)
    eps = 1e-5
    halfPoints = totalDetunings // 2

    left = np.linspace(-1.0, -eps, halfPoints, endpoint=True)
    right = np.linspace(eps, 1.0, totalDetunings - halfPoints, endpoint=True)
    bParallelList = np.concatenate((left, right))

    timesNs = np.linspace(0, maxTime, totalPoints)

    maxCores = min(24, cpu_count())
    logging.info(f"Using {maxCores} cores with joblib.")

    # Matrices para almacenar resultados
    rabiFreqMap = np.zeros((len(bParallelList), len(detuningList)))
    dominanceMap = np.zeros((len(bParallelList), len(detuningList)))
    currentMap = np.zeros((len(bParallelList), len(detuningList)))

    # Computar para cada valor de B_parallel
    for i, b_parallel in enumerate(bParallelList):
        logging.info(f"Processing B_parallel = {b_parallel:.4f} ({i+1}/{len(bParallelList)})")
        
        parameters = deepcopy(params)
        parameters[DQDParameters.B_X.value] = b_parallel
        
        # Computar dinámica para todos los detunings
        currents = Parallel(n_jobs=maxCores)(
            delayed(runDynamics)(parameters, maxTime, totalPoints, detuning, cutOffN, T1, T2star)
            for detuning in detuningList
        )
        currents = np.array(currents)
        
        # Calcular frecuencias de Rabi para cada detuning
        for j, detuning in enumerate(detuningList):
            signal = currents[j, :]
            freq, maxCurrent, dominance = computeRabiIfCurrent(signal, timesNs, currentThreshold=1e-2)
            rabiFreqMap[i, j] = freq
            currentMap[i, j] = maxCurrent
            dominanceMap[i, j] = dominance

    return detuningList, bParallelList, rabiFreqMap, currentMap, dominanceMap, params

# ---------------- gradient computation ----------------
def computeGradients(detuningList, bParallelList, rabiFreqMap):
    """
    Calcula gradientes 2D de las frecuencias de Rabi
    """
    # Gradiente respecto a detuning (eje x)
    grad_detuning = np.gradient(rabiFreqMap, detuningList, axis=1)
    
    # Gradiente respecto a B_parallel (eje y)
    grad_bparallel = np.gradient(rabiFreqMap, bParallelList, axis=0)
    
    # Magnitud del gradiente total
    grad_magnitude = np.sqrt(grad_detuning**2 + grad_bparallel**2)
    
    return grad_detuning, grad_bparallel, grad_magnitude

# ---------------- sweet spots detection ----------------
def findSweetSpots2D(grad_magnitude, threshold_factor=0.05):
    """
    Encuentra sweet spots donde la magnitud del gradiente es mínima
    """
    min_grad = np.min(grad_magnitude)
    max_grad = np.max(grad_magnitude)
    threshold = min_grad + threshold_factor * (max_grad - min_grad)
    
    sweet_spots = grad_magnitude <= threshold
    return sweet_spots


if __name__ == "__main__":
    parameters_file = os.path.join(root_dir, "global_parameters.json")
    with open(parameters_file, "r") as f:
        params = json.load(f)

    params[DQDParameters.GV_L.value] = 0.5*params[DQDParameters.GV_R.value]

    # Compute 2D map
    detuningList, bParallelList, rabiFreqMap, currentMap, dominanceMap, fixedParameters = computeRabiFrequencyMap(params)
        
    # Compute gradients
    grad_detuning, grad_bparallel, grad_magnitude = computeGradients(detuningList, bParallelList, rabiFreqMap)
        
    # Plot everything
    fig, axes = plot_2D_rabi_scan(detuningList, bParallelList, rabiFreqMap, grad_detuning, dominanceMap)
        
    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figures_dir = os.path.join(current_dir, "figures")
    data_dir = os.path.join(current_dir, "data")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    fig_filename = os.path.join(figures_dir, f"rabi_frequency_analysis_{timestamp}.png")
    fig.savefig(fig_filename, bbox_inches="tight", dpi=300)

    paramsFilename = os.path.join(data_dir, f"detuning_protocol_params_{timestamp}.json")
    with open(paramsFilename, "w") as f:
            json.dump(fixedParameters, f, indent=4)

    npz_filename = os.path.join(data_dir, f"rabi_frequency_analysis_{timestamp}.npz")
    np.savez(npz_filename,
                detuningList=detuningList,
                bParallelList=bParallelList,
                rabiFreqMap=rabiFreqMap,
                grad_magnitude=grad_magnitude,
                grad_detuning=grad_detuning,
                currentMap=currentMap,
                fixedParameters=fixedParameters,
                dominanceMap=dominanceMap)
        
    print(f"Figure saved to: {fig_filename}")
    print(f"Parameters saved to: {paramsFilename}")
    print(f"Data saved to: {npz_filename}")

    plt.close(fig)
        
    logging.info("Analysis completed and results saved.")