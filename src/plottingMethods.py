from src.ManyBodyHamiltonian import ManyBodyHamiltonian
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap
from matplotlib.animation import FuncAnimation
from qutip import Bloch, Qobj, Result


import os
import logging

def setupLogger(logDir):
        os.makedirs(os.path.join(logDir, "data"), exist_ok=True)
        logPath = os.path.join(logDir, "data", "log_results.txt")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(logPath),
                logging.StreamHandler()
            ]
        )

#---------------------- Eigenvalues, spectrum and Hamiltonian ------------------------------------------------------------------

def plot_eigenvalues_bipartition(
    results_list,
    x_values,
    mbh: ManyBodyHamiltonian,
    basis_name: str,
    splitValue: int,
    max_eigenvalues: int = None,
    figsize=(8, 5),
    scatter_kwargs=None
):
    """
    Plots eigenvalues for each point in x_values, coloring each eigenvalue according to the index of the most similar basis vector.

    Args:
        results_list: List of (eigs, eigvecs) tuples, one for each x value.
        x_values: Array-like, values for the x-axis (length must match results_list).
        mbh: ManyBodyHamiltonian object (must have classify_eigenstate_in_basis).
        basis_name: Name of the basis to use for classification.
        splitValue: Integer, index to split coloring (<= splitValue: blue, > splitValue: red).
        max_eigenvalues: Maximum number of eigenvalues to plot (from lowest up).
        figsize: Tuple, figure size.
        scatter_kwargs: Optional dict, extra kwargs for plt.scatter.

    Returns:
        fig, ax: The matplotlib figure and axes objects.
    """
    if scatter_kwargs is None:
        scatter_kwargs = {}

    eigvals = []
    color_indices = []

    # Precompute all eigenvalues and color indices
    for i, (eigs, eigvecs) in enumerate(results_list):
        if max_eigenvalues is not None:
            eigs = eigs[:max_eigenvalues]
            eigvecs = eigvecs[:, :max_eigenvalues]
        eigvals.append(eigs)
        color_row = []
        basis_labels = mbh.basesDict[basis_name]['labels']
        for j in range(len(eigs)):
            eigvec = eigvecs[:, j]
            classification = mbh.classify_eigenstate_in_basis(eigvec, basis_name)
            max_label = classification[0]['label']
            idx = basis_labels.index(max_label)
            # Use -1 for blue, +1 for red (for use with cmap='bwr')
            color_row.append(-1 if idx <= splitValue else 1)
        color_indices.append(color_row)

    eigvals = np.array(eigvals)  # shape (totalPoints, num_eigenvalues)
    color_indices = np.array(color_indices)  # shape (totalPoints, num_eigenvalues)
    x_values = np.array(x_values)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each eigenvalue as a line of points with color
    for j in range(eigvals.shape[1]):
        ax.scatter(
            x_values,
            eigvals[:, j],
            c=color_indices[:, j],
            cmap='bwr',
            vmin=-1, vmax=1,
            **scatter_kwargs
        )

    ax.set_xlabel("Parameter")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Eigenvalues bipartitioned by basis similarity")
    plt.tight_layout()
    return fig, ax


def plot_eigenvalues_state_similarity(
    results_list,
    x_values,
    mbh,
    basis_name: str,
    max_eigenvalues: int = None,
    figsize=(8, 5),
    scatter_kwargs=None,
    color_palette_name='tab20'
):
    """
    Plots eigenvalues for each point in x_values, coloring each eigenvalue according to the most similar basis state.

    Args:
        results_list: List of (eigs, eigvecs) tuples, one for each x value.
        x_values: Array-like, values for the x-axis (length must match results_list).
        mbh: ManyBodyHamiltonian object (must have classify_eigenstate_in_basis).
        basis_name: Name of the basis to use for classification.
        max_eigenvalues: Maximum number of eigenvalues to plot (from lowest up).
        figsize: Tuple, figure size.
        scatter_kwargs: Optional dict, extra kwargs for plt.scatter.
        color_palette_name: Name of matplotlib colormap to use for state coloring.

    Returns:
        fig, ax: The matplotlib figure and axes objects.
    """
    if scatter_kwargs is None:
        scatter_kwargs = {}

    basis_labels = mbh.basesDict[basis_name]['labels']
    base_cmap = get_cmap(color_palette_name)
    N = 20 if color_palette_name == 'tab20' else base_cmap.N
    color_palette = [base_cmap(i / N) for i in range(N)]

    eigvals = []
    color_indices = []
    unique_colors_used = {}

    # Precompute all eigenvalues and color indices
    for i, (eigs, eigvecs) in enumerate(results_list):
        if max_eigenvalues is not None:
            eigs = eigs[:max_eigenvalues]
            eigvecs = eigvecs[:, :max_eigenvalues]
        eigvals.append(eigs)
        color_row = []
        for j in range(len(eigs)):
            eigvec = eigvecs[:, j]
            classification = mbh.classify_eigenstate_in_basis(eigvec, basis_name)
            max_label = classification[0]['label']
            idx = basis_labels.index(max_label)
            color_row.append(idx)
            if idx % N not in unique_colors_used:
                unique_colors_used[idx % N] = color_palette[idx % N]
        color_indices.append(color_row)

    eigvals = np.array(eigvals)
    color_indices = np.array(color_indices)
    x_values = np.array(x_values)

    fig, ax = plt.subplots(figsize=figsize)

    for j in range(eigvals.shape[1]):
        colors = [color_palette[idx % N] for idx in color_indices[:, j]]
        ax.scatter(
            x_values,
            eigvals[:, j],
            color=colors,
            **scatter_kwargs
        )

    ax.set_xlabel("Parameter")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Eigenvalues colored by most similar basis state")
    ax.grid(True)

    # Add legend for unique colors used
    handles = []
    for idx, rgba in sorted(unique_colors_used.items()):
        label = basis_labels[idx]
        handles.append(Patch(facecolor=rgba, edgecolor='black', label=label))
    if handles:
        ax.legend(
            handles=handles,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            title='State similarity',
            frameon=False
        )

    plt.tight_layout()
    return fig, ax


def plot_hamiltonian_heatmap(
    H: np.ndarray,
    sector_sizes: dict = None,
    figsize=(8, 8),
    min_val: float = 1e-5
):
    """
    Plots a heatmap of the absolute value of a Hamiltonian matrix, with optional block boundaries and labels inside the matrix.

    Args:
        H: np.ndarray, Hamiltonian matrix in the chosen basis.
        sector_sizes: dict, mapping sector labels to their sizes (e.g. {'(2,0)': 6, '(1,1)': 16, '(0,2)': 6}).
        figsize: tuple, figure size.
        min_val: float, minimum value for log scale.

    Returns:
        fig, ax: The matplotlib figure and axes objects.
    """
    absH = np.abs(H).copy()
    absH[absH < min_val] = min_val

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(absH, cmap='viridis', norm=LogNorm(vmin=min_val, vmax=np.max(absH)))

    # Draw block boundaries and labels if sector_sizes is provided
    if sector_sizes is not None:
        boundaries = np.cumsum(list(sector_sizes.values()))
        for pos in boundaries[:-1]:
            ax.axhline(pos - 0.5, color='white', linewidth=1.5)
            ax.axvline(pos - 0.5, color='white', linewidth=1.5)

        # Block labels inside the matrix
        starts = np.cumsum([0] + list(sector_sizes.values())[:-1])
        labels = list(sector_sizes.keys())
        for start, size, label in zip(starts, sector_sizes.values(), labels):
            center = start + size / 2 - 0.5
            ax.text(center, center, label, ha='center', va='center', fontsize=12, color='white', weight='bold', alpha=0.8)

    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label='log(|H|)', shrink=0.7)
    plt.tight_layout()
    return fig, ax






# -------------------------------- Dynamics ------------------------------------------------------------

def plot_protocol_results(result, cutoffN, labels, sweepvalues, subspace0Indices, subspace1Indices, tlist):
    """
    Generate plots for a general protocol.
    Populations will be obtained from result.
    sweepvalues is agnostic with respect which parameter was changed in the protocol, so axes labels must be changed externally.
    labels make the correspondence between the position in the matrix representation of the density matrix and the name given to that base state.
    subspaceIndices indicate which index correspond to each state (no labels needed, just indices). For example, in the SWT representation with 4x4 Hamiltonians,
    singlet subspace have indices 1 and 2, while triplet one have 0 and 3.

    Args:
        result: qutip.Result from mesolve
        cutoffN: int, number of states (defaults to 4 if None)
        labels: list of str, basis state labels
        sweepvalues: np.ndarray, sweep values during protocol
        subspace0Indices: list[int], indices for subspace |0>
        subspace1Indices: list[int], indices for subspace |1>
        slopesShapes: list, protocol details [[start, end, duration], ...]
        interactionDetuning: float, reference detuning
    """

    populations = np.array([state.diag() for state in result.states])

    # --- Create figure ---
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, sharex=True, figsize=(10, 10), height_ratios=[3, 1, 1]
    )

    # Panel 1: Individual populations
    for i in range(min(cutoffN, len(labels))):
        ax1.plot(tlist, populations[:, i], label=labels[i])
    ax1.set_ylabel("Population")
    ax1.set_title("Population dynamics")
    ax1.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax1.grid()

    # Panel 2: Detuning
    ax2.plot(tlist, sweepvalues, color="black", linewidth=2)
    ax2.grid()

    # Panel 3: Subspace populations
    sum1Subspace = np.sum(populations[:, subspace1Indices], axis=1)
    sum0Subspace = np.sum(populations[:, subspace0Indices], axis=1)
    sumTotal = sum0Subspace + sum1Subspace

    ax3.plot(tlist, sum0Subspace, label="|0> read out", linestyle="--", color="tab:blue")
    ax3.plot(tlist, sum1Subspace, label="|1> read out", linestyle="--", color="tab:green")
    ax3.plot(tlist, sumTotal, label="Total", linestyle="-", color="tab:red")
    ax3.set_xlabel("Time (ns)")
    ax3.set_ylabel("Populations")
    ax3.set_title("Subspace populations")
    ax3.legend()
    ax3.grid()

    plt.subplots_adjust(hspace=0.4, right=0.75)

    return fig, (ax1, ax2, ax3), populations


def rhoToBlochHybrid(rhoFull, iSym, iAnti, detuning=None, detThreshold=None, sRep=None, tRep=None):
    """Compute Bloch vector (sx, sy, sz) for ST qubit given full density matrix and symmetry indices.
    This is likely to be deleted as is not working properly"""
    rho = Qobj(np.asarray(rhoFull, dtype=complex)) if not isinstance(rhoFull, Qobj) else rhoFull

    # Dynamics choice of representative for the singlet (1,1 or 2,0 representative based on detuning)
    if detuning is not None and detThreshold is not None and len(iSym) > 1:
        sRepDynamic = iSym[1] if detuning > detThreshold else iSym[0]
    else:
        sRepDynamic = sRep if sRep is not None else iSym[0]

    tRep = tRep if tRep is not None else iAnti[0]

    # populations projectors
    dim = rho.shape[0]
    eye = np.eye(dim)
    P_S = sum(Qobj(eye[:, [i]]) * Qobj(eye[:, [i]]).dag() for i in iSym)
    P_T = sum(Qobj(eye[:, [i]]) * Qobj(eye[:, [i]]).dag() for i in iAnti)
    P_Q = P_S + P_T

    # Coherence only with representatives
    ketS = Qobj(eye[:, [sRepDynamic]])
    ketT = Qobj(eye[:, [tRep]])
    Sx = ketS * ketT.dag() + ketT * ketS.dag()
    Sy = -1j * ketS * ketT.dag() + 1j * ketT * ketS.dag()

    # Compute population in each subspace
    p = (rho * P_Q).tr().real
    if p < 1e-12:
        return np.array([0.0, 0.0, 0.0])

    # Compute coherences
    sx = ((rho * Sx).tr() / p).real
    sy = ((rho * Sy).tr() / p).real

    # Z axis can be understood as the hole populations due to read out
    sz = (((rho * P_S).tr() - (rho * P_T).tr()) / p).real

    return np.array([sx, sy, sz], dtype=float)



def rhoToBlochHybridCanonical(rhoFull, iSym, iAnti):
    """
    Compute Bloch vector (sx, sy, sz) for a pure state density matrix.
    iSym = indices for subspace A
    iAnti = indices for subspace B
    """
    rho = Qobj(np.asarray(rhoFull, dtype=complex)) if not isinstance(rhoFull, Qobj) else rhoFull

    # Obtain state vector assuming is pure
    evals, evecs = rho.eigenstates()
    idx = np.argmax(np.real(evals)) 
    psi = evecs[idx].full().flatten()

    # Project onto qubit subspaces A (symmetric) and B (antisymmetric)
    psiA = psi[iSym]
    psiB = psi[iAnti]

    normA = np.linalg.norm(psiA)
    normB = np.linalg.norm(psiB)

    if normA < 1e-14 and normB < 1e-14:
        return np.array([0.0, 0.0, 0.0])

    # Obtain coherences between subspaces
    coh = np.vdot(psiA, psiB)  # <psiA|psiB>

    # Bloch vector
    sx = 2 * coh.real
    sy = 2 * coh.imag
    sz = normA**2 - normB**2

    return np.array([sx, sy, sz], dtype=float)


def plot_bloch_sphere(result: Result, tlist, labels, iSym, iAnti, sweepValues,
                      cutOffN=None, nFrames=300, fps=25):
    """
    Create animation of ST qubit Bloch vector and population dynamics.

    Returns:
        fig, ani, populations, blochVectors
    """

    states = result.states if hasattr(result, "states") else result
    populations = np.array([state.diag() for state in states])

    # Precompute Bloch vectors
    blochVectors = np.array([
        rhoToBlochHybridCanonical(rho, iSym, iAnti)
        for rho in states
    ])

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.3, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1:, 1], projection="3d")

    # Plot populations
    lines = []
    for i in range(min(cutOffN, len(labels))):
        line, = ax1.plot(tlist, populations[:, i], label=labels[i])
        lines.append(line)
    ax1.set_ylabel("Population")
    ax1.set_title("Individual populations")
    ax1.grid()

    # Legend in separate axis
    ax_legend.axis("off")
    ax_legend.legend(lines, [l.get_label() for l in lines], loc="center", ncol=2)

    # Detuning sweep
    ax2.plot(tlist, sweepValues, color="black", linewidth=2)
    ax2.set_ylabel("Parameter changed")
    ax2.set_xlabel("Time (ns)")
    ax2.set_title("sweep")
    ax2.grid()

    # Subspace populations
    sumTriplet = np.sum(populations[:, iAnti], axis=1)
    sumSinglet = np.sum(populations[:, iSym], axis=1)
    sumTotal = sumTriplet + sumSinglet
    ax3.plot(tlist, sumTriplet, "--", label="Triplet", color="tab:blue")
    ax3.plot(tlist, sumSinglet, "--", label="Singlet", color="tab:green")
    ax3.plot(tlist, sumTotal, "-", label="Total", color="tab:red")
    ax3.set_xlabel("Time (ns)")
    ax3.set_ylabel("Populations")
    ax3.set_title("Subspace populations")
    ax3.legend()
    ax3.grid()

    # Vertical lines
    vline1 = ax1.axvline(x=tlist[0], color="red", linestyle=":")
    vline2 = ax2.axvline(x=tlist[0], color="red", linestyle=":")
    vline3 = ax3.axvline(x=tlist[0], color="red", linestyle=":")

    # Bloch sphere
    b = Bloch(fig=fig, axes=ax4)
    b.vector_color = ["#1f77b4"]
    b.point_color = ["#2ca02c"]
    b.view = [90, 30]
    b.xlabel = [r"$|\ominus\rangle$", r"$|\oplus\rangle$"]
    b.ylabel = [r"$|-\rangle$", r"$|+\rangle$"]
    b.zlabel = [r"$|S\rangle$", r"$|T\rangle$"]

    def drawBloch(i):
        b.clear()
        b.add_vectors(blochVectors[i])
        k = max(0, i - 10)
        b.add_points(blochVectors[k:i+1].T)
        b.make_sphere()

    def update(i):
        drawBloch(i)
        vline1.set_xdata([tlist[i], tlist[i]])
        vline2.set_xdata([tlist[i], tlist[i]])
        vline3.set_xdata([tlist[i], tlist[i]])
        return []

    framesToSave = np.linspace(0, len(tlist) - 1, nFrames, dtype=int) if nFrames < len(tlist) else range(len(tlist))
    ani = FuncAnimation(fig, update, frames=framesToSave, interval=1000/fps, blit=False)

    return fig, ani, populations, blochVectors


# ----------------------------------------- Rabi frequency analysis ---------------------------


def plot_rabi_vs_time(currents, tlistNano, detuningList, paramName, paramValue, symDetuning):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        currents,
        aspect="auto",
        origin="lower",
        extent=[tlistNano[0], tlistNano[-1], detuningList[0], detuningList[-1]],
        cmap="viridis"
    )
    fig.colorbar(im, ax=ax, label="I (no Pauli Blockade)")
    ax.axhline(symDetuning, color="red", linestyle="--", label=f"Central detuning: {symDetuning:.4f} meV")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("E_R (meV)")
    ax.set_title(f"Current map ({paramName} = {paramValue:.4f})")
    ax.legend()
    return fig, ax

def phaseVsTime(currents, tlistNano, detuningList):
    fig, (axCurr, axPhi) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    phis = np.arccos(2* currents -1) * 180 / np.pi 
    
    for i, detuning in enumerate(detuningList):
        axCurr.plot(
            tlistNano,
            currents[i, :]
        )
    axCurr.set_ylabel("I (no Pauli Blockade)")
    axCurr.set_title("Final population after Ramsey protocol")
    
    for i, detuning in enumerate(detuningList):
        axPhi.plot(
            tlistNano,
            phis[i, :]
        )
    axPhi.set_xlabel("Plateau (ns)")
    axPhi.set_ylabel("Phase φ (deg)")
    axPhi.set_title("Computed phase angle")
    
    fig.tight_layout()
    axCurr.grid()
    axPhi.grid()
    return fig, (axCurr, axPhi)


def formatFrequencies(freqsGHz):
    maxFreq = np.max(freqsGHz)
    try:
        if maxFreq >= 1:
            return freqsGHz, "GHz"
        elif maxFreq >= 1e-3:
            return freqsGHz * 1e3, "MHz"
        elif maxFreq >= 1e-6:
            return freqsGHz * 1e6, "kHz"
        else:
            return freqsGHz * 1e9, "Hz"
        
    except:
        return freqsGHz, "GHz"

def plot_combined_rabi_results(arrayOfParameters, rabiFreqs_sym, rabiPeriods_sym, symmetryAxes, parameterToChange):
    """
    Plots Rabi frequency, Rabi period, and central detuning in a single figure.
    Returns fig and axes for further use.
    """
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
    return fig, axes

def plot_2D_rabi_scan(detuningList, bParallelList, rabiFreqMap, grad_detuning, dominanceMap):
    """
    Creates 2D plots of the results.
    Regions excluded (NaN in rabiFreqMap) are shown in light gray.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Selected indices of specific b_values
    selected_indices = [
        np.argmin(np.abs(bParallelList - (-0.85))),
        np.argmin(np.abs(bParallelList - (-0.25))),
        np.argmin(np.abs(bParallelList - (0.15))),
        np.argmin(np.abs(bParallelList - (0.75)))
    ]
    colors = ['red', 'blue', 'green', 'purple']
    
    # Create modified colormap to show NaN as gray
    cmap_rabi = get_cmap('viridis_r').copy()
    cmap_rabi.set_bad(color='lightgray')
    
    # 1. Rabi frequency map
    im1 = axes[0, 0].imshow(rabiFreqMap, aspect='auto', 
                            extent=[detuningList[0], detuningList[-1], 
                                    bParallelList[0], bParallelList[-1]],
                            origin='lower', cmap=cmap_rabi)
    axes[0, 0].set_xlabel('Detuning (meV)')
    axes[0, 0].set_ylabel('B_parallel (T)')
    axes[0, 0].set_title('Rabi Frequency Map (GHz)')
    plt.colorbar(im1, ax=axes[0, 0], label="GHz")

    for idx, color in zip(selected_indices, colors):
        axes[0, 0].axhline(y=bParallelList[idx], color=color, linestyle='--', alpha=0.7, linewidth=1.5)
    
    # 2. Gradient with sign with respect to detuning
    im2 = axes[0, 1].imshow(grad_detuning, aspect='auto',
                             extent=[detuningList[0], detuningList[-1],
                                     bParallelList[0], bParallelList[-1]],
                             origin='lower', cmap='bwr', vmin=-np.max(np.abs(grad_detuning)), vmax=np.max(np.abs(grad_detuning)))
    axes[0, 1].set_xlabel('Detuning (meV)')
    axes[0, 1].set_ylabel('B_parallel (T)')
    axes[0, 1].set_title('Gradient wrt Detuning (GHz per meV)')
    plt.colorbar(im2, ax=axes[0, 1], label="d(freq)/d(detuning)")

    for idx, color in zip(selected_indices, colors):
        axes[0, 1].axhline(y=bParallelList[idx], color=color, linestyle='--', alpha=0.7, linewidth=1.5)
    
    # 3. Frequency dominance map
    im3 = axes[1, 0].imshow(dominanceMap, aspect='auto',
                             extent=[detuningList[0], detuningList[-1],
                                     bParallelList[0], bParallelList[-1]],
                             origin='lower', cmap='inferno')
    axes[1, 0].set_xlabel('Detuning (meV)')
    axes[1, 0].set_ylabel('B_parallel (T)')
    axes[1, 0].set_title('Dominance: Peak Height / FWHM')
    plt.colorbar(im3, ax=axes[1, 0], label="Dominance (Height/GHz)")

    for idx, color in zip(selected_indices, colors):
        axes[1, 0].axhline(y=bParallelList[idx], color=color, linestyle='--', alpha=0.7, linewidth=1.5)
    
    # 4. Transverse cuts (frequency only)
    ax4 = axes[1, 1]

    for idx, color in zip(selected_indices, colors):
        b_val = bParallelList[idx]
        rabiRow = rabiFreqMap[idx, :]

        # Frequency (left axis)
        ax4.plot(detuningList, rabiRow, color=color, label=f'B_parallel = {b_val:.3f} T')

    # Axes and legend configuration
    ax4.set_xlabel('Detuning (meV)')
    ax4.set_ylabel('Frequency (GHz)')
    ax4.set_title('Transverse Cuts')
    ax4.legend(loc='upper right')
    ax4.grid(True)
    
    plt.tight_layout()
    return fig, axes
