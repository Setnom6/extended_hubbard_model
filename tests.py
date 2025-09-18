
from src.plottingMethods import plot_eigenvalues_bipartition, plot_eigenvalues_state_similarity, plot_hamiltonian_heatmap
from DQD21 import DQD21, DQDParameters, BasesNames
import numpy as np
import matplotlib.pyplot as plt

params = {
        'bz': 1.5,
        'bx': 0.1,
        'gsLeft': 2.0,
        'gsRight': 2.0,
        'gvLeft': 20.0,
        'gvRight': 20.0,
        'DeltaSO': 0.066,
        'DeltaKK': 0.02,
        'ERight': 12.0,
        't': 0.05,
        't_soc': 0.01,
        'U0': 6.0,
        'U1': 1.5,
        'X': 0.1,
        'A': 0.01,
        'P': 0.005,
        'J': 0.000075,
        'g_ortho': 10,
        'g_zz': 100,
        'g_z0': 6.66,
        'g_0z': 6.66
    }

dqd = DQD21(params=params)
HSingletTriplet = dqd.getHamiltonianInBase(BasesNames.SIMMETRIC_ANTISYMMETRIC.name)

# Plot hamiltonian heatmap with sector sizes and labels
sector_sizes = {
    'Symmetric': 18,
    'Antisymmetric': 10,
}   

fig, ax = plot_hamiltonian_heatmap(
    HSingletTriplet,
    sector_sizes=sector_sizes,
    figsize=(8, 8),
    min_val=1e-5
)
plt.show()
