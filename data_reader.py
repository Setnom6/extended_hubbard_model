"""
General script to select and load .npz files from a given directory.
After loading, you can access the variables inside and process them
depending on the dataset structure.

In this example, we specialize to simulation results containing:
- tlistNano
- currents
- eiValues
and plot them with phaseVsTime.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add root directory to sys.path so 'src' can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Directory where .npz files are stored
data_dir = os.path.join(current_dir, "dynamics_analysis", "server", "data")

# --- Step 1: List all .npz files ---
npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]

if not npz_files:
    print(f"No .npz files found in {data_dir}")
    sys.exit(1)

# Sort by modification time (most recent last)
npz_files.sort(key=lambda f: os.path.getmtime(os.path.join(data_dir, f)))

print("Available .npz files:")
for i, f in enumerate(npz_files, 1):
    print(f"[{i}] {f}")

choice = input("Select the number of the file to open (ENTER for the latest): ")

if choice.strip() == "":
    npz_filename = os.path.join(data_dir, npz_files[-1])
else:
    try:
        idx = int(choice) - 1
        npz_filename = os.path.join(data_dir, npz_files[idx])
    except (ValueError, IndexError):
        print("Invalid selection.")
        sys.exit(1)

print(f"\nLoading file: {npz_filename}")

# --- Step 2: Load the file ---
data = np.load(npz_filename, allow_pickle=True)

# Show what’s inside (keys)
print("\nVariables stored in this .npz file:")
print(list(data.keys()))

# --- Step 3: Specialized processing (only if this file matches the simulation structure) ---
# For your simulation files we expect: tlistNano, currents, eiValues
if all(key in data for key in ["tlistNano", "currents", "eiValues"]):
    from src.plottingMethods import phaseVsTime

    tlistNano = data["tlistNano"]
    currents = data["currents"]
    eiValues = data["eiValues"]

    fig, axes = phaseVsTime(currents, tlistNano, eiValues)
    plt.show()
else:
    print("\nNo plotting performed. This file does not match the expected simulation format.")


