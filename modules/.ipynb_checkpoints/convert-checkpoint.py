import numpy as np 
import pandas as pd
import time


def convert_npy_to_csv2(npy_file, limit=int(1e5), index=0, I=10, dt=1):
"""
Load a 2D NumPy array from a .npy file and save it as a CSV file.

The first dimension represents time, and the second dimension represents 512 space dimensions.

Parameters:
npy_file (str): Path to the .npy file.
csv_file (str): Path to save the output CSV file.
"""

# Load the .npy file
for type, data, start in [("train", np.load(npy_file)[:, :sep], 0)]:
    data_x = data[:2*I+1, :-1]
    data_y = data[I, 1:]
    
    # Ensure it is a 2D array with the correct shape
    if data.ndim != 2:
        raise ValueError("The loaded NumPy array must be 2D.")
    
    space_dim, time_steps = data_x.shape
    
    # Create column names
    column_names = [f"Space_{i}" for i in range(space_dim)]
    df = pd.DataFrame(data_x.T, columns=column_names)
    
    # Add a time column
    df.insert(0, "Time", (start + np.arange(time_steps)) * dt)
    df["Output"] = data_y 
    
    # Save to CSV
    df.to_csv(npy_file.replace(".npy", ".csv").replace("train", type), index=False)