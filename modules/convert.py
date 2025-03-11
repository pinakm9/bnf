import numpy as np 
import pandas as pd
import time


def npy2csv(npy_file, sep=int(1e5), index=0, I=10, dt=1):
    """
    Converts a .npy file to CSV format, splitting data into training and testing sets.

    Parameters
    ----------
    npy_file : str
        Path to the .npy file to be converted.
    sep : int, optional
        The index separating training and testing data within the .npy file. Defaults to 1e5.
    index : int, optional
        The reference index used for slicing data around it. Defaults to 0.
    I : int, optional
        The range around the index to include in the training data. Defaults to 10.
    dt : int, optional
        The time step increment for the time column in the CSV. Defaults to 1.

    The function loads data from the specified .npy file, processes it into a specified format,
    and saves it as CSV files for both training and testing datasets.
    """
    start_time = time.time()
    for name, data, start in [("train", np.load(npy_file)[:, :sep], 0), ("test", np.load(npy_file)[:, sep:], sep)]:
        if name == "train":
            ii = (index + np.arange(-I, I+1).astype(int)) % data.shape[0]
            data_x = data[ii, :-1]
            data_y = data[index, 1:]
            
            space_dim, time_steps = data_x.shape
            
            # Create column names
            column_names = [f"Space_{i}" for i in range(space_dim)]
            df = pd.DataFrame(data_x.T, columns=column_names)
            
            # Add a time column
            df.insert(0, "Time", (start + np.arange(time_steps)) * dt)
            df["Output"] = data_y 
            
            # Save to CSV
            df.to_csv(npy_file.replace(".npy", ".csv").replace("train", name), index=False)
        else:
            for index in range(data.shape[0]):
                ii = (index + np.arange(-I, I+1).astype(int)) % data.shape[0]
                print(f"Working on {ii}->{index}", end="\r")
                data_x = data[ii, :-1]
                data_y = data[index, 1:]
                
                space_dim, time_steps = data_x.shape
                
                # Create column names
                column_names = [f"Space_{i}" for i in range(space_dim)]
                df = pd.DataFrame(data_x.T, columns=column_names)
                
                # Add a time column
                df.insert(0, "Time", (start + np.arange(time_steps)) * dt)
                df["Output"] = data_y 
                
                # Save to CSV
                df.to_csv(npy_file.replace(".npy", ".csv").replace("train", name + f"_{index}"), index=False)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")


def arr2csv(data, index=0, I=10, t0=0, dt=0.01, filename="./train"):
    start_time = time.time()
    ii = (index + np.arange(-I, I+1).astype(int)) % data.shape[0]
    data_x = data[ii, :-1]
    data_y = data[index, 1:]
    
    space_dim, time_steps = data_x.shape
    
    # Create column names
    column_names = [f"Space_{i}" for i in range(space_dim)]
    df = pd.DataFrame(data_x.T, columns=column_names)
    
    # Add a time column
    df.insert(0, "Time", (t0 + np.arange(time_steps)) * dt)
    df["Output"] = data_y 
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Time taken to produce csv file: {time.time() - start_time:.2f} seconds")