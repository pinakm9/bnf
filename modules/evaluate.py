import pandas as pd
import numpy as np  
import glob
import utility as ut


def get_files(test_path):
    files = []
    for file in glob.glob(f"{test_path}/test_*.csv"):
        index = int(file.split('_')[-1].split('.')[0])
        files.append((file, index))
    return sorted(files, key = lambda x: x[-1])


def evaluate(model, test_path, length):
    files = get_files(test_path)
    y = np.zeros((len(files), length))
    for file, index in files:
        data = pd.read_csv(file).iloc[:length, :]
        yhat, _ = model.predict(data,  quantiles=(0.025, 0.5, 0.975))
        y[index] = yhat[0].mean(axis=0)
    return y 


def get_data(npy_file, sep):
    data = np.load(npy_file)    
    return data[:, :sep], data[:, sep:]



def forecast(model, t, x, I):
    D = len(x)
    y = np.zeros_like(x)
    for index in range(D):
        ii = (index + np.arange(-I, I+1).astype(int)) % D
        df_dict = {"Time": [t]}
        for j, i in enumerate(ii):
            df_dict[f"Space_{j}"] = [x[i]]
        df = pd.DataFrame(df_dict)
        yhat, _ = model.predict(df,  quantiles=(0.025, 0.5, 0.975))
        y[index] = yhat[0].mean(axis=0)[0]
    return y


@ut.timer
def parallel_forecast(model, t, x, I):
    D, N = x.shape
    y = np.zeros_like(x)
    for index in range(D):
        ii = (index + np.arange(-I, I+1).astype(int)) % D
        df_dict = {"Time": [t] * N}
        for j, i in enumerate(ii):
            df_dict[f"Space_{j}"] = x[i]
        df = pd.DataFrame(df_dict)
        yhat, _ = model.predict(df,  quantiles=(0.025, 0.5, 0.975))
        y[index] = yhat[0].mean(axis=0)
    return y


def multistep_forecast(model, t, x, I, n_steps, dt):
    Y = np.zeros((n_steps, x.shape[0], x.shape[1]))
    y = x + 0.
    Y[0] = y
    for step in range(n_steps - 1):
        y = parallel_forecast(model, t, y, I)
        t += dt
        Y[step + 1] = y
    return Y
