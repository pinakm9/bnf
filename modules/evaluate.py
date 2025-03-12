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
    indices = (np.arange(D)[:, None] + np.arange(-I, I + 1)) % D
    X = x[indices]
    df_dict = {"Time": [t] * D}
    df_dict.update({f"Space_{i}": X[:, i] for i in range(2*I+1)})
    yhat, _ = model.predict(pd.DataFrame(df_dict),  quantiles=())
    return  yhat[0].mean(axis=0)



@ut.timer
def parallel_forecast(model, t, x, I):
    D = len(x)
    indices = (np.arange(D)[:, None] + np.arange(-I, I + 1)) % D
    X = x[indices]
    df_dict = {"Time": [t] * np.prod(x.shape)}
    df_dict.update({f"Space_{i}": X[:, i, :].flatten() for i in range(2*I+1)})
    yhat, _ = model.predict(pd.DataFrame(df_dict),  quantiles=())
    return yhat[0].mean(axis=0).reshape(x.shape)



@ut.timer
def multistep_forecast(model, t, x, I, n_steps, dt):
    Y = np.zeros((len(x), n_steps))
    y = x + 0.
    Y[:, 0] = y
    for step in range(1, n_steps):
        Y[:, step] = forecast(model, t, Y[:, step-1], I)
        t += dt
    return Y
