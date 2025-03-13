import evaluate as ev
import utility as ut 
import l96
import convert
import bayesnf
import jax
import pandas as pd
import numpy as np
import os
import json
import wasserstein
import torch
import time
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


@ut.timer
def run_single(bnf_kwargs, data_gen_kwargs, train_kwargs, eval_kwargs):
    if not os.path.exists(train_kwargs["save_folder"]):
        os.makedirs(train_kwargs["save_folder"])
    results = {}

    # prepare data
    data, _ = l96.gen_data(**data_gen_kwargs)
    N = int(data.shape[1]/2)
    np.save("{}/data.npy".format(train_kwargs["save_folder"]), data) 
    # np.save("{}/test.npy".format(train_kwargs["save_folder"]), data[:, N:]) 
    convert.arr2csv(data[:, :N], index=0, I=train_kwargs["I"], t0=0, dt=data_gen_kwargs["dt"], filename=f"{train_kwargs['save_folder']}/train.csv")
    # del data
    std = data.std(axis=1)
    # train model
    start = time.time()
    model = bayesnf.BayesianNeuralFieldMAP(**bnf_kwargs)
    df_train = pd.read_csv(f"{train_kwargs['save_folder']}/train.csv")
    model_seed = np.random.randint(100)
    model.fit(df_train, seed=jax.random.PRNGKey(model_seed), num_epochs=train_kwargs["epochs"])
    train_time = time.time() - start
    print(f"Model trained for {train_time:.2f} seconds")

    # save model
    ut.save(model, os.path.abspath(f"{train_kwargs['save_folder']}/bnf_model"), step=train_kwargs["epochs"])

    # generate data for VPT
    x = data[:, N]
    Y = ev.multistep_forecast(model, 0, x, train_kwargs["I"], eval_kwargs["vpt_steps"], data_gen_kwargs["dt"])
    np.save("{}/vpt_trajectory.npy".format(train_kwargs["save_folder"]), Y)

    # calculate VPT
    Y0 = data[:, N:N+eval_kwargs["vpt_steps"]]
    l = np.argmax((((Y - Y0) / std[:, None])**2).sum(axis=0) < eval_kwargs["vpt_epsilon"]**2)
    results["VPT"] = float(l * data_gen_kwargs["dt"] / eval_kwargs["Lyapunov_time"])

    # generate data for RMSE
    x = data[:, N:N+eval_kwargs["n_RMSE"]]
    Y = ev.parallel_forecast(model, 0, x, train_kwargs["I"])
    np.save("{}/rmse_trajectory.npy".format(train_kwargs["save_folder"]), Y)

    # calculate RMSE and MAE
    Y0 = data[:, N:N+eval_kwargs["n_RMSE"]]
    results["RMSE"] = float(np.sqrt(np.mean(((Y - Y0)**2).sum(axis=0))))
    results["MAE"] = float(np.mean(np.abs(Y - Y0).sum(axis=0)))

    # generate data for Wasserstein
    x = data[:, N]
    Y = ev.multistep_forecast(model, 0, x, train_kwargs["I"], eval_kwargs["w2_steps"], data_gen_kwargs["dt"])
    # Y = np.squeeze(Y, axis=-1).T
    np.save("{}/w2_trajectory.npy".format(train_kwargs["save_folder"]), Y)

    # calculate Wasserstein distance
    Y0 = data[:, N:]
    A = torch.tensor(Y.T[:eval_kwargs["n_sample_w2"]], dtype=torch.float32)
    B = torch.tensor(Y0.T[:eval_kwargs["n_sample_w2"]], dtype=torch.float32)
    results["W2"] = float(wasserstein.sinkhorn_div(A, B).item())

    # add model size and training time
    results["training_time"] = float(train_time)
    results["model_size"] = int(ut.count_params(model))
    results["experiment_seed"] = int(data_gen_kwargs["train_seed"])
    results["model_seed"] = int(model_seed)

    # save results
    # print(results)
    with open(f"{train_kwargs['save_folder']}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    config = {}
    config.update(bnf_kwargs)
    config.update(data_gen_kwargs)
    config.update(train_kwargs)
    config.update(eval_kwargs)  

    with open(f"{train_kwargs['save_folder']}/config.json", "w") as f:  
        json.dump(config, f, indent=2)




def get_autonomous_params(root, D_r=256):
    train_kwargs = {'save_folder': f"{root}/autonomous", "epochs": 5000, "I": 4}
    data_gen_kwargs = {"dt": 1e-2, "train_size": int(2e5), "train_seed": np.random.randint(1e6), "test_num": 1}
    feature_cols = [f"Space_{i}" for i in range(2*train_kwargs["I"]+1)]
    bnf_kwargs   = {"width": D_r,
                    "depth": 2,
                    "freq": None,
                    "seasonality_periods": None,
                    "num_seasonal_harmonics": None,
                    "feature_cols": [f"Space_{i}" for i in range(9)],
                    "target_col": "Output",
                    "observation_model": 'NORMAL',
                    "timetype": 'float',
                    "standardize":  None,
                    "interactions": [(i, j) for i in range(len(feature_cols)) for j in range(len(feature_cols)) if i < j]}
    eval_kwargs = {"vpt_steps": 500, "n_RMSE": 500, "w2_steps": int(1e5), "vpt_epsilon": 0.5,\
                "Lyapunov_time": 1/2.27, "n_sample_w2": 20000}
    return bnf_kwargs, data_gen_kwargs, train_kwargs, eval_kwargs




def get_nonautonomous_params(root, D_r=256):
    train_kwargs = {'save_folder': f"{root}/autonomous", "epochs": 5000, "I": 4}
    data_gen_kwargs = {"dt": 1e-2, "train_size": int(2e5), "train_seed": np.random.randint(1e6), "test_num": 1}
    feature_cols = ["Time"] + [f"Space_{i}" for i in range(2*train_kwargs["I"]+1)]
    bnf_kwargs   = {"width": D_r,
                    "depth": 2,
                    "freq": None,
                    "seasonality_periods": None,
                    "num_seasonal_harmonics": None,
                    "feature_cols": [f"Space_{i}" for i in range(9)],
                    "target_col": "Output",
                    "observation_model": 'NORMAL',
                    "timetype": 'float',
                    "standardize":  None,
                    "interactions": [(i, j) for i in range(1,len(feature_cols)) for j in range(1, len(feature_cols)) if i < j]}
    
    eval_kwargs = {"vpt_steps": 500, "n_RMSE": 500, "w2_steps": int(1e5), "vpt_epsilon": 0.5,\
                "Lyapunov_time": 1/2.27, "n_sample_w2": 20000}
    return bnf_kwargs, data_gen_kwargs, train_kwargs, eval_kwargs 


@ut.timer
def run_batch(root, D_r=256, n_exprs=5, type="auto"):
    for i in range(n_exprs):
        if type == "auto":
            run_single(*get_autonomous_params(root + f"_{i}", D_r=D_r))
        else:
            run_single(*get_nonautonomous_params(root + f"_{i}", D_r=D_r))
    