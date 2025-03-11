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
    np.save("{}/train.npy".format(train_kwargs["save_folder"]), data[:, :N]) 
    # np.save("{}/test.npy".format(train_kwargs["save_folder"]), data[:, N:]) 
    convert.arr2csv(data[:, :N], index=0, I=train_kwargs["I"], t0=0, dt=data_gen_kwargs["dt"], filename=f"{train_kwargs['save_folder']}/train.csv")
    # del data
    std = data.std(axis=1)
    # train model
    model = bayesnf.BayesianNeuralFieldMAP(**bnf_kwargs)
    df_train = pd.read_csv(f"{train_kwargs['save_folder']}/train.csv")
    model.fit(df_train, seed=jax.random.PRNGKey(0), num_epochs=train_kwargs["epochs"])

    # save model
    ut.save(model, os.path.abspath(f"{train_kwargs['save_folder']}/bnf_model"), step=train_kwargs["epochs"])

    # generate data for VPT
    x = data[:, N:N+1]
    Y = ev.multistep_forecast(model, 0, x, train_kwargs["I"], eval_kwargs["vpt_steps"], data_gen_kwargs["dt"])
    Y = np.squeeze(Y, axis=-1).T
    np.save("{}/vpt_trajectory.npy".format(train_kwargs["save_folder"]), Y)

    # calculate VPT
    Y0 = data[:, N:N+eval_kwargs["vpt_steps"]]
    # print((Y-Y0).shape, std.shape)
    l = np.argmax((((Y - Y0) / std[:, None])**2).sum(axis=0) < eval_kwargs["vpt_epsilon"]**2)
    results["VPT"] = l * data_gen_kwargs["dt"] / eval_kwargs["Lyapunov_time"]

    # generate data for RMSE
    x = data[:, N:N+eval_kwargs["n_RMSE"]]
    Y = ev.parallel_forecast(model, 0, x, train_kwargs["I"])
    np.save("{}/rmse_trajectory.npy".format(train_kwargs["save_folder"]), Y)

    # calculate RMSE and MAE
    Y0 = data[:, N:N+eval_kwargs["n_RMSE"]]
    results["RMSE"] = np.sqrt(np.mean(((Y - Y0)**2).sum(axis=0)))
    results["MAE"] = np.mean(np.abs(Y - Y0).sum(axis=0))

    # generate data for Wasserstein
    x = data[:, N:N+1]
    Y = ev.multistep_forecast(model, 0, x, train_kwargs["I"], eval_kwargs["w2_steps"], data_gen_kwargs["dt"])
    Y = np.squeeze(Y, axis=-1).T
    np.save("{}/w2_trajectory.npy".format(train_kwargs["save_folder"]), Y)

    # calculate Wasserstein distance
    Y0 = data[:, N:]
    A = torch.tensor(Y.T[:eval_kwargs["n_sample_w2"]], dtype=torch.float32)
    B = torch.tensor(Y0.T[:eval_kwargs["n_sample_w2"]], dtype=torch.float32)
    results["W2"] = wasserstein.sinkhorn_div(A, B).item()

    # save results
    with open(f"{train_kwargs['save_folder']}/results.json", "w") as f:
        json.dump(results, f)

