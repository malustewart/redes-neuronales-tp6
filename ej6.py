import numpy as np
from numpy.polynomial.polynomial import Polynomial as Polynomial
import pandas as pd
import matplotlib.pyplot as plt
import random

def load_data(filepath='data/akc.csv'):
    df_orig = pd.read_csv(filepath)
    df_kept = df_orig[['height', 'weight']]
    df = df_kept.dropna()
    return df

def split_data(data, ratio):
    N = len(data.index)
    idx = random.sample(range(N), int(N*ratio))
    labels = data.index[idx]
    training = data.loc[labels]
    evaluation = data.drop(labels)
    return training, evaluation

def get_poly_regression(x, y, deg, xrange, λ=0):
    X = np.array([x.to_numpy()**i for i in range(deg + 1)])
    Y = y.to_numpy()
    w = np.linalg.inv(X@X.T + λ * np.eye(len(X))) @ X @ Y.T
    return w

def evaluate_poly_in_range(p, xrange):
    return p.linspace(200, xrange)

def params_to_str(params:dict, sep:str = " - "):
    return sep.join(f"{k}: {v}" for k,v in params.items())

def plot_height_weight_regression(height, weight, regressions:list, regression_order: list, title="Regresión altura-peso", ylabel="Altura", xlabel="Peso", params={}, filename=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.gca()
    ax.scatter(height, weight)
    for reg, order in zip(regressions, regression_order):
        ax.plot(reg[0], reg[1], label=f"Regresión orden {order}")
    fig.legend()
    ax.set_title(title + " - " + params_to_str(params))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if filename:
        fig.savefig(filename)

def plot_mse(x, mses, labels=None, title="MSE", ylabel="mse", xlabel="Orden regresión", params={}, filename=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.gca()
    for mse, label in zip(mses, labels):
        ax.plot(x, mse, label=label)
    fig.legend()
    ax.set_title(title + " - " + params_to_str(params))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if filename:
        fig.savefig(filename)

def calc_mse(A, B):
    return np.mean((A - B)**2)

def calc_fit_and_mse(data, training_ratio, polydegrees):
    training_ratio = 0.5
    training_data, evaluation_data = split_data(data, training_ratio)

    hrange = [np.min(training_data['height']), np.max(training_data['height'])]
    regs_poly_coef = [get_poly_regression(training_data['height'], training_data['weight'], n, hrange) for n in polydegrees]
    regs_poly = [Polynomial(coef) for coef in regs_poly_coef]

    w_est_all_n_training = np.array([[reg_poly(h) for h in training_data['height']] for reg_poly in regs_poly])
    w_est_all_n_evaluation = np.array([[reg_poly(h) for h in evaluation_data['height']] for reg_poly in regs_poly])
    mse_training = [calc_mse(w_est, training_data['weight']) for w_est in w_est_all_n_training]
    mse_evaluation = [calc_mse(w_est, evaluation_data['weight']) for w_est in w_est_all_n_evaluation]
    return regs_poly_coef, mse_training, mse_evaluation

if __name__ == '__main__':
    random.seed(123456)

    filepath = 'data/akc.csv'
    data = load_data(filepath)

    polydegrees = list(range(7))

    hrange = [np.min(data['height']), np.max(data['height'])]

    # 100% de datos para entrenamiento

    training_ratio=1
    regs_poly_coef, mse_training, mse_evaluation = calc_fit_and_mse(data=data, training_ratio=training_ratio, polydegrees=polydegrees)
    hspace = np.linspace(hrange[0], hrange[1], 100)
    w_est_to_plot = [[hspace, Polynomial(coef)(hspace)] for coef in regs_poly_coef]
    plot_height_weight_regression(data['height'], data['weight'], w_est_to_plot, polydegrees)
    plot_mse(polydegrees, [mse_training], labels=["Entrenamiento"], params={r"% de datos para entrenamiento":100*training_ratio})

    # 80% de datos para entrenamiento, 20% para evaluacion
       
    training_ratio=0.8
    regs_poly_coef, mse_training, mse_evaluation = calc_fit_and_mse(data=data, training_ratio=training_ratio, polydegrees=polydegrees)
    hspace = np.linspace(hrange[0], hrange[1], 100)
    w_est_to_plot = [[hspace, Polynomial(coef)(hspace)] for coef in regs_poly_coef]
    plot_height_weight_regression(data['height'], data['weight'], w_est_to_plot, polydegrees)
    plot_mse(polydegrees, [mse_training, mse_evaluation], labels=["Entrenamiento", "Evaluación"], params={r"% de datos para entrenamiento":100*training_ratio})

    plt.show()