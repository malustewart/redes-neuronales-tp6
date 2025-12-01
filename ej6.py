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

def get_poly_regression(x, y, deg, λ=0):
    X = np.array([x.to_numpy()**i for i in range(deg + 1)])
    Y = y.to_numpy()
    w = np.linalg.inv(X@X.T + λ * np.eye(len(X))) @ X @ Y.T
    return w
 
def evaluate_poly_in_range(p, xrange):
    return p.linspace(200, xrange)

def params_to_str(params:dict, sep:str = " - "):
    return sep.join(f"{k}: {v}" for k,v in params.items())

def plot_height_weight_regression(height, weight, regressions:list, regression_order: list, title="Regresión altura-peso", ylabel="Altura", xlabel="Peso", plotlabel="Regresión orden", params={}, filename=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.gca()
    ax.scatter(height, weight)
    for reg, order in zip(regressions, regression_order):
        ax.plot(reg[0], reg[1], label=f"{plotlabel} {order}")
    fig.legend()
    ax.set_title(title + " - " + params_to_str(params))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid()

    if filename:
        fig.savefig(filename)
        plt.close(fig)

def plot_mse(x, mses, labels=None, title="MSE", ylabel="mse", xlabel="Orden regresión", params={}, filename=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.gca()
    for mse, label in zip(mses, labels):
        ax.plot(x, mse, label=label)
    fig.legend()
    ax.set_title(title + " - " + params_to_str(params))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid()

    if filename:
        fig.savefig(filename)
        plt.close(fig)

def calc_mse(A, B):
    return np.mean((A - B)**2)

def evaluate_poly(coefs, X):
    # coefs: polynomial coefficients (lowest order first)
    return np.array([np.polyval(coefs[::-1], x) for x in X])

def calc_fit_and_mse(data, training_ratio, polydegrees, lambdas=[0]):
    training_data, evaluation_data = split_data(data, training_ratio)
    N_deg = len(polydegrees)
    N_lambda = len(lambdas)

    coefs = np.array([np.array([get_poly_regression(training_data['height'], training_data['weight'], n, lamda) for n in polydegrees], dtype=object) for lamda in lambdas])
    mse_training = np.array([np.array([calc_mse(training_data['weight'], evaluate_poly(coefs[i][j], training_data['height'])) for j in range(N_deg)]) for i in range(N_lambda)])
    mse_evaluation = np.array([np.array([calc_mse(evaluation_data['weight'], evaluate_poly(coefs[i][j], evaluation_data['height'])) for j in range(N_deg)]) for i in range(N_lambda)])
    return coefs, mse_training, mse_evaluation

def plot_mse_heatmap(mse_training, mse_evaluation, lambdas, polydeg, params, filename):
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    vmax = max(mse_training.max(), mse_evaluation.max())

    # First heatmap: Training MSE
    im1 = axes[0].imshow(mse_training, aspect='auto', origin='lower', vmin=0, vmax=vmax, cmap='gray_r')
    axes[0].set_title("MSE (Entrenamiento)")
    axes[0].set_ylabel("λ")

    if lambdas is not None:
        axes[0].set_yticks(np.arange(len(lambdas)))
        axes[0].set_yticklabels(lambdas)

    if polydeg is not None:
        axes[0].set_xticks(np.arange(len(polydeg)))
        axes[0].set_xticklabels(polydeg)

    fig.colorbar(im1, ax=axes[0], label="MSE")

    # Second heatmap: Evaluation MSE
    im2 = axes[1].imshow(mse_evaluation, aspect='auto', origin='lower', vmin=0, vmax=vmax, cmap='gray_r')
    axes[1].set_title("MSE (Evaluación)")
    axes[1].set_xlabel("Orden de regresión")
    axes[1].set_ylabel("λ")

    if lambdas is not None:
        axes[1].set_yticks(np.arange(len(lambdas)))
        axes[1].set_yticklabels(lambdas)

    if polydeg is not None:
        axes[1].set_xticks(np.arange(len(polydeg)))
        axes[1].set_xticklabels(polydeg)

    fig.colorbar(im2, ax=axes[1], label="MSE")
    fig.suptitle(params_to_str(params))

    plt.tight_layout()

    if filename:
        fig.savefig(filename)
        plt.close(fig)

def process_and_plot(data, training_ratio, lambdas, polydegrees, training=True, evaluation=True):
    hrange = [np.min(data['height']), np.max(data['height'])]
    coefs, mse_training, mse_evaluation = calc_fit_and_mse(data=data, training_ratio=training_ratio, polydegrees=polydegrees, lambdas=lambdas)

    params = {
            r"% de datos para entrenamiento":f"{100*training_ratio}%",
        }
    plot_mse_heatmap(mse_training, mse_evaluation, lambdas, polydegrees, params, filename=f"figs/mse_tr_{training_ratio}.png")

    mses=[]
    labels=[]
    if training:
        mses.append(mse_training)
        labels.append("Entrenamiento")
    if evaluation:
        mses.append(mse_evaluation)
        labels.append("Evaluación")

    hspace = np.linspace(hrange[0], hrange[1], 100)
    for i, lamda in enumerate(lambdas):
        w_est_to_plot = [[hspace, Polynomial(coef)(hspace)] for coef in coefs[i]]
        params = {
            r"% de datos para entrenamiento":f"{100*training_ratio}%",
            "lambda": lamda
        }
        plot_height_weight_regression(data['height'], data['weight'], w_est_to_plot, polydegrees, params=params, filename=f"figs/fit_tr_{training_ratio}_lambda_{lamda}.png")
        plot_mse(polydegrees, mses=[mse[i] for mse in mses], labels=labels, params=params, filename=f"figs/mse_tr_{training_ratio}_lambda_{lamda}.png")
    
    for i, deg in enumerate(polydegrees):
        w_est_to_plot = [[hspace, Polynomial(coef)(hspace)] for coef in coefs[:,i]]
        params = {
            r"% de datos para entrenamiento":f"{100*training_ratio}%",
            "Orden polinomio": deg
        }
        plot_height_weight_regression(data['height'], data['weight'], w_est_to_plot, lambdas, plotlabel="λ:", params=params, filename=f"figs/fit_tr_{training_ratio}_polydeg_{deg}.png")
        plot_mse(lambdas, mses=[mse[:,i] for mse in mses], labels=labels, xlabel="λ", params=params, filename=f"figs/mse_tr_{training_ratio}_polydeg_{deg}.png")

if __name__ == '__main__':
    random.seed(123456)

    filepath = 'data/akc.csv'
    data = load_data(filepath)

    # 100% de datos para entrenamiento
    training_ratio=1
    lambdas = [i*0.5 for i in range(10)]
    polydegrees = list(range(5))
    process_and_plot(data, training_ratio, lambdas, polydegrees, evaluation=False)

    # 80% de datos para entrenamiento, 20% para evaluacion
    training_ratio=0.8
    process_and_plot(data, training_ratio, lambdas, polydegrees)
