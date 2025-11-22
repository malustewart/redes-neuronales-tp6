import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filepath):
    df_orig = pd.read_csv('data/akc.csv')
    df_kept = df_orig[['height', 'weight']]
    df = df_kept.dropna()
    return df

def split_data(ratio):
    pass
    # return training, evaluation

def get_poly_regression(x, y, deg, xrange):
    return np.polynomial.polynomial.Polynomial.fit(x, y, deg, xrange)

def evaluate_poly_in_range(p, xrange):
    return p.linspace(200, xrange)

def params_to_str(params:dict, sep:str = " - "):
    return sep.join(f"{k}: {v}" for k,v in params.items())

def plot_height_weight_regression(height, weight, regressions:list, regression_order: list, params, filename=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.gca()

    ax.scatter(height, weight)
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,len(regressions))))   # no funciona
    for reg, order in zip(regressions, regression_order):
        ax.plot(reg[0], reg[1], label=f"Regresión orden {order}")
    fig.legend()
    if filename:
        pass

if __name__ == '__main__':
    filepath = 'data/akc.csv'
    data = load_data(filepath)

    polydegree = list(range(7))

    hrange = [np.min(data['height']), np.max(data['height'])]

    regs_poly = [get_poly_regression(data['height'], data['weight'], n, hrange) for n in polydegree]
    regs_to_plot = [evaluate_poly_in_range(reg_poly, hrange) for reg_poly in regs_poly]
    plot_height_weight_regression(data['height'], data['weight'], regs_to_plot, polydegree, {})

    w_est_all_n = np.array([[reg_poly(h) for h in data['height']] for reg_poly in regs_poly])

    mse  = [ np.mean((w_est - data['weight'])**2) for w_est in w_est_all_n]

    plt.figure()
    plt.semilogy(mse)

    plt.show()