import numpy as np
import numpy.linalg
from matplotlib import pyplot as plt
from utils import BayesianLinearRegression, polynomial_basis_functions, load_prior

def log_evidence(model: BayesianLinearRegression, X, y):
    """
    Calculate the log-evidence of some data under a given Bayesian linear regression model
    :param model: the BLR model whose evidence should be calculated
    :param X: the observed x values
    :param y: the observed responses (y values)
    :return: the log-evidence of the model on the observed data
    """
    # extract the variables of the prior distribution
    mu = model.mu
    sig = model.cov
    n = model.sig

    # extract the variables of the posterior distribution
    model.fit(X, y)
    map = model.fit_mu
    map_cov = model.fit_cov

    # calculate the log-evidence
    first_term_log_evid = np.log(numpy.linalg.det(map_cov) / numpy.linalg.det(sig))
    second_term_log_evid = ((map - mu).T @ np.linalg.pinv(sig) @ (map - mu) + (1/n) * (y - model.predict(X)).T @
                            (y - model.predict(X)) + len(y) * np.log(n))
    third_term_log_evid = model.h(X).shape[1] * np.log(2 * np.pi)

    log_evid = 0.5 * first_term_log_evid - 0.5 * second_term_log_evid - 0.5 * third_term_log_evid
    return log_evid


def main():
    # set up the response functions
    f1 = lambda x: x ** 2 - 1
    f2 = lambda x: (-x ** 2 + 10 * x ** 3 + 50 * np.sin(x / 6) + 10) / 100
    f3 = lambda x: (.5 * x ** 6 - .75 * x ** 4 + 2.75 * x ** 2) / 50
    f4 = lambda x: 5 / (1 + np.exp(-4 * x)) - (x - 2 > 0) * x
    f5 = lambda x: 1 * (np.cos(x * 4) + 4 * np.abs(x - 2))
    functions = [f1, f2, f3, f4, f5]

    noise_var = .25
    x = np.linspace(-3, 3, 500)

    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    alpha = 1

    # go over each response function and polynomial basis function
    for i, f in enumerate(functions):
        y = f(x) + np.sqrt(noise_var) * np.random.randn(len(x))

        ev_lst = []
        for j, d in enumerate(degrees):
            # set up model parameters
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha

            # calculate evidence
            ev = log_evidence(BayesianLinearRegression(mean, cov, noise_var, pbf), x, y)
            ev_lst.append(ev)
        best_model = np.asarray(ev_lst).argmax() + 2
        worst_model = np.asarray(ev_lst).argmin() + 2

        # plot evidence versus degree
        plt.figure()
        plt.title(f'log-evidence vs polynomial degree \n function {i + 1} \n'
                  f'best model: d = {best_model}, worst model: d = {worst_model}')
        plt.xlabel('degree')
        plt.ylabel('log-evidence')
        plt.plot(degrees, ev_lst)
        plt.show()

        # plot predicted fit
        plt.figure(figsize=(10,6))
        models_names = "worst model", "best model"
        models = (worst_model, best_model)
        for index, k in enumerate(models):
            pbf = polynomial_basis_functions(k)
            mean, cov = np.zeros(k + 1), np.eye(k + 1) * alpha
            blm = BayesianLinearRegression(mean, cov, noise_var, pbf)
            blm.fit(x, y)
            predictions = blm.predict(x)
            std = blm.predict_std(x)
            plt.fill_between(x, predictions - std, predictions + std, alpha=.4, label=f'{models_names[index]} confidence interval')
            plt.plot(x, predictions, label=models_names[index])

        plt.scatter(x, y, alpha=0.3, label='data')
        plt.title(f'Function {i + 1} best and worst fittings')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    # load relevant data
    nov16 = np.load('nov162024.npy')
    hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    hours_train = hours[:len(nov16) // 2]

    # load prior parameters and set up basis functions
    mu, cov = load_prior()
    pbf = polynomial_basis_functions(7)

    noise_vars = np.linspace(.05, 2, 100)
    evs = np.zeros(noise_vars.shape)
    for i, n in enumerate(noise_vars):
        # calculate the evidence
        mdl = BayesianLinearRegression(mu, cov, n, pbf)
        ev = log_evidence(mdl, hours_train, train)
        evs[i] = ev

    # plot log-evidence versus amount of sample noise
    plt.plot(noise_vars, evs)
    plt.title(f'log-evidence vs noise variances \n'
              f'noise with maximal evidence is {round(noise_vars[evs.argmax()], 2)}')
    plt.xlabel('noise variances')
    plt.ylabel('log-evidence')
    plt.show()

if __name__ == '__main__':
    main()

