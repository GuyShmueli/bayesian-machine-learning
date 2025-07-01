import numpy as np
from matplotlib import pyplot as plt
from typing import Callable


def polynomial_basis_functions(degree: int) -> Callable:
    """
    Create a function that calculates the polynomial basis functions up to (and including) a degree
    :param degree: the maximal degree of the polynomial basis functions
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             polynomial basis functions, a numpy array of shape [N, degree+1]
    """

    def pbf(x: np.ndarray):
        N = x.shape[0]

        # Initialize the design matrix with ones for the bias term
        design_matrix = np.ones((N, degree + 1))

        for deg in range(1, degree + 1):
            design_matrix[:, deg] = (x / deg) ** deg
        return design_matrix

    return pbf


def fourier_basis_functions(num_freqs: int) -> Callable:
    """
    Create a function that calculates the fourier basis functions up to a certain frequency
    :param num_freqs: the number of frequencies to use
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             Fourier basis functions, a numpy array of shape [N, 2*num_freqs + 1]
    """

    def fbf(x: np.ndarray):
        N = x.shape[0]

        PI = np.pi

        HOUR_CYCLE = 24  # hours

        # Initialize the design matrix with ones for the bias term
        design_matrix = np.ones((N, 2 * num_freqs + 1))

        for freq in range(1, num_freqs + 1):
            design_matrix[:, freq] = np.cos(2 * PI * freq * x / HOUR_CYCLE)
            design_matrix[:, freq + num_freqs] = np.sin(2 * PI * freq * x / HOUR_CYCLE)

        return design_matrix

    return fbf


def spline_basis_functions(knots: np.ndarray) -> Callable:
    """
    Create a function that calculates the cubic regression spline basis functions around a set of knots
    :param knots: an array of knots that should be used by the spline
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             cubic regression spline basis functions, a numpy array of shape [N, len(knots)+4]
    """

    def csbf(x: np.ndarray):

        N = x.shape[0]

        # Initialize the design matrix with the first column as ones for the bias term
        design_matrix = np.ones((N, len(knots) + 4))

        # Add cubic polynomial terms (x^1, x^2, x^3)
        for i in range(1, 4):
            design_matrix[:, i] = x ** i

        # Add spline basis functions for each knot, starting from column index 4
        for idx, knot in enumerate(knots):
            design_matrix[:, idx + 4] = ((x - knot) ** 3).clip(min=0)  # to ensure t-a->0 when t-a<0

        return design_matrix

    return csbf


def learn_prior(hours: np.ndarray, temps: np.ndarray, basis_func: Callable) -> tuple:
    """
    Learn a Gaussian prior using historic data
    :param hours: an array of vectors to be used as the 'X' data
    :param temps: a matrix of average daily temperatures in November, as loaded from 'jerus_daytemps.npy', with shape
                  [# years, # hours]
    :param basis_func: a function that returns the design matrix of the basis functions to be used
    :return: the mean and covariance of the learned covariance - the mean is an array with length dim while the
             covariance is a matrix with shape [dim, dim], where dim is the number of basis functions used
    """
    thetas = []
    # iterate over all past years
    for i, t in enumerate(temps):
        ln = LinearRegression(basis_func).fit(hours, t)
        thetas.append(ln.weights)  # append learned parameters here

    thetas = np.array(thetas)

    # take mean over parameters learned each year for the mean of the prior
    mu = np.mean(thetas, axis=0)
    # calculate empirical covariance over parameters learned each year for the covariance of the prior
    cov = (thetas - mu[None, :]).T @ (thetas - mu[None, :]) / thetas.shape[0]
    return mu, cov


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig: float, basis_functions: Callable):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sig:                 the signal noise to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.theta_mean = theta_mean
        self.theta_cov = theta_cov
        self.sig = sig
        self.basis_functions = basis_functions

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        H_design = self.basis_functions(X)
        H_design_transpose = np.transpose((H_design))
        inverse_cov = np.linalg.pinv(self.theta_cov)
        H_transpose_H_matmul = np.matmul(H_design_transpose, H_design)
        squared_inv_sig = (1 / np.square(self.sig))

        posterior_cov = np.linalg.pinv(inverse_cov + squared_inv_sig * H_transpose_H_matmul)

        posterior_mean = np.dot(posterior_cov,
                                squared_inv_sig * np.dot(H_design_transpose, y) + np.dot(inverse_cov, self.theta_mean))

        self.theta_mean = posterior_mean
        self.theta_cov = posterior_cov

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        H_design = self.basis_functions(X)
        predictions = np.dot(H_design, self.theta_mean)
        return predictions

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        H_design = self.basis_functions(X)

        H_cov_H_transpose = H_design @ self.theta_cov @ H_design.T

        var = np.diag(H_cov_H_transpose) + self.sig ** 2

        std = np.sqrt(var)

        return std

    def posterior_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model and sampling from the posterior
        :param X: the samples to predict
        :return: the predictions for X
        """
        H_design = self.basis_functions(X)

        # Sample a parameter set from the posterior distribution of parameters
        theta_sample = np.random.multivariate_normal(self.theta_mean, self.theta_cov)

        predictions = H_design @ theta_sample

        return predictions


class LinearRegression:

    def __init__(self, basis_functions: Callable):
        """
        Initializes a linear regression model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.basis_functions = basis_functions
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the model to the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        H_design = self.basis_functions(X)
        H_design_transpose = np.transpose((H_design))
        H_transpose_H_matmul = np.matmul(H_design_transpose, H_design)
        inverse_H_transpose_H_matmul = np.linalg.pinv(H_transpose_H_matmul)
        inverse_H_transpose_H_H_transpose_matmul = np.matmul(inverse_H_transpose_H_matmul, H_design_transpose)
        self.weights = np.dot(inverse_H_transpose_H_H_transpose_matmul, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        H_design = self.basis_functions(X)
        predictions = np.dot(H_design, self.weights)
        return predictions

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and return the predicted values for X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)


def plot_prior(x: np.ndarray, mu: np.ndarray, cov: np.ndarray, sig: float, basis_func: Callable, title: str):
    """
    Plot smoothly the model by taking the parameters of the prior (by learn_prior function).
    :param x: long-ranged array to smoothly plot the function
    :param mu: the mean of the prior
    :param cov: the covariance matrix of the prior
    :param sig: the signal noise to use when fitting the model
    :param basis_func: a function that receives data points as inputs and returns a design matrix
    :param title: the title of the graph
    :return: a graph of temperature vs hour, with the model fit only by the prior distribution
    """

    plt.figure()

    H_design = basis_func(x)

    mean = H_design @ mu

    H_cov_H_transpose = H_design @ cov @ H_design.T

    std = np.sqrt(np.diag(H_cov_H_transpose) + sig ** 2)

    plt.fill_between(x, mean - std, mean + std, alpha=.5, label='confidence interval')

    plt.plot(x, mean, 'k', lw=3, label='prior mean')

    for i in range(5):
        random_no_mean = H_design @ np.random.multivariate_normal(mu, cov)
        plt.plot(x, random_no_mean)

    plt.legend()

    plt.xlabel('Hour')

    plt.ylabel('Temperature')

    plt.title(title)


def plot_mmse(x: np.ndarray, train: np.ndarray, train_hours: np.ndarray, test: np.ndarray, test_hours: np.ndarray,
               mu: np.ndarray, cov: np.ndarray, sig: float, basis_func: Callable, title: str):
    """
    Plot smoothly the model by taking the parameters of the posterior,
    by doing Bayesian Linear Regression and taking theta MMSE (which is the
    best estimator of theta in BMSE term, which is the mean of theta given the data.
    :param x: long-ranged array to smoothly plot the function
    :param train: the output variable of the data used for training
    :param train_hours: the input variable data used for training
    :param test: the output variable of the data used for testing
    :param test_hours: the input variable data used for testing
    :param mu: the mean of the prior
    :param cov: the covariance matrix of the prior
    :param sig: the signal noise to use when fitting the model
    :param basis_func: a function that receives data points as inputs and returns a design matrix
    :param title: the title of the graph
    :return: a graph of temperature vs hour, with the model fit by the posterior distribution
    """

    plt.figure()

    blm = BayesianLinearRegression(mu, cov, sig, basis_func)

    blm.fit(train_hours, train)

    predictions = blm.predict(x)

    mse = np.mean((test - blm.predict(test_hours)) ** 2)

    std = blm.predict_std(x)

    plt.fill_between(x, predictions - std, predictions + std, alpha=.5, label='confidence interval')

    plt.plot(x, predictions, 'k', lw=3, label='MMSE')

    for i in range(5):
        random_no_mean = blm.posterior_sample(x)
        plt.plot(x, random_no_mean)

    plt.scatter(train_hours, train, label='train')
    plt.scatter(test_hours, test, label='test')
    plt.xlabel('Hours')
    plt.ylabel('Temperatures (C)')
    plt.title(f'{title}, error={mse:.2f}')
    plt.legend()
    plt.show()


def main():
    # load the data for November 16 2024
    nov16 = np.load('nov162024.npy')
    nov16_hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    train_hours = nov16_hours[:len(nov16) // 2]
    test = nov16[len(nov16) // 2:]
    test_hours = nov16_hours[len(nov16) // 2:]

    # setup the model parameters
    degrees = [3, 7]

    # ----------------------------------------- Classical Linear Regression

    for d in degrees:
        ln = LinearRegression(polynomial_basis_functions(d)).fit(train_hours, train)

        # print average squared error performance
        print(f'Average squared error with LR and d={d} is {np.mean((test - ln.predict(test_hours)) ** 2):.2f}')

        # plot graphs for linear regression part
        predictions = ln.predict(test_hours)
        plt.figure()
        plt.scatter(train_hours, train, label='Train')
        plt.scatter(test_hours, test, label='Test')

        plt.plot(test_hours, predictions, label=f'Preicted: d={d}')

        plt.title(f'Linear Regression with degree {d}, MSE {round(np.mean((test - predictions) ** 2), 2)}')
        plt.xlabel('Hour')
        plt.ylabel('Temperature')
        plt.legend()

    plt.tight_layout()
    plt.show()

    # ----------------------------------------- Bayesian Linear Regression

    # load the historic data
    temps = np.load('jerus_daytemps.npy').astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    # setup the model parameters
    sigma = np.sqrt(0.25)
    degrees = [3, 7]  # polynomial basis functions degrees

    # frequencies for Fourier basis
    freqs = [1, 2, 3]

    # sets of knots K_1, K_2 and K_3 for the regression splines
    knots = [np.array([12]),
             np.array([8, 16]),
             np.array([6, 12, 18])]

    # ---------------------- polynomial basis functions
    for deg in degrees:
        pbf = polynomial_basis_functions(deg)
        mu, cov = learn_prior(hours, temps, pbf)

        # plot prior graphs

        # plot mean with confidence intervals
        title = f'Polynomial with deg = {deg}'
        plot_prior(x, mu, cov, sigma, pbf, title)

        # plot posterior graphs
        plot_mmse(x, train, train_hours, test, test_hours, mu, cov, sigma, pbf, title)

    # ---------------------- Fourier basis functions
    for ind, K in enumerate(freqs):
        rbf = fourier_basis_functions(K)
        mu, cov = learn_prior(hours, temps, rbf)

        # blr = BayesianLinearRegression(mu, cov, sigma, rbf)

        # plot prior graphs
        title = f'Fourier with freq = {K}'
        plot_prior(x, mu, cov, sigma, rbf, title)

        # plot posterior graphs
        plot_mmse(x, train, train_hours, test, test_hours, mu, cov, sigma, rbf, title)

    # ---------------------- cubic regression splines
    for ind, k in enumerate(knots):
        spline = spline_basis_functions(k)
        mu, cov = learn_prior(hours, temps, spline)

        # plot prior graphs
        title = f'Cubic with knot K_{ind + 1}'
        plot_prior(x, mu, cov, sigma, spline, title)

        # plot posterior graphs
        plot_mmse(x, train, train_hours, test, test_hours, mu, cov, sigma, spline, title)


if __name__ == '__main__':
    main()
