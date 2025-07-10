import numpy as np


# Generating dummy data
def generate_dummy_data(n=10, p=3, noise_std=1.0, seed=420):
    np.random.seed(seed)

    X_raw = np.random.randn(
        n, p - 1
    )  # generate a matrix with "n" lines and p-1 columns (Without intercept)

    intercept = np.ones((n, 1))  # generate an intercept vector

    X = np.hstack([intercept, X_raw])

    beta_true = np.array(
        [3] + [2] * (p - 1)
    )  # true value of beta intercept = 3 , rest of the coefficients =2
    noise = np.random.randn(n) * noise_std  # generating the irreducible error
    y = X @ beta_true + noise  # @ is matrix multiplication operator

    return X, y, beta_true, noise


# Ordinary Least Squares to estimate the closed-form of beta
def ols_estimate(X, Y):
    # X is also called design matrix
    X_transpose = np.transpose(X)  # Alernatively X.T
    gram_matrix = X_transpose @ X
    beta_hat = (
        np.linalg.inv(gram_matrix) @ X_transpose @ Y
    )  # beta matrix closed form OLS is (X^TX)^-1X^TY
    return beta_hat, gram_matrix  # estimated beta


# Sample standard deviation Which is the Estimate of the standard deviaion sigma^2
def mean_squared_error(y, y_hat, p):
    # Compute the MSE
    n = len(y)
    residual_sum_squares = 0.0
    for i in range(n):
        residual_sum_squares += np.square(y[i] - y_hat[i])
    mse = residual_sum_squares / (n - p)  # MSE = 1/n sum(y-y_hat)^2
    return mse, residual_sum_squares


def standard_error_beta(mse, gram_matrix):
    # Compute the variance of beta_hat
    inv_gram_matrix = np.linalg.inv(gram_matrix)
    beta_hat_variance = (
        mse * inv_gram_matrix
    )  # The diagonal represent the variances, while off-diagonal are the covarianecs
    beta_hat_standard_error = np.sqrt(
        np.diag(beta_hat_variance)
    )  # taking the square root of only the variances
    return beta_hat_variance, beta_hat_standard_error


X, y, beta_true, noise = generate_dummy_data()

beta_hat, gram_matrix = ols_estimate(X, y)
y_hat = X @ beta_hat
mse, rss = mean_squared_error(y, y_hat, p=X.shape[1])
beta_hat_var, beta_hat_se = standard_error_beta(mse, gram_matrix)


print("X shape:", X.shape)
print("y shape:", y.shape)
print("True Beta (Coefficients)", beta_true)
print("Estimated beta (Coefficients) are :", beta_hat)
print("y_hat:", y_hat.shape)
print("Residual squared sum =", round(rss, 4))
print("Mean squared error= ", round(mse, 4))
print("The Gram matrix: \n", gram_matrix)
print("beta hat variance :\n", beta_hat_var)
print("beta hat standard error :\n", beta_hat_se)
