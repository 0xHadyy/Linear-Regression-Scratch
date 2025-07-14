import numpy as np
from tabulate import tabulate
import random

t_table = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
    float("inf"): 1.960,  # Z‐value approximation
}

# utils
np.set_printoptions(precision=4, suppress=True, linewidth=100)


def is_invertible(A):
    return np.linalg.matrix_rank(A) == A.shape[0]


# Generating dummy data
def generate_dummy_data(n=10000, p=3, noise_std=1.0, seed=420):
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
    X_transpose = X.T  # Alernatively  np.transpose
    gram_matrix = X_transpose @ X
    if not is_invertible(gram_matrix):
        raise ValueError(f"The gram matrix with shape{gram_matrix.shape} is sningular")
    beta_hat = (
        np.linalg.inv(gram_matrix) @ X_transpose @ Y
    )  # beta matrix closed form OLS is (X^TX)^-1X^TY
    return beta_hat, gram_matrix  # estimated beta


def predict(X, beta_hat):
    # Computer the predicted response y_hat
    y_hat = X @ beta_hat
    return y_hat


# Sample standard deviation Which is the Estimate of the standard diviation sigma^2
def mean_squared_error(y, y_hat, p, n):
    # Compute the MSE
    n = y.shape[0]
    residuals = y - y_hat
    RSS = np.sum(residuals**2)

    if n - p <= 0:
        raise ValueError(f"Degrees of freedom (n-p) must be >0, got n={n},p={p}")
    MSE = RSS / (n - p)
    return MSE, RSS


def standard_error_beta(mse, gram_matrix):
    # Compute the variance of beta_hat
    if not is_invertible(gram_matrix):
        raise ValueError("The Gram matrix is signular")

    inv_gram_matrix = np.linalg.inv(gram_matrix)
    beta_hat_variance = (
        mse * inv_gram_matrix
    )  # The diagonal represent the variances, while off-diagonal are the covarianecs

    beta_hat_standard_error = np.sqrt(
        np.diag(beta_hat_variance)
    )  # taking the square root of only the variances

    return beta_hat_variance, beta_hat_standard_error


def confidence_interval_beta(beta_hat, standard_error, df, t_critical_95):
    # compute a 95% confidence interval for the coefficients beta , alpha = 0.05
    closest_df = min(t_critical_95, key=lambda x: abs(x - df))
    t_value = t_critical_95[closest_df]
    upper_bound = beta_hat + (t_value * standard_error)
    lower_bound = beta_hat - (t_value * standard_error)
    confidence_interval = np.stack((lower_bound, upper_bound))
    return confidence_interval


def f_statistic(RSS, y, p, n):
    # Compute the f-statistic, testing null hypothesis -> there is no relationship between the predictors and response
    y_mean = np.mean(y)
    TSS = np.sum((y - y_mean) ** 2)
    if p == 0:
        raise ValueError("invalid Features number")
    numerator = (TSS - RSS) / p
    denominator = RSS / (n - p - 1)
    f_test = numerator / denominator

    return f_test, TSS


def residual_standard_error(RSS, p, n):
    # Computer RSE -> Describe variability left unexplained by the regerssion model
    if n - p - 1 <= 0:
        raise ValueError("Degrees of freedom <= 0 Check n and p")
    RSE = np.sqrt(RSS / (n - p - 1))
    return RSE


def r_squared(RSS, TSS):
    # Compute R_squared which is the measure of fit and linear relationship between X <-> Y
    r_squared = 1 - (RSS / TSS)
    if not (0 <= r_squared <= 1):
        raise ValueError("Error happened while computing R_sqaured")
    return r_squared


def standard_error_reponse(X0, MSE, gram_matrix):
    # compute the variance and standard error in a new observed response Y0
    p = X0.shape[0]
    if gram_matrix.shape != (p, p):
        raise ValueError(f"Gram matrix shape {gram_matrix.shape}")
    if MSE < 0:
        raise ValueError("MSE is zero or negative")
    if not is_invertible(gram_matrix):
        raise ValueError("The gram fucntion is singular")
    inv_gram_matrix = np.linalg.inv(gram_matrix)
    X0_transpose = X0.T
    var_Y0 = MSE * (
        X0_transpose @ inv_gram_matrix @ X0
    )  # the formula var(Y_0)=MSE*X_0^T(Gram matrix)^-1X_0
    var_Y0_hat = MSE * (1 + X0_transpose @ inv_gram_matrix @ X0)
    se_Y0_hat = np.sqrt(var_Y0_hat)
    se_Y0 = np.sqrt(var_Y0)
    return var_Y0, se_Y0, var_Y0_hat, se_Y0_hat


def confidence_interval_mean_response(Y0, se_Y0, df, t_table):
    # compute the confidence interval for the true mean of a response
    closest_df = min(t_table, key=lambda x: abs(x - df))
    t_value = t_table[closest_df]
    upper_bound = Y0 + (t_value * se_Y0)
    lower_bound = Y0 - (t_value * se_Y0)
    response_mean_cl = np.stack((lower_bound, upper_bound))
    return response_mean_cl


def prediction_interval_response(Y0, se_Y0_hat, df, t_table):
    # compute the prediction interval for an individual response
    closest_df = min(t_table, key=lambda x: abs(x - df))
    t_value = t_table[closest_df]
    upper_bound = Y0 + (t_value * se_Y0_hat)
    lower_bound = Y0 - (t_value * se_Y0_hat)
    observation_pl = np.stack((lower_bound, upper_bound))
    return observation_pl


def gradient_descent(y, X, alpha, batch_size, n, p, n_iters):
    y = y.reshape(-1, 1)
    beta_hat = np.zeros((p, 1))
    for i in range(n_iters):
        idx = np.random.choice(n, size=batch_size, replace=False)
        Xb = X[idx]
        yb = y[idx].reshape(-1, 1)
        y_hat = Xb @ beta_hat
        RSS = y_hat - yb
        gradient_mse = (Xb.T @ RSS) / batch_size
        beta_hat_new = beta_hat - alpha * gradient_mse

        if np.linalg.norm(beta_hat_new - beta_hat) < 1e-6:
            break
        beta_hat = beta_hat_new
    return beta_hat


X, y, beta_true, noise = generate_dummy_data()
n = len(y)
p = X.shape[1]

# Estimating The Coefficient Beta with both OLS and Gradient Descent
beta_hat, gram_matrix = ols_estimate(X, y)
# Hyperparameters
alpha = 0.1
batch_size = 500
number_iterations = 5000
beta_hat_descent = gradient_descent(y, X, alpha, batch_size, n, p, number_iterations)

# Predicted response Y using both estimation methods
y_hat = predict(X, beta_hat)
y_hat_descent = predict(X, beta_hat_descent)


MSE, RSS = mean_squared_error(y, y_hat, p, n)

beta_hat_var, beta_hat_se = standard_error_beta(MSE, gram_matrix)
confidence_interval_coe = confidence_interval_beta(beta_hat, beta_hat_se, 3, t_table)

f_score, TSS = f_statistic(RSS, y, p, n)

RSE = residual_standard_error(RSS, p, n)
r2 = r_squared(RSS, TSS)

# Randomly picking value for i
i = random.randint(1, n)

X0 = X[i]  # sample feature
y0 = y_hat[i]  # predicted value for the sample feature

var_Y0, se_Y0, var_Y0_hat, se_Y0_hat = standard_error_reponse(X0, MSE, gram_matrix)
mean_response_cl = confidence_interval_mean_response(y0, se_Y0, 3, t_table)
new_obs_pl = prediction_interval_response(y0, se_Y0_hat, 3, t_table)

# Displaying the  Estimates, Standard Error, Confidence intervals for Coeficcients beta
labels = [f"β{j}" for j in range(len(beta_true))]
rows = []
for betas, estimate, std, (low, high) in zip(
    labels, beta_hat.flatten(), beta_hat_se, confidence_interval_coe.T
):
    rows.append((betas, f"{estimate:.4f}", f"{std:.4f}", f"[{low:.4f},{high:.4f}]"))
print(
    tabulate(
        rows,
        headers=["Coefficient", "Estimates", "Std.Err", "95% CI"],
        tablefmt="github",
    )
)

rows = []
rows.append(
    (
        f"{r2:.4f}",
        f"{RSE:.4f}",
        f"[{mean_response_cl[0]:.4f},{mean_response_cl[1]}:.4f]",
        f"[{new_obs_pl[0]:.4f},{new_obs_pl[1]:.4f}]",
    )
)
print(
    "----------------------------------------------------------------------------------------------"
)
print(
    tabulate(
        rows,
        headers=["R-Squared", "RSE", "95% CI mean", "95% PI new obs"],
        tablefmt="github",
    )
)
print("this is response y", y)
print("True Beta (Coefficients)", beta_true)
print("Estimated beta with OLS (Coefficients)  is :\n", beta_hat)
print("Estimated beta with Gradient Descent (Coefficients) is :\n", beta_hat_descent)
print("y_hat:", y_hat.shape)
print("Residual squared sum =", round(RSS, 4))
print("Mean squared error= ", round(MSE, 4))
print("The Gram matrix: \n", gram_matrix)
print("beta hat variance :\n", beta_hat_var)
