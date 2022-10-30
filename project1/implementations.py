from typing import Tuple
import numpy as np


def mean_squarred_error_gd(y, tx, initial_w, max_iters: int, gamma: float) -> Tuple[np.ndarray, float]:
    """Linear regression using gradient descent

    Args:
        y: Array of shape (N,) containing the target values
        tx: Array of shape (N, D) containing the input data
        initial_w: Array of shape (D,) containing the initial weights
        max_iters: Number of iterations to run
        gamma: Step size

    Returns:
        w: Array of shape (D,) containing the final weights
        loss: Final loss
    """

    return mean_squarred_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=len(y))


def mean_squarred_error_sgd(y, tx, initial_w, max_iters: int, gamma: float, batch_size: int=1) -> Tuple[np.ndarray, float]:
    """Linear regression using stochastic gradient descent

    Args:
        y: Array of shape (N,) containing the target values
        tx: Array of shape (N, D) containing the input data
        initial_w: Array of shape (D,) containing the initial weights
        max_iters: Number of iterations to run
        gamma: Step size

    Returns:
        w: Array of shape (D,) containing the final weights
        loss: Final loss
    """

    weights = [initial_w]
    losses = []

    w = initial_w
    loss = -1
    for n_iter in range(max_iters):
        shuffled_indices = np.random.permutation(len(y))
        for batch_indices in np.array_split(shuffled_indices, len(y) // batch_size):
            y_batch = y[batch_indices]
            tx_batch = tx[batch_indices]

            # Computing the gradient
            e = y_batch - np.einsum('nd,d->n', tx_batch, w)
            gradient = -1/len(y_batch) * np.einsum('n,nd->d', e, tx_batch)
            # Update
            w = w - gamma * gradient

        loss = compute_mse(y, tx, w)
        print(f"MSE GD ({n_iter + 1}/{max_iters}): {loss=} {w=}")

        weights.append(w)
        losses.append(loss)
    return losses, weights
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations

    Args:
        y: Array of shape (N, 1) containing the target values
        tx: Array of shape (N, D) containing the input data

    Returns:
        (w, loss) Array of shape (D,) containing the ideal weights and the corresponding loss
    """
    N, D = tx.shape
    assert y.shape == (N, 1), f"y.shape = {y.shape}, expected {(N, 1)}"


    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    return w, compute_mse(y, tx, w)


def compute_mse(y, tx, w) -> float:
    """Computes the mean squared error loss at w.

    Args:
        y: numpy array of shape (N,1), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.

    Returns:
        mse: scalar corresponding to the mse with factor (1 / 2 n) in front of the sum

    >>> compute_mse(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), np.array([0.03947092, 0.00319628]))
    0.006417022764962313
    """
    N, D = tx.shape
    assert y.shape == (N,1), f"y.shape = {y.shape}, expected {(N,)}"
    assert w.shape == (D,1), f"w.shape = {w.shape}, expected {(D,)}"

    return np.sum((y - (tx @ w)) ** 2) / (2 * N)

def mse_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: Array of shape (N,) containing the target values
        tx: Array of shape (N, D) containing the input data
        w: Array of shape (D,) containing the weights

    Returns:
        Array of shape (D,) containing the gradient
    """

    e = y - np.einsum('nd,d->n', tx, w)
    return -1/len(y) * np.einsum('n,nd->d', e, tx)
    # return -1 / len(y) * np.dot(tx.T, y - np.dot(tx, w))


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    D = 1 if len(tx.shape) == 1 else tx.shape[1]
    N = len(y)
    a = np.array((tx.transpose() @ tx) + (2 * N * lambda_) * np.eye(D))
    b = np.array(tx.transpose() @ y)

    w = np.linalg.inv(a) @ b

    return w, compute_mse(y, tx, w)


# ------------------- #
# Logistic regression #
# ------------------- #

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1 / (1 + np.exp(-t))

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a hessian matrix of shape=(D, D)
    """

    N = len(y)
    sig = sigmoid(tx @ w)
    diag = np.zeros((N, N))
    np.fill_diagonal(diag,  sig * (1 - sig))

    return (1 / N) * ((tx.T @ diag) @ tx)

def calculate_loss(y, tx, w, penalty=0):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    assert y.shape[0]  == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    sig = sigmoid(tx @ w)
    left  = y * np.log(sig)
    right = (1-y) * np.log(1 - sig)

    return - np.mean(left + right) + penalty

def calculate_gradient(y, tx, w, lambda_=0):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """

    sig = sigmoid(tx @ w)

    ## last term is for adding a lambda_ * ||w||^2 penalty
    return (1 / len(y)) * (tx.T @ (sig - y)) + (2 * lambda_ * w)

## Uses gradient descent
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    losses = []

    w = initial_w

    # start the logistic regression with GD
    for iter in range(max_iters):
        # get loss and update w.
        loss = calculate_loss(y, tx, w)
        w = w - gamma * calculate_gradient(y, tx, w)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break


    losses.append(calculate_loss(y, tx, w))
    print("loss={l}".format(l=losses[-1]))

    return w, losses[-1]


# ------------------------------- #
# Regularized logistic regression #
# ------------------------------- #


def compute_penalty_term(lambda_, w):
    return lambda_ * (np.linalg.norm(w) ** 2)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    losses = []

    w = initial_w

    # start the logistic regression with GD
    for iter in range(max_iters):
        # get loss and update w.
        loss = calculate_loss(y, tx, w, penalty=compute_penalty_term(lambda_, w))
        w = w - gamma * calculate_gradient(y, tx, w, lambda_=lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break


    losses.append(calculate_loss(y, tx, w))
    print("loss={l}".format(l=losses[-1]))

    return w, losses[-1]