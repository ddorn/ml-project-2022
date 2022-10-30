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

        loss = mse_loss(y, tx, w)
        print(f"MSE GD ({n_iter + 1}/{max_iters}): {loss=} {w=}")

        weights.append(w)
        losses.append(loss)
    return losses, weights
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations

    Args:
        y: Array of shape (N,) containing the target values
        tx: Array of shape (N, D) containing the input data

    Returns:
        (w, loss) Array of shape (D,) containing the ideal weights and the corresponding loss
    """

    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    return w, mse_loss(y, tx, w)


def mse_loss(y, tx, w) -> float:
    """Computes the mean squared error loss at w."""
    return np.sum((y - np.dot(tx, w)) ** 2) / (2 * len(y))

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
