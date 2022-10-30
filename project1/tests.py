import numpy as np
from implementations import *

RTOL=1e-4
ATOL=1e-8

MAX_ITERS = 2
GAMMA = 0.1

def initial_w_testing():
    return np.array([[0.5], [1.0]])

def y_testing():
    return np.array([[0.1], [0.3], [0.5]])

def tx_testing():
    return np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])

def test_least_squares(y, tx):
    w, loss = least_squares(y, tx)

    expected_w = np.array([[0.218786], [-0.053837]])
    expected_loss = 0.026942

    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_ridge_regression_lambda0(y, tx):
    lambda_ = 0.0
    w, loss = ridge_regression(y, tx, lambda_)

    expected_loss = 0.026942
    expected_w = np.array([[0.218786], [-0.053837]])

    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_ridge_regression_lambda1(y, tx):
    lambda_ = 1.0
    w, loss = ridge_regression(y, tx, lambda_)

    expected_loss = 0.03175
    expected_w = np.array([[0.054303], [0.042713]])

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_logistic_regression_0_step(y, tx):
    expected_w = np.array([[0.463156], [0.939874]])
    y = (y > 0.2) * 1.0
    w, loss = logistic_regression(y, tx, expected_w, 0, GAMMA)

    expected_loss = 1.533694

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_logistic_regression(y, tx, initial_w):
    y = (y > 0.2) * 1.0
    w, loss = logistic_regression(
        y, tx, initial_w, MAX_ITERS, GAMMA
    )

    expected_loss = 1.348358
    expected_w = np.array([[0.378561], [0.801131]])

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape

def test_reg_logistic_regression(y, tx, initial_w):
    lambda_ = 1.0
    y = (y > 0.2) * 1.0
    w, loss = reg_logistic_regression(
        y, tx, lambda_, initial_w, MAX_ITERS, GAMMA
    )

    expected_loss = 0.972165
    expected_w = np.array([[0.216062], [0.467747]])

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_reg_logistic_regression_0_step(y, tx):
    lambda_ = 1.0
    expected_w = np.array([[0.409111], [0.843996]])
    y = (y > 0.2) * 1.0
    w, loss = reg_logistic_regression(
        y, tx, lambda_, expected_w, 0, GAMMA
    )

    expected_loss = 1.407327

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


ALL_TESTS = [
    test_least_squares,
    test_ridge_regression_lambda0,
    test_ridge_regression_lambda1,
    test_logistic_regression_0_step,
    test_logistic_regression,
    test_reg_logistic_regression_0_step,
    test_reg_logistic_regression,
]