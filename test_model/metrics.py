import numpy as np
import scipy.optimize
import warnings


def check_linex_args(function):
    """
    Decorator of linex_score function. Provide some checks on arguments, and some doctests.

    Doctests
    --------
    >>> linex_score([1, 1], [1, 1])
    0.0
    >>> linex_score(np.array([1, 2]), np.array([1, 12])) == np.exp(-1) / 2
    True
    >>> linex_score([0, 0], [5, 5]) == linex_score([0], [5])
    True
    """
    def assert_args(*args, **kwargs):
        y_true = kwargs['y_true'] if 'y_true' in kwargs else args[0]
        y_predict = kwargs['y_predict'] if 'y_predict' in kwargs else args[1]
        try:
            assert_shape_msg = (
                "In linex_error function, argument y_true and y_predict must have the same shape. {} != {}"
                .format(y_true.shape, y_predict.shape)
            )
            assert (y_true.shape == y_predict.shape), assert_shape_msg
        except AttributeError:
            # That's fine, y_true or y_predict are not mandatory arrays
            pass
        assert_len_msg = (
            "In linex_error function, argument y_true and y_predict must have the same length. {} != {}"
            .format(len(y_true), len(y_predict))
        )
        assert (len(y_true) == len(y_predict)), assert_len_msg
        return function(*args, **kwargs)
    return assert_args


@check_linex_args
def linex_score(y_true, y_predict, alpha=0.1):
    """
    Compute LinEx loss from true labels and predicted labels

    Parameters
    ----------
    y_true : true values of y. Could be a list, a flat array or an array of shape (1, N). Must be the same length as y_predict.
    y_predict : predicted values of y. Could be a list, a flat array or an array of shape (1, N). Must be the same length as y_true.
    alpha : exponential coefficient in linex loss. Should be a negative float as stated in challenge description.
    """
    error = np.array(y_true) - np.array(y_predict)
    max_error = max(error)
    log_sum_exp = alpha * max_error + np.log(np.sum(np.exp(alpha * (error - max_error))))
    return (np.exp(log_sum_exp) - np.sum(alpha * error)) / len(error) - 1


def check_inverse_linex_args(function):
    """
    Decorator of inverse_linex_score function. Provide some checks on arguments, and some doctests.

    Doctests
    --------
    >>> 5 == linex_score([inverse_linex_score(5)], [0])
    True
    """
    def assert_args(*args, **kwargs):
        score = kwargs['score'] if 'score' in kwargs else args[0]
        assert(score > 0), "score must be positive in inverse_linex_score"
        return function(*args, **kwargs)
    return assert_args


@check_inverse_linex_args
def inverse_linex_score(score):
    """
    Given a score, returns the mean error according to linex loss with alpha=0.1 (default value of linex_score).
    The mean error is assumed to be a negative integer, which means the prediction is assumed to be all the time
    overestimating the true number of calls.
    Basically, this function solve e^(0.1 * mean_error) - 0.1 * mean_error - 1 = score, with mean_error in Z-.

    Parameters
    ----------
    score : type float, positive
    """
    alpha = 0.1
    objective = lambda mean_err: np.exp(alpha * mean_err) - alpha * mean_err - 1 - score
    # Use exponential linear approximation for score < 2 and ignore exponential for score >=2 to estimate the solution
    x0_estimation = np.array(- np.sqrt(2 * score / pow(alpha, 2))) if score < 2 else - (score + 1) / alpha
    warnings.filterwarnings('ignore', 'The iteration is not making good progress')
    return scipy.optimize.fsolve(objective, x0_estimation, xtol=1e-13)[0]
