"""Utility math functions which can be shared between Tasks.

Functions:
    gaussian(x_vals, amp, x0, sigma, bkgnd): Calculate a 1D Gaussian distirbution.

    sigma_to_fwhm(sigma): Convert the standard deviation of a Gaussian to Full
        Width at Half Maximum (FWHM).
"""

__all__ = ["gaussian", "sigma_to_fwhm"]
__author__ = "Gabriel Dorlhiac"

import numpy as np


def gaussian(
    x_vals: np.ndarray[np.float64], amp: float, x0: float, sigma: float, bkgnd: float
) -> np.ndarray[np.float64]:
    """1D Gaussian distribution with specified parameters.

    Args:
        x_vals (np.ndarray[np.float64]): Values over which to calculate the
            distribution.

        amp (float): Amplitude of the distribution.

        x0 (float): Center (mean) of the distribution

        sigma (float): Standard deviation of the distribution.

        bkgnd (float | np.ndarray[np.float64]): Background/noise. Constant
            offset (float) or an array of offsets of the same length as `x_vals`.

    Returns:
        distribution (np.ndarray[np.float64]): Calculated Gaussian distribution
            based on given parameters. Same shape as x_vals.
    """
    numerator = -((x_vals - x0) ** 2)
    return amp * np.exp(numerator / (2 * sigma**2)) + bkgnd


def sigma_to_fwhm(sigma: float) -> float:
    """Convert the standard deviation of a Gaussian to Full Width Half Max.

    Args:
        sigma (float): Standard deviation of a Gaussian.

    Returns:
        fhwm (float): Full width at half max of a Gaussian.
    """
    constant: float = (8 * np.log(2)) ** 0.5
    return sigma * constant
