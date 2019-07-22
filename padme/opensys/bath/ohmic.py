#  Copyright (c) 2019 Humberto Munoz Bauza.
#  Licensed under the MIT License. See LICENSE.txt for license terms.
#  This software is based partially on work supported by IARPA.
#  The U.S. Government is authorized to reproduce and distribute this software for
#  Governmental purposes notwithstanding the conditions of the stated license.
#

import numpy as np

from .bath import Bath

# TODO: Implement Lamb Shift calculation
class Ohmic(Bath):
    def __init__(self, eta, wc, beta, gamma0=1):
        self.eta = eta
        self.wc = wc
        self.beta = beta
        self.gamma0 = gamma0

    def gamma(self, w):
        """
        Handles any array of frequencies w, including values numerically close to zero
        :param w:
        :return: array of ohmic bath rates corresponding to the frequencies w
        """
        w_zeros = np.asarray(np.isclose(0, w), dtype=np.float64)

        denom = 1 - np.exp(-self.beta*w)  # denominator is 0 when w is 0
        denom += w_zeros  # denominator of 0s set to 1

        num = w * np.exp(-np.abs(w)/self.wc)  # denominator is 0 when w is 0
        num += w_zeros / self.beta  # set zeroes to 1 / beta
        num *= 2*np.pi*self.eta

        return self.gamma0 * num / denom



class SimpleKMSRate(Bath):
    """
    Primarily for debugging purposes
    """
    def __init__(self, beta, gamma0=1.0):
        self.gamma0 = gamma0
        self.beta = beta

    def gamma(self, w):
        return self.gamma0 * np.exp(self.beta * w / 2)
