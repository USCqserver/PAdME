#  Copyright (c) 2019 Humberto Munoz Bauza.
#  Licensed under the MIT License. See LICENSE.txt for license terms.
#  This software is based partially on work supported by IARPA.
#  The U.S. Government is authorized to reproduce and distribute this software for
#  Governmental purposes notwithstanding the conditions of the stated license.
#
"""Frequently used types for type annotations"""

from typing import Union, Optional, Callable, List

import numpy as np
from numpy.core._multiarray_umath import ndarray
from qutip import Qobj

OperatorType = Union[np.ndarray,  Qobj]
OperatorCoef = Optional[
        Union[
            Callable[[float], complex],
            Callable[[float], float]]
    ]
QobjOperatorTerm = Union[Qobj,
                         List[Union[Qobj, Callable]]]
NdArrayOperatorTerm = Union[ndarray,
                            List[Union[ndarray, Callable]]]
OperatorTerm = Union[NdArrayOperatorTerm,  QobjOperatorTerm]
NdArrayOperatorCallable = Callable