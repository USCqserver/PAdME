#  Copyright (c) 2019 Humberto Munoz Bauza.
#  Licensed under the MIT License. See LICENSE.txt for license terms.
#  This software is based partially on work supported by IARPA.
#  The U.S. Government is authorized to reproduce and distribute this software for
#  Governmental purposes notwithstanding the conditions of the stated license.
#
"""Simple tests for TimeDependentOperator"""

import numpy as np
import qutip as qt
from padme.util.operators import TimeDependentOperator

def test_np_tdo():
    #sx = np.asarray([[0.0, 1.0],  [1.0, 0.0]])
    #sz = np.asarray([[1.0, 0.0], [0.0, -1.0]])
    sx = qt.sigmax()
    sz = qt.sigmaz()

    f1 = lambda t, _: t
    f2 = lambda t, _: 1.0

    haml = [[sz, f1], [sx, f2]]
    eval_haml = lambda t: qt.Qobj.evaluate(haml, t, {})

    print(eval_haml(0))
    print(eval_haml(1.0))
    print(eval_haml(-1.0))

