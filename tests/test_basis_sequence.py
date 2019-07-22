#  Copyright (c) 2019 Humberto Munoz Bauza.
#  Licensed under the MIT License. See LICENSE.txt for license terms.
#  This software is based partially on work supported by IARPA.
#  The U.S. Government is authorized to reproduce and distribute this software for
#  Governmental purposes notwithstanding the conditions of the stated license.
#

"""Basic comparisons for basis_sequence"""

import numpy as np
import qutip as qt
import pytest

from padme.util.operators import TimeDependentOperator
from padme.util.basis_sequence import BasisSequence

def test_basis_sequence():
    sx = qt.sigmax()
    sz = qt.sigmaz()
    f1 = lambda t, _: 1.0
    f2 = lambda t, _: 2.0*t

    haml = [[sz, f1], [sx, f2]]
    haml_op = TimeDependentOperator(haml)

    print("\n\n**** using BasisSequence ****\n\n")
    seq = BasisSequence(haml_op, 1.0, 2, 10)
    print(seq.haml(0.05))
    print(seq.haml(0.26))
    print("\n\n .... \n\n")
    vals1, vecs1, vecsn1 = seq.eig(0.05, p=0, basis='np')
    print(vals1)
    print(vecs1)
    print(vecsn1)

    print("\n\n**** using Qobj.evaluate ****\n\n")
    vals2, vecs2 = qt.Qobj.evaluate(haml_op.op_list, 0.05, {}).eigenstates()
    print(vals2)
    print(vecs2)

    return 0
