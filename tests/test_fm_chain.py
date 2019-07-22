#  Copyright (c) 2019 Humberto Munoz Bauza.
#  Licensed under the MIT License. See LICENSE.txt for license terms.
#  This software is based partially on work supported by IARPA.
#  The U.S. Government is authorized to reproduce and distribute this software for
#  Governmental purposes notwithstanding the conditions of the stated license.
#
"""Test with a ferromagnetic chain (under development)"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from typing import List, Callable, Dict
from padme.solve.p_ame import AMEPartitionSolver
from padme.util.operators import TimeDependentOperator,  transform_qobjs
from padme.opensys.bath import Ohmic

mKToGHz = 0.02084
eta = 1.0e-2
#eta = 0
T = 20 * mKToGHz * 2 * np.pi
beta = 1.0 / T
omega_c = (2 * np.pi) * 4
ohmic_bath = Ohmic(eta, omega_c, beta)

def make_hamiltonian(num_qubits: int, A: Callable, B: Callable):

    id = qt.qeye(2)
    sx = qt.sigmax()
    sz = qt.sigmaz()

    Hx = qt.qzero([2 for _ in range(num_qubits)])
    Hz = qt.qzero([2 for _ in range(num_qubits)])
    col: List[qt.Qobj] = []

    def tensor_op(op, k):
        li = [id for _ in range(num_qubits)]
        li[k] = op
        return qt.tensor(li)

    def _op_sx(k):
        return tensor_op(sx, k)

    def _op_sz(k):
        return tensor_op(sz, k)

    for i in range(num_qubits):
        Hx += _op_sx(i) / 2.0
        col.append(_op_sz(i))

    for i in range(num_qubits - 1):
        Hz += -_op_sz(i) * _op_sz(i+1)

    #Hz += 0.25 *_op_sz(0)
    haml = [[Hx, A], [Hz, B]]

    return haml, col


def sched_A_lin(t, args: Dict[str, float]):
    return args['hx'] * (1.0 - (t/args['tf']))


def sched_B_lin(t, args: Dict[str, float]):
    return args['hz'] * (t/args['tf'])

def test_AdME():
    tf = 50.0
    hx = 10.0
    hz = 2.0
    args = {'tf': tf, 'hx': hx, 'hz': hz}

    haml, col = make_hamiltonian(9, sched_A_lin, sched_B_lin)
    haml_op = TimeDependentOperator(haml)
    col_ops = [TimeDependentOperator([col[i]]) for i in range(len(col))]

    solver = AMEPartitionSolver(haml_op, tf, partitions=50, basis_energies=32,
                                system_coupling_ops=col_ops,
                                bath=ohmic_bath, sched_args=args)

    result = solver.evolve(initial_condition=[1.0, 0.0], t_pnts_per_part=20, verbose=True,
                           ensure_trace_preserving=True)

    e_basis_kets = solver.instantaneous_eigenbasis_transform(result)
    plt.plot(result.t, e_basis_kets[0, 0, :])
    plt.plot(result.t, e_basis_kets[1, 1, :])
    plt.plot(result.t, e_basis_kets[2, 2, :])
    plt.plot(result.t, e_basis_kets[3, 3, :])
    plt.show()
    plt.plot(result.t, np.stack([np.trace(e_basis_kets[:, :, i]) for i in range(len(result.t))]))
    plt.show()

    return 0
