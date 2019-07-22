#  Copyright (c) 2019 Humberto Munoz Bauza.
#  Licensed under the MIT License. See LICENSE.txt for license terms.
#  This software is based partially on work supported by IARPA.
#  The U.S. Government is authorized to reproduce and distribute this software for
#  Governmental purposes notwithstanding the conditions of the stated license.
#
"""Tests with single qubit schedules (under development)"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from padme.util.operators import TimeDependentOperator
from padme.solve.se import TDSEPartitionSolver
from padme.solve.p_ame import AMEPartitionSolver
from padme.opensys.bath import Ohmic

mKToGHz = 0.02084
eta = 5.0e-2
#eta = 0
T = 20 * mKToGHz * 2 * np.pi
beta = 1.0 / T
omega_c = (2 * np.pi) * 4
ohmic_bath = Ohmic(eta, omega_c, beta)

def test_TDSE():
    sx = qt.sigmax()
    sz = qt.sigmaz()
    f1 = lambda t, _: 1.0
    f2 = lambda t, _: 1.0 * t

    haml = [[sz, f1], [sx, f2]]
    haml_op = TimeDependentOperator(haml)
    sz_op = TimeDependentOperator([sz])
    print("\n\n**** Using partition solver ****\n\n")
    solver = TDSEPartitionSolver(haml_op, 1.0, partitions=10, basis_energies=2,
                                 observables={"sz": sz_op})
    result = solver.evolve(initial_condition=[1.0, 0.0])
    print("\n\n**** Using qutip solver ****\n\n")

    u0 = qt.ket("1")
    results2: qt.solver.Result = qt.mesolve(haml, u0, np.linspace(0.0, 1.0, 1000, endpoint=True),
                                            e_ops=[sz])
    plt.plot(result.t, result.obs_arrs["sz"])
    plt.plot(results2.times, results2.expect[0])
    plt.show()

    e_basis_kets = solver.instantaneous_eigenbasis_transform(result.t, result.psi, result.p_arr)

    plt.plot(result.t, np.abs(e_basis_kets[0, :])**2)
    plt.show()
    return 0


def test_AdME():
    sx = qt.sigmax()
    sz = qt.sigmaz()
    f1 = lambda t, _: 0.1
    f2 = lambda t, _: 0.5 * t #np.cos(2.0 * np.pi * t/20.0)

    haml = [[sz, f1], [sx, f2]]
    haml_op = TimeDependentOperator(haml)
    sz_op = TimeDependentOperator([sz])

    print("\n\n**** Using partition solver ****\n\n")
    solver = AMEPartitionSolver(haml_op, 1.0, partitions=10, basis_energies=2,
                                system_coupling_ops=[ sz_op ],
                                bath=ohmic_bath,
                                observables={"sz": sz_op})
    result = solver.evolve(initial_condition=[1.0, 0.0], t_pnts_per_part=20, verbose=True,
                           ensure_trace_preserving=True)

    #print("\n\n**** Using qutip solver ****\n\n")

    #u0 = qt.ket("1")
    #results2: qt.solver.Result = qt.mesolve(haml, u0, np.linspace(0.0, 1.0, 1000, endpoint=True),
    #                                        e_ops=[sz])
    #plt.plot(result.t, result.obs_arrs["sz"])
    #plt.plot(results2.times, results2.expect[0])
    #plt.show()

    e_basis_kets = solver.instantaneous_eigenbasis_transform(result)

    plt.plot(result.t, e_basis_kets[0, 0, :])
    plt.show()

    plt.plot(result.t, result.obs_arrs["sz"])
    plt.show()
    #plt.plot(result.t, np.abs(e_basis_kets[0, 1, :]) ** 2)
    #plt.show()
    #plt.plot(result.t, np.abs(e_basis_kets[0, 0, :] - e_basis_kets[1, 1, :]) )
    #plt.show()
    print(np.abs(e_basis_kets[0, 0, -4:] - e_basis_kets[1, 1, -4:]))
    print(np.tanh(beta * np.sqrt(2) * 0.25))
    return 0
