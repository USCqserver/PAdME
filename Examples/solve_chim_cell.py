#  Copyright (c) 2019 Humberto Munoz Bauza.
#  Licensed under the MIT License. See LICENSE.txt for license terms.
#  This software is based partially on work supported by IARPA.
#  The U.S. Government is authorized to reproduce and distribute this software for
#  Governmental purposes notwithstanding the conditions of the stated license.
#

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from padme.util.operators import TimeDependentOperator,  transform_qobjs
from padme.util.basis_sequence import BasisSequence
from padme.solve.se import TDSEPartitionSolver
from padme.solve.p_ame import AMEPartitionSolver
from typing import List, Dict, Callable
from dwutil import AdjacencyList, read_adjacency
from padme.opensys.bath import Ohmic

#AdjacencyList = List[Dict[int, float]]

mKToGHz = 0.02084
eta = 1.0e-4
#eta = 0
T = 15 * mKToGHz * 2 * np.pi
beta = 1.0 / T
omega_c = (2 * np.pi) * 4
ohmic_bath = Ohmic(eta, omega_c, beta)

def anneal_hamiltonian(adjacency_list : AdjacencyList,
                       A : Callable, B : Callable):
    num_qubits = len(adjacency_list)

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

    for i, d in enumerate(adjacency_list):
        for j, K in d.items():
            if i == j:
                Hz += K * _op_sz(i) / 6.0
            else:
                Hz += K * (_op_sz(i) * _op_sz(j)) / 6.0


    haml = [[Hx, A], [Hz, B]]

    return haml, col


def sched_A(t, args : Dict[str, float]):
    s = t / args['tf']
    return 2 * np.pi * np.exp(3.17326 * s) * (
        207.253 * np.power(1 - s, 9)
        + 203.843 * np.power(1 - s, 7)
        - 380.659 * np.power(1 - s, 8))


def sched_B(t, args):
    s = t / args['tf']
    return 2 * np.pi * (0.341734 + 6.71328*s + 32.9702*s*s)


def find_steady():
    adjli = read_adjacency("Examples/Data/C0989_v1_G000.txt")
    t1 = 1000.0
    tf = 1500.0
    coarse_ts = np.linspace(0.0, tf, 30, endpoint=True)
    args = {'tf': tf}
    haml, col = anneal_hamiltonian(adjli, sched_A, sched_B)
    col_dict = {i: TimeDependentOperator([col[i]]) for i in range(8)}
    haml_op = TimeDependentOperator(haml)
    haml_part = BasisSequence(haml_op, tf, basis_size=10, partitions=10,
                              observables=col_dict, sched_args=args)
    h1 = haml_part.haml(t1, sched_args=args)
    cols1 = [haml_part.observable_at(i, t1, sched_args=args) for i in range(8)]

    r000 = qt.ket('00000000').proj()
    r255 = qt.ket('11111111').proj()
    r253 = qt.ket('11111101').proj()
    r125 = qt.ket('01111101').proj()
    rhos = [r000, r255, r253, r125]
    rhos = transform_qobjs(rhos, haml_part.basis_at(t1))

    steady: qt.Qobj = qt.steadystate(h1, cols1)
    ps = [qt.expect(r, steady) for r in rhos]
    print(steady)
    print(ps)


def run_adme():
    adjli = read_adjacency("C0989_v1_G000.txt")
    tf = 200.0
    coarse_ts = np.linspace(0.0, tf, 30, endpoint=True)
    args = {'tf': tf}
    haml, col = anneal_hamiltonian(adjli, sched_A, sched_B)
    haml_op = TimeDependentOperator(haml)
    col_ops = [TimeDependentOperator([col[i]]) for i in range(len(col))]

    # samp1 = [qt.Qobj.evaluate(haml, t, args) for t in coarse_ts]
    # print(coarse_ts[0], "\n\t", samp1[0].eigenenergies(eigvals=10))
    # print(coarse_ts[10], "\n\t", samp1[10].eigenenergies(eigvals=10))
    # print(coarse_ts[-1], "\n\t", samp1[-1].eigenenergies(eigvals=10))


    print("Solving...")
    #solver = TDSEPartitionSolver(haml_op, tf, partitions=10, basis_energies=40,
    #                             sched_args=args)
    solver = AMEPartitionSolver(haml_op, tf, partitions=40, basis_energies=16, system_coupling_ops=col_ops,
                                bath=ohmic_bath, sched_args=args)
    result = solver.evolve(initial_condition=[1.0, 0.0], t_pnts_per_part=50, verbose=True,
                           ensure_trace_preserving=True)

    e_basis_kets = solver.instantaneous_eigenbasis_transform(result)
    plt.plot(result.t, e_basis_kets[0, 0, :])
    plt.plot(result.t, e_basis_kets[1, 1, :])
    plt.plot(result.t, e_basis_kets[2, 2, :])
    plt.plot(result.t, e_basis_kets[3, 3, :])
    plt.show()
    plt.plot(result.t, np.stack([np.trace(e_basis_kets[:, :, i]) for i in range(len(result.t))]))
    plt.show()

if __name__ == "__main__":
    run_adme()