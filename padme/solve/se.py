#  Copyright (c) 2019 Humberto Munoz Bauza.
#  Licensed under the MIT License. See LICENSE.txt for license terms.
#  This software is based partially on work supported by IARPA.
#  The U.S. Government is authorized to reproduce and distribute this software for
#  Governmental purposes notwithstanding the conditions of the stated license.
#

import numpy as np
import qutip as qt
from typing import List, Dict, Optional, Sequence
from qutip.solver import Result
from ..util.basis_sequence import BasisSequence, standardize_momentum_basis
from ..util.operators import TimeDependentOperator, instantaneous_qobj_eigenbasis_transform, \
    instantaneous_eigenbasis_transform


class TDSEResult:
    def __init__(self):
        self.t: Optional[np.ndarray] = None
        self.psi: Optional[np.ndarray] = None
        self.p_arr: Optional[np.ndarray] = None
        self.obs_arrs: Optional[Dict[str, np.ndarray]] = None


class TDSEPartitionSolver:
    def __init__(self, hamiltonian: TimeDependentOperator, tf,
                 partitions=10, basis_energies=20,
                 observables=None,
                 sched_args : Dict = None):
        """
        Performs an adiabatic ME anneal using the given Hamiltonian and a single coupling operator
        (the persistent current for a circuit)
        :param hamiltonian:
        :param partitions:  Number of partitions to use during time evolution
        :param basis_energies:  The size of the tagged basis for each partition
                (Also the size of the density matrix)
        """

        self._denmat_shape = [basis_energies, basis_energies]
        self._hamiltonian = hamiltonian
        self._tf = tf
        self._num_partitions = partitions
        self._basis_energies = basis_energies
        self._sched_args = sched_args
        self._observables: Dict[str, TimeDependentOperator] = {} if observables is None else observables
        self._basis_sequence = BasisSequence(self._hamiltonian, tf, basis_energies, partitions=partitions,
                                             observables=self._observables, sched_args=sched_args)
        self._sequenced_hamiltonian = self._basis_sequence.hamiltonian_as_ndarray_operator()

    def evolve(self,  initial_condition=None,
                      delta_s=1.0e-4,  verbose=False) -> TDSEResult:
        """
        Performs a closed-system evolution
        :param initial_condition:
        :param delta_s:
        :param verbose:
        :param sched_args:
        :return:
        """
        tf = self._tf
        t_list = []
        p_list = []
        result_list: List[Result] = []
        sol_list = []
        obs_lists = {}
        sequenced_observables = self._basis_sequence.observables_as_ndarray_operators()
        sched_args = self._sched_args

        if sched_args is None:
            sched_args = {}

        for name, op in self._observables.items():
            obs_lists[name] = []
        options = qt.Options()
        options.atol = 1.0e-6
        options.rtol = 1.0e-4
        options.nsteps = 10000
        u0 = self._initial_vec(initial_condition)

        for p in range(self._num_partitions):
            if verbose:
                print("{:02.0f} %".format((100.0 * p) / self._num_partitions))

            t0 = tf * p / self._num_partitions
            tp = tf * (p + 1) / self._num_partitions
            t_eval = np.linspace(t0, tp, int(np.ceil(1.0 / (self._num_partitions * delta_s))),
                                 endpoint=True)
            num_ts = len(t_eval)
            # Solution array is t x n during the run

            Hp = self._basis_sequence.haml_tdo(p).op_list

            result: Result = qt.mesolve(Hp, u0, t_eval, args=sched_args, options=options)

            result_list.append(result)
            ket_arr: List[qt.Qobj] = result.states # iterable of T qobj of dimension K
            vec_arr: np.ndarray = np.hstack([ket.full() for ket in ket_arr])  # K x T array
            obs_p: Dict[str, List] = {}

            # Calculate observables at each time
            #if self._observables is not None:
            for name, op in sequenced_observables[p].items():
                obs_p[name] = [op.ev(t_eval[i], vec_arr[:, i], sched_args=sched_args) for i in range(num_ts)]

            # Determine the inital condition for the next partition
            if p < self._num_partitions - 1:
                projected_psi = self._basis_sequence.advance_vector_basis(ket_arr[-1], p)
                u0 = projected_psi
            # Save the solution array as a n x t array
            if p < self._num_partitions - 1:
                p_list.append(np.ones(num_ts - 1, dtype=np.int64) * p)
                t_list.append(t_eval[:-1])
                sol_list.append(vec_arr[:, :-1])
                #if self._observables is not None:
                for name, _ in self._observables.items():
                    obs_lists[name].append(obs_p[name][:-1])
            else:  # include the endpoint at the final partition
                p_list.append(np.ones(num_ts, dtype=np.int64) * p)
                t_list.append(t_eval)
                sol_list.append(vec_arr)
                #if self._observables is not None:
                for name, _ in self._observables.items():
                    obs_lists[name].append(obs_p[name])

        results = TDSEResult()
        results.t = np.concatenate(t_list, axis=0)  # t
        results.psi = np.concatenate(sol_list, axis=1)  # k x t
        results.p_arr = np.concatenate(p_list, axis=0)  # t
        results.obs_arrs = {name: np.concatenate(obs_lists[name])
                            for name, _ in self._observables.items()}

        return results

    def _initial_vec(self, initial_cond=None):
        """
        Constructs the inital densitiy matrix from the given wavefunction
            (using standardized bases)
        :param initial_cond:
        :return:
        """
        if initial_cond is None:
            initial_cond = [1.0]

        initial_cond = np.asarray(initial_cond)
        initial_cond /= np.linalg.norm(initial_cond)  # type: np.ndarray

        vals, vecs_p = self._basis_sequence.eig(0, 0, basis='p', sched_args=self._sched_args)
        idx = np.argsort(vals)[0:len(initial_cond)]
        #vecs_n, phases = standardize_momentum_basis(vecs_n[:, idx])
        vecs_p = vecs_p[idx]

        #vecs_p = vecs_p * phases[np.newaxis, :]
        u0 = qt.Qobj(0)
        for i in range(len(initial_cond)):
            u0 += initial_cond[i] * vecs_p[i]

        return u0

    def instantaneous_eigenbasis_transform(self, t_arr, state_arr, p_arr=None) -> Sequence[qt.Qobj]:
        """
        Transforms the time sequence of states from the tagged partition basis
        to the instantaneous energy basis
        :param t_arr: [T] array
        :param state_arr: [N, N, T] or [N, T]
        :param p_arr: [T]
        :return: The transformed states as a [state_energies, state_energies, T]
        or [state_energies, T] array
        """
        #new_kets = instantaneous_qobj_eigenbasis_transform(
        #    t_arr, state_arr, H=lambda i: self._basis_sequence.haml(t_arr[i], p_arr[i]),
        #    num_energies=self._basis_energies, h_arg='i')
        new_kets = instantaneous_eigenbasis_transform(
            t_arr, state_arr, num_energies=self._basis_energies,
            H=lambda i: self._sequenced_hamiltonian[p_arr[i]](t_arr[i], sched_args=self._sched_args), h_arg='i')

        return new_kets