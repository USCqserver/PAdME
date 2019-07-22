#  Copyright (c) 2019 Humberto Munoz Bauza.
#  Licensed under the MIT License. See LICENSE.txt for license terms.
#  This software is based partially on work supported by IARPA.
#  The U.S. Government is authorized to reproduce and distribute this software for
#  Governmental purposes notwithstanding the conditions of the stated license.
#

import numpy as np
from scipy.integrate import solve_ivp
import qutip as qt
from typing import List, Dict, Optional, Sequence
from qutip.solver import Result
from ..util.basis_sequence import BasisSequence, standardize_qubit_basis
from ..util.operators import TimeDependentOperator, instantaneous_qobj_eigenbasis_transform, \
    instantaneous_eigenbasis_transform
from ..opensys.bath import Bath, SimpleKMSRate

ListOperators = List[TimeDependentOperator]
NamedOperators = Dict[str, TimeDependentOperator]


class AdMEResult:
    def __init__(self):
        self.t: Optional[np.ndarray] = None
        self.rho: Optional[np.ndarray] = None
        self.p_arr: Optional[np.ndarray] = None
        self.obs_arrs: Optional[Dict[str, np.ndarray]] = None


class AMEPartitionSolver:
    def __init__(self, hamiltonian: TimeDependentOperator, tf,
                 partitions=10, basis_energies=10,
                 system_coupling_ops: Optional[ListOperators] = None,
                 bath: Bath = None,
                 observables: Optional[NamedOperators]=None,
                 sched_args: Dict = None):
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


        #if system_coupling_ops is not None and len(system_coupling_ops) > 1:
        #    raise ValueError("Only a single coupling operator is currently supported")

        self._lind_a_ops: Optional[ListOperators] = system_coupling_ops
        self._num_lind_a = 0 if self._lind_a_ops is None else len(self._lind_a_ops)
        if bath is None:
            bath = SimpleKMSRate(0.0, 0.0)
        self.bath = bath
        self._observables: Dict[str, TimeDependentOperator] = {} if observables is None else observables
        self._ame_keys =[]
        if self._num_lind_a > 0:
            for i in range(self._num_lind_a):
                key = "_p_ame_"+str(i)
                self._ame_keys.append(key)
                self._observables[key] = self._lind_a_ops[i]

        self._num_observables = len(self._observables)
        self._basis_sequence = BasisSequence(self._hamiltonian, tf, basis_energies, partitions=partitions,
                                             observables=self._observables, sched_args=sched_args)
        self._sequenced_hamiltonian = self._basis_sequence.hamiltonian_as_ndarray_operator()

        self._off_diag = (1.0 - np.identity(self._basis_energies))

    def _adiabatic_lindbladian(self, t, p):
        """
        Returns the Lindbladian operations for t
        :param time: time point
        :param tf: anneal time
        :param p: partition index
        :return: vals, vecs, Lind_D, Lind_H
        where Lind_D is the n x n diagonal (population) action matrix
        and Lind_H is the n x n Hadamard (coherence) action matrix
        That is,
                d(rho_ad)/dt = diag(Lind_D @ diag(rho_ad)) (+) Lind_H * coh(rho_ad)
        where rho_ad is in the instantaneous energy basis of the Hamiltonian at time t
        (The direct sum structure here is very helpful)
        """
       # t = time / tf
        #As_op = self._lind_a_ops[0]
        num_A = self._num_lind_a
        vals, vecs = self._basis_sequence.eig(t, p, sched_args=self._sched_args)
        idx = np.argsort(vals)[0:self._basis_energies]
        w = vals[idx]
        v = vecs[:, idx]  # n x k
        #assert(is_orthonormal(v))

        vH = v.conj().transpose()  # k x n
        w_ab = w[:, np.newaxis] - w[np.newaxis, :]  # w_ab[a,b] = w_a - w_b

        haml_mat = -1.0j * w_ab
        if num_A == 0:
            return w, v, np.zeros_like(haml_mat), haml_mat
        #Ip_mat: np.ndarray = self._basis_sequence.observable_at("_p_ame_1", t, p)  # As_op(t)
        # [n, k, k]
        A_ab = np.stack([
                vH @ (self._basis_sequence.observable_at(key, t, p, sched_args=self._sched_args)
                      @ v)
            for key in self._ame_keys])

        g_ab = self.bath.gamma(-w_ab)  # w_ba = - w_ab
        #Asq = np.abs(A_ab) ** 2
        E_Gamma = g_ab * np.sum(np.conj(A_ab[:, np.newaxis])*A_ab[np.newaxis, :],
                                axis=(0,1))
        E_Gamma_offdiag = E_Gamma * self._off_diag
        # [n, k]
        diag_A = np.stack([np.diag(A_ab[i]) for i in range(num_A)])
        GammaZero = g_ab[0, 0] * np.sum(
            np.conj(diag_A[:, np.newaxis,   :, np.newaxis])
                * diag_A[np.newaxis, :,     np.newaxis, :],
            axis=(0, 1))

        GammaZeroDiag = np.diag(GammaZero)
        # diag_A_sq = diag_A * diag_A

        Out_rate = np.sum(E_Gamma_offdiag, axis=0)  # sum_a g_ab Asq[a,b]
        pauli_mat = E_Gamma_offdiag - np.diag(Out_rate)

        #lind_sum_G = np.sum(E_Gamma, axis=0)  # sum_a g_ab Asq[a,b]

        Lind_coh_1 = -(1.0 / 2.0) * (Out_rate[:, np.newaxis] + Out_rate[np.newaxis, :])
        Lind_coh_2 = np.conj(GammaZero[:, :]) + \
                        - (1.0 / 2.0) * (GammaZeroDiag[:, np.newaxis] + GammaZeroDiag[np.newaxis, :])
        #Lind1 = np.diag(E_Gamma_offdiag @ np.diag(rho))
        # The diagonal components are already the out rates in the Pauli matrix
        # and partially cancelled by Lind3
        #Lind2 = -(1.0 / 2.0) * (lind_sum_G[:, np.newaxis] + lind_sum_G[np.newaxis, :]) \
        #    * self._off_diag
        # diagonal cancels out part of the on-diagonal elements of Lind2
        #Lind3 = g_ab[0, 0] * diag_A[:, np.newaxis] * diag_A[np.newaxis, :] \
        #    * self._off_diag

        Lind_D = pauli_mat
        #Lind_H = haml_mat + Lind2 + Lind3
        Lind_H = haml_mat + Lind_coh_1 + Lind_coh_2
        Lind_H *= self._off_diag

        return w, v, Lind_D, Lind_H

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
        initial_cond /= np.linalg.norm(initial_cond)

        vals, vecs_p, vecs_n = self._basis_sequence.eig(0, 0, basis='np', sched_args=self._sched_args)

        idx = np.argsort(vals)[0:len(initial_cond)]
        # vecs_n, phases = standardize_qubit_basis(vecs_n[:, idx])
        vecs_p = vecs_p[:, idx]

        # vecs_p = vecs_p * phases[np.newaxis, :]

        u0 = np.sum(initial_cond[np.newaxis, :] * vecs_p, axis=1)
        # u0 = np.outer(u0, u0.conj()).flatten()

        return u0

    def _ode_step(self, t, y, p):
        rho = y.reshape(self._basis_energies, self._basis_energies)
        vals, v, Lind_D, Lind_H = \
            self._adiabatic_lindbladian(t, p)
        vH = np.transpose(v.conj())
        rho_ad = vH @ (rho @ v)
        rhodot = np.diag(Lind_D @ np.diag(rho_ad)) + Lind_H * (rho_ad * self._off_diag)

        return (v @ (rhodot @ vH)).reshape((-1,))

    def evolve(self, initial_condition=None, t_pnts_per_part=None, verbose=False,
               ensure_trace_preserving=True):
        """
        Performs an open-system evolution using the partitioned anneal method
        Integrated via 4th order Runge-Kutta
        :param initial_condition: initial condition, list of unnormalized amplitudes
            (Mixing eigenstates is still very flimsy, but an excited initial state can be
            indicated by [0.0, 1.0] )
            Defaults to ground state [1.0] if None
        :param t_pnts_per_part:
                Number of time points to sample per partition
                Returned time points determined automatically by ODE if None
        :param verbose
                Prints out progress after each partition
        :return: t, rho, p_arr
            The time array, density matrix array, and the partition index array. If len(t) = T,
            rho.shape = [basis_energies, basis_energies, T]
            p_arr.shape = [T]
        """
        u0 = self._initial_vec(initial_condition)
        y0 = np.outer(u0, u0.conj()).flatten()

        t_list = []
        p_list = []
        sol_list = []
        obs_lists = {}
        # sched_args = self._sched_args

        for name, op in self._observables.items():
            obs_lists[name] = []
        if verbose:
            print("Unital Losses of Partition:\n{}".format(self._basis_sequence.unital_losses()))
        for p in range(self._num_partitions):
            if verbose:
                print("{:02.0f} %".format((100.0 * p) / self._num_partitions))

            # Determine time array if specified
            t0 = p * self._tf / self._num_partitions
            tp = (p + 1) * self._tf / self._num_partitions
            if t_pnts_per_part is None:
                t_eval = None
            else:
                t_eval = np.linspace(t0, tp, t_pnts_per_part + 1, endpoint=True)
            # The persistent current matrix for this partition
            # Ip_op = self._basis_sequence.observable("Ip", p)
            # Get solution for partition
            fun = lambda t, y: self._ode_step(t, y, p)
            solution = solve_ivp(fun, t_span=(t0, tp), y0=y0, method="RK45", t_eval=t_eval)
            sol_t : np.ndarray = solution.t
            sol_y : np.ndarray = solution.y
            obs_p = {idx: np.zeros_like(sol_t) for idx, _ in self._observables.items()}

            for idx, op in self._basis_sequence.observables_at(p).items():
                for i in range(len(sol_t)):
                    y = sol_y[:, i]
                    rho = np.reshape(y, self._denmat_shape)
                    obs_p[idx][i] = op.ev_den(sol_t[i], rho, sched_args=self._sched_args)

            # Determine the inital condition for the next partition
            if p < self._num_partitions - 1:
                rho = np.reshape(sol_y[:, -1], self._denmat_shape)
                projected_rho = self._basis_sequence.advance_basis(rho, p)
                if verbose:
                    print("\t\t Trace Loss = {:6.5f}".format(
                        np.real(np.trace(rho) - np.trace(projected_rho))))
                if ensure_trace_preserving:
                    projected_rho *= np.trace(rho)/np.trace(projected_rho)

                y0 = np.reshape(projected_rho, (-1,))

            # Save the solution array
            if p < self._num_partitions - 1:
                p_list.append(np.ones(len(sol_t) - 1, dtype=np.int64) * p)
                t_list.append(sol_t[:-1])
                sol_list.append(sol_y[:, :-1])
                for name, _ in self._observables.items():
                    obs_lists[name].append(obs_p[name][:-1])
            else:  # include the endpoint at the final partition
                p_list.append(np.ones(len(sol_t), dtype=np.int64) * p)
                t_list.append(sol_t)
                sol_list.append(sol_y)
                for name, _ in self._observables.items():
                    obs_lists[name].append(obs_p[name])

        results = AdMEResult()
        results.t = np.concatenate(t_list, axis=0)
        y = np.concatenate(sol_list, axis=1)
        results.rho = np.reshape(y, [self._basis_energies, self._basis_energies, -1])
        results.p_arr = np.concatenate(p_list, axis=0)
        results.obs_arrs = {name: np.concatenate(obs_lists[name])
                            for name, _ in self._observables.items()}

        return results

    def instantaneous_eigenbasis_transform(self, results: AdMEResult):
        """
        Transforms the time sequence of states from the tagged partition basis
        to the instantaneous energy basis
        :param results
        :return: The transformed states as a [state_energies, state_energies, T]
        or [state_energies, T] array
        """
        t_arr = results.t
        state_arr = results.rho
        p_arr = results.p_arr

        rho_els = instantaneous_eigenbasis_transform(
            t_arr, state_arr, H=lambda i: self.haml(t_arr[i], p_arr[i], sched_args=self._sched_args),
            num_energies=self._basis_energies, h_arg='i')
        return rho_els

    def haml(self, t, p=None, sched_args=None):
        return self._basis_sequence.haml(t, p, sched_args=sched_args)
