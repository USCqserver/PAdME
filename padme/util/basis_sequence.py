#  Copyright (c) 2019 Humberto Munoz Bauza.
#  Licensed under the MIT License. See LICENSE.txt for license terms.
#  This software is based partially on work supported by IARPA.
#  The U.S. Government is authorized to reproduce and distribute this software for
#  Governmental purposes notwithstanding the conditions of the stated license.
#
"""Utility for constructing an energy basis time partition"""

import numpy as np
import qutip as qt
from qutip import Qobj
from .operators import TimeDependentOperator
from scipy.linalg import eigh
from typing import List, Dict, Union

Idx = Union[str, int]

def _is_hermitian(h: np.ndarray):
    return np.all(np.isclose(h - np.conj(h.transpose()), 0.0, atol=1.0e-10))


def is_orthonormal(v: np.ndarray):
    n = v.shape[1]
    return np.all(
        np.isclose(np.conj(v.transpose()) @ v - np.identity(n), 0.0, atol=1.0e-10)
    )


def BasisFidelity(vecs1:np.ndarray, vecs2:np.ndarray):
    """

    :param vecs1: A (d x k1) basis matrix
    :param vecs2: (d x k2)
    :return:
    """
    vshape = vecs1.shape
    if vecs2.shape != vshape:
        raise ValueError("Shapes must be the same: {} {}".format(vecs1.shape, vecs2.shape))

    amplitude_mat = np.conj(vecs2.transpose()) @ vecs1  # k2 x k1
    prob_mat = np.abs(amplitude_mat) ** 2
    overlap1 = np.cumsum(prob_mat, axis=0).transpose()  # k1 x k2
    overlap2 = np.cumsum(prob_mat, axis=1)              # k2 x k1

    return amplitude_mat, overlap1, overlap2

def standardize_momentum_basis(basis: np.ndarray):
    """
    Standardizes the phase of each basis vector given in a 2q + 1 momentum basis
    If the q = 0 momentum component is nonzero, the phase of the vector is
    such that this component is a positive real number.
    If it is close to zero, then the q = +1 and -1 components are checked, and
            (1) their phases are constrained to be opposite of each other
            (2) the q = +1 component should be in the LHP if not real and positive
    (If the basis vector is the fourier transform of a real vector times a phase, this scheme
        eliminates the phase)
    :param basis:  (2q + 1) x k array
    :return:
    """
    basis_shape = np.shape(basis)
    n = basis_shape[0]
    k = basis_shape[1]
    qmax = (n - 1)//2
    phase_arr = np.zeros((k,), dtype=np.complex128)
    for i in range(k):
        vec = basis[:, i]
        a0 = vec[qmax]
        if not np.isclose(a0, 0):  # set a0 to be a positive real number
            th = np.angle(a0)
            ph = np.exp(- 1.0j * th)
            basis[:, i] *= ph
            phase_arr[i] = ph
        else:   # otherwise adjust the phases of the lowest nonzero momentum components
            for q in range(1, qmax+1):
                a = vec[qmax + q]
                b = vec[qmax - q]
                if not np.isclose(a, 0):
                    tha, thb = np.angle(a), np.angle(b)
                    ph = np.exp(- 1.0j * (tha + thb)/2)
                    aph = a * ph
                    if (not np.isclose(np.imag(aph), 0)) and np.imag(aph) > 0:
                        ph = -ph
                    basis[:, i] *= ph
                    phase_arr[i] = ph
                    break
    return basis, phase_arr


def standardize_qubit_basis(basis: np.ndarray):
    """
    Standardize the phase of each basis vector in a qubit basis
    Finds the lowest computational state (by unsigned bits) with a non-zero amplitude and
    sets it to be real.
    :param basis:
    :return:
    """
    basis_shape = np.shape(basis)
    n = basis_shape[0]
    k = basis_shape[1]
    phase_arr = np.zeros((k,), dtype=np.complex128)
    for i in range(k):
        for j in range(n):
            a = basis[j, i]
            if not np.isclose(a, 0.0):
                th = np.angle(a)
                ph = np.exp(- 1.0j * th)
                basis[:, i] *= ph
                phase_arr[i] = ph
                break
    return basis, phase_arr


class BasisSequence:
    """
    This class creates partition of the time interval [0,1], where each partition
    is tagged by the eigenvectors of a Hamiltonian at its midpoint.
    By creating a truncated eigenvector tagged partition, Diagonalization of the
    Hamiltonian can be greatly sped up.
    Currently, this class only implements an evenly spaced partition, but
    it could be modified so that more partitions are created near avoided level
    crossings
    """
    def __init__(self, haml: TimeDependentOperator, tf, basis_size,
                 partitions=10, observables: Dict[Idx, TimeDependentOperator] = None,
                 sched_args : Dict = None):
        """

        :param haml: The Hamiltonian as a TimeDependentOperator
        :param basis_size: The number of tagged eigenvectors to use in each partition
        :param partitions: The number of partitions to create
        :param observables: A string-indexed dictionary of other operators to compute
                in the truncated energy basis of each partition
        """
        self._global_haml = haml
        if haml.op_type is not Qobj:
            raise TypeError("Hamiltonian must be a Qobj term TimeDependentOperator")

        self._tf = tf
        self._num_partitions = partitions
        self._basis_size = basis_size
        self._haml_sequence: List[TimeDependentOperator] = []
        self._basis_sequence_qobj: List[qt.Qobj] = []
        self._basis_sequence_arr : List[np.ndarray] = []
        self._projectors: List[qt.Qobj] = []
        self._projectors_mat: List[np.ndarray] = []
        self._unital_loss = []
        self.intervals = None
        self._original_observables : Dict[Idx, TimeDependentOperator] = observables
        self._observable_sequence : List[Dict[Idx, TimeDependentOperator]] = []
        self._sched_args = sched_args
        #self._trace_preserving = trace_preserving
        self._construct_sequence()

    def _construct_sequence(self):
        """
        Constructs _haml_sequence and _basis_sequence
        :return:
        """
        time_intervals = np.linspace(0.0, self._tf, self._num_partitions + 1, endpoint=True)
        self.intervals = time_intervals
        mid_points = (time_intervals[:-1] + time_intervals[1:]) / 2

        for t in mid_points:
            # Diagonalize the Hamiltonian in the global basis at each partition midpoint
            ht: Qobj = self._global_haml(t, sched_args=self._sched_args)
            #assert(_is_hermitian(ht))

            vals, kets = ht.eigenstates()
            #assert(is_orthonormal(vecs))

            idx = np.argsort(vals)[0:self._basis_size]
            kets = kets[idx]  # kets are n x 1 matrices
            vecs = np.hstack([kets[i].full() for i in range(kets.shape[0])])  # n x k matrix
            qvecs = qt.Qobj(vecs)  # n x k matrix Qobj

            #vecs, _ = standardize_momentum_basis(vecs)
            assert(is_orthonormal(vecs))
            self._basis_sequence_arr.append(vecs)
            self._basis_sequence_qobj.append(qvecs)
            # Evaluate each term of the Hamiltonian in the truncated energy basis
            # This is the compressed Hamiltonian that will be diagonalized during the ODE loop
            # while it is within the interval
            h_transformed = self._global_haml.transform(qvecs).into_dense_array_terms()
            #assert(_is_hermitian(h_transformed(t)))
            #assert(_is_hermitian(h_transformed(t + 0.5/self._num_partitions)))
            #assert(_is_hermitian(h_transformed(t - 0.5/self._num_partitions)))

            self._haml_sequence.append(h_transformed)
            # Also transform any observables
            if self._original_observables is not None:
                d = {}
                for name, op in self._original_observables.items():
                    if not isinstance(op, TimeDependentOperator):
                        raise TypeError("The given operator {} is not a TimeDependentOperator.".format(name))
                    d[name] = op.transform(qvecs).into_dense_array_terms()
                self._observable_sequence.append(d)

        for i in range(self._num_partitions - 1):
            wiH = self._basis_sequence_qobj[i].dag()
            wj = self._basis_sequence_qobj[i + 1]
            forward_projector = wiH * wj
            self._projectors.append(forward_projector)
            proj_id = forward_projector.dag() * forward_projector
            self._unital_loss.append(qt.tracedist(proj_id, qt.identity(self._basis_size)))

            wiH = self._basis_sequence_arr[i].transpose().conj()
            wj = self._basis_sequence_arr[i+1]
            forward_projector_mat = wiH @ wj
            self._projectors_mat.append(forward_projector_mat)

            assert(np.allclose(forward_projector.data - forward_projector_mat, 0.0))

    def _partition_of(self, t):
        # Avoid using. Floating point comparison bug.
        idx = np.nonzero(np.less(t, self.intervals))[0]
        if len(idx) != 0:
            return idx[0] - 1
        elif t <= self._tf:  # Edge case of t == tf
            return self._num_partitions - 1
        else:
            raise ValueError("Could not find time {} inside the intervals\n{}".format(t, self.intervals))

    def eig(self, t, p=None, basis='p', sched_args=None):
        """
        Returns the eigenvalues and eigenvectors of the Hamiltonian at time t
        using partition index p (determined automatically if none)
        :param t:
        :param p:
        :param basis:  'p' (tagged eigenvector basis), 'n' (global basis), or 'np' (both)
        :return: a two tuple vals, vecs if basis is 'p' or 'n'.
        returns a three tuple vals, vecp, and vecn if basis is 'np
        """
        if p is None:
            p = self._partition_of(t)
        ht = self._haml_sequence[p](t, sched_args=sched_args)
        #assert(_is_hermitian(ht))
        vals, vecs = eigh(ht)  # k x k2
        #vals, vecs = ht.eigenstates()  # vecs is an k array of Qobj k-dim kets
        if basis == 'n':
            nvecs = self._basis_sequence_qobj[p].full() #  n x k
            vecs = nvecs @ vecs
            return vals, vecs
        elif basis == 'pn' or basis == 'np':
            nvecs = self._basis_sequence_qobj[p]  # n x k
            vecs_n = nvecs * vecs
            return vals, vecs, vecs_n
        elif basis == 'p':
            return vals, vecs

    def haml(self, t, p=None, sched_args=None):
        if p is None:
            p = self._partition_of(t)
        return self._haml_sequence[p](t, sched_args=sched_args)

    def haml_tdo(self, p) -> TimeDependentOperator:
        return self._haml_sequence[p]

    def observables_as_ndarray_operators(self):
        array_observables : List[Dict[Idx, TimeDependentOperator]] = []
        for observables in self._observable_sequence:
            d = {}
            for name, op in observables.items():
                arr_op = op.into_dense_array_terms()
                d[name] = arr_op
            array_observables.append(d)
        return array_observables

    def hamiltonian_as_ndarray_operator(self):
        new_haml_sequence : List[TimeDependentOperator] = []
        for haml in self._haml_sequence:
            new_haml_sequence.append(haml.into_dense_array_terms())
        return new_haml_sequence

    def observables_at(self, p) -> Dict[Idx, TimeDependentOperator]:
        return self._observable_sequence[p]

    def observable_at(self, obs: Idx, t, p=None, sched_args=None):
        if p is None:
            p = self._partition_of(t)
        return self._observable_sequence[p][obs](t, sched_args=sched_args)

    def observable(self, obs: Idx, p):
        return self._observable_sequence[p][obs]

    def observable_eig(self, obs: Idx, t, p=None, sched_args=None):
        if p is None:
            p = self._partition_of(t)
        op = self.observable(obs, p)(t, sched_args)
        vals, vecs = eigh(op)

        return vals, vecs

    def advance_basis(self, A : np.ndarray, current_p):
        proj: np.ndarray = self._projectors_mat[current_p]
        Aproj = proj.transpose().conj() @ (A @ proj)
        #if self._trace_preserving:
        #    Aproj *= np.trace(A) / np.trace(Aproj)
        return Aproj

    def advance_vector_basis(self, v: Union[qt.Qobj, np.ndarray], current_p):
        proj: qt.Qobj = self._projectors[current_p]
        return proj.dag() * v

    def basis_at(self, t, p=None):
        if p is None:
            p = self._partition_of(t)
        return self._basis_sequence_qobj[p]

    def unital_losses(self):
        return self._unital_loss