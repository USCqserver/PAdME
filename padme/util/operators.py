#  Copyright (c) 2019 Humberto Munoz Bauza.
#  Licensed under the MIT License. See LICENSE.txt for license terms.
#  This software is based partially on work supported by IARPA.
#  The U.S. Government is authorized to reproduce and distribute this software for
#  Governmental purposes notwithstanding the conditions of the stated license.
#
"""Utility for time-dependent operators"""

import numpy as np
from scipy.linalg import eigh
#from scipy.sparse.linalg import LinearOperator
import qutip as qt
from qutip import Qobj
from numpy import ndarray
import types
from typing import List, Union, Dict

from padme.util.types import QobjOperatorTerm, NdArrayOperatorTerm, OperatorTerm, NdArrayOperatorCallable


def transform_operators(haml_terms: List[NdArrayOperatorTerm], new_basis: ndarray) -> List[np.ndarray]:
    """
    Transforms or projects the list of Hamiltonian terms onto the new basis
    :param haml_terms: list of (d x d) hamiltonians
    :param new_basis: A normalized (d x k) orthogonal basis matrix
    :return: list of transformed (k x k) hamiltonians
    """
    new_terms: List[NdArrayOperatorTerm] = []
    w = np.asarray(new_basis)
    wH = np.transpose(w.conj())

    for term in haml_terms:
        if isinstance(term, ndarray):
            new_terms.append(wH @ (term @ w))
        elif isinstance(term, list) and len(term) == 2:
            op: ndarray = term[0]
            new_terms.append([wH @ (op @ w), term[1]])
        else:
            raise TypeError()
    return new_terms


def transform_qobjs(haml_terms: List[QobjOperatorTerm], new_basis: Qobj) \
        -> List[QobjOperatorTerm]:
    """
    Transforms (or projects) the list of Hamiltonian terms onto the new basis
    i.e. H' = wT @ H @ W
    :param haml_terms: list of (d x d) hamiltonians
    :param new_basis: A normalized (d x k) orthogonal basis matrix
    :return: list of transformed (k x k) hamiltonians
    """
    t_haml: List[QobjOperatorTerm] = []
    w = new_basis.data
    wH = new_basis.dag().data
    if not (len(new_basis.dims[0]) == 1 and len(new_basis.dims[1]) == 1):
        raise ValueError()
    new_basis_len = new_basis.dims[1][0]
    new_dims = [[new_basis_len], [new_basis_len]]
    for h in haml_terms:
        if isinstance(h, Qobj):
            op: Qobj = h
            data = wH * (op.data * w)
            t_haml.append(Qobj(data, dims=new_dims))
        elif isinstance(h, list) and len(h) == 2:
            op: Qobj = h[0]
            data = wH * (op.data * w)
            t_haml.append([Qobj(data, dims=new_dims), h[1]])
        else:
            raise TypeError()
    return t_haml


def _term_type(term: OperatorTerm):
    if isinstance(term, List) and len(term) == 2:
        q = term[0]
    elif isinstance(term, List):
        raise TypeError("Could not recognize data type of operator terms")
    else:
        q = term

    if isinstance(q, Qobj):
        return QobjOperatorTerm, Qobj
    elif isinstance(q, ndarray):
        return NdArrayOperatorTerm, ndarray
    else:
        raise TypeError("Could not recognize data type of operator term")


def _qobj_term_to_array_term(term : QobjOperatorTerm):
    if isinstance(term, list) and len(term) == 2:
        q: Qobj = term[0]
        return [q.full(), term[1]]
    elif isinstance(term, Qobj):
        return term.full()
    else:
        raise TypeError()


def _term_op_coef(term: OperatorTerm):
    if isinstance(term, list) and len(term) == 2:
        return term[0], term[1]
    elif not isinstance(term, list):
        return term, None
    else:
        raise TypeError()


def evaluate_time_dep_operator(haml_list: List[NdArrayOperatorTerm],
                               t, fmt='dense', triangular=False, sched_args : Dict =None) -> ndarray:
    """
    Evaluate a time-dependent Hermitian operator
    where each fi is a scalar callback, or None if the term time independent
    H = f0(t) H0 + f1(t) H1 + ...

    Furthermore, if the type of fk is complex, the Hermitian conjugate of fk(t) Hk is also added,
    *unless* triangular is set to true. In this case, it is assumed that all Hamiltonian terms
    contribute to the same (upper or lower) triangular component.
    :param haml_list:
    :param f_list
    :param t
    :param fmt
    :param triangular
    :return:
    """
    if isinstance(haml_list, List) and len(haml_list) > 0:
        pass
    else:
        raise TypeError("Hamiltonian terms must be a non-empty list.")

    if fmt == 'dense':
        first_op, _ = _term_op_coef(haml_list[0])
        h = np.zeros_like(first_op, dtype=np.complex128)
        for term in haml_list:
            hk, fk = _term_op_coef(term)
            if not isinstance(hk, ndarray):
                raise TypeError("Failed to recognize type of term.")

            if fk is None:
                h += np.asarray(hk)
            else:
                ft = fk(t, sched_args)
                h += ft * np.asarray(hk)
                if not triangular and \
                        (isinstance(ft, complex) or isinstance(ft, np.complexfloating)):
                    h += np.conj(ft * hk.transpose())
        return h
    else:
        raise ValueError("Invalid format {}".format(fmt))


def _expect(op : ndarray, state: ndarray):
    if len(state.shape) == 1:
        state = state[:, np.newaxis]
    return np.squeeze(np.conj(state.transpose()) @ op @ state)


def _expect_rho(op : ndarray, rho: ndarray):
    if len(rho.shape) != 2:
        raise ValueError("Invalid density matrix shape {}".format(rho.shape))
    return np.trace(op @ rho)


def instantaneous_qobj_eigenbasis_transform(t_arr, state_arr: np.ndarray,
                                            H, num_energies, h_arg='s'):
    """
    Transform the array of states into the instantaneous energy basis specified
    by the hamiltonian callback H
    :param t_arr: Array of times with length T
    :param state_arr: 1d array of length T with Qobj density matrices or kets
    :param H: Hamiltonian callable
    :param num_energies: Number of energies to truncate to
    :param h_arg: Whether the callable H accepts s=t or i=0..T-1 as its argument

    :return: A (num_energies x T) or (num_energies x num_energies x T) array
    """

    # arr_shape = state_arr.shape
    # if len(arr_shape) == 2:
    #     new_shape = [num_energies, len(t_arr)]
    #     mat_transform = False
    # elif len(arr_shape) == 3:
    #     new_shape = [num_energies, num_energies, len(t_arr)]
    #     mat_transform = True
    # else:
    #     raise ValueError("Invalid shape {}".format(arr_shape))

    new_psi = [Qobj() for _ in range(len(t_arr))]

    for i in range(len(t_arr)):
        if h_arg == 's':
            haml: Qobj = H(t_arr[i])
        elif h_arg == 'i':
            haml: Qobj = H(i)
        else:
            raise ValueError('Invalid h_arg {}'.format(h_arg))

        vals, kets = haml.eigenstates()
        idx = np.argsort(vals)[0:num_energies]
        kets = kets[idx]  # n x k
        old_state: Qobj = state_arr[i]
        new_psi[i] = old_state.transform(kets)
        #if mat_transform:
        #    new_psi[:, :, i] = np.conj(np.transpose(vecs)) @ state_arr[:, :, i] @ vecs
        #else:
        #    new_psi[:, i] = np.conj(np.transpose(vecs)) @ state_arr[:, i]

    return np.asarray(new_psi, dtype=object)


def instantaneous_eigenbasis_transform(t_arr: ndarray, state_arr: ndarray,
                                       H: NdArrayOperatorCallable, num_energies, h_arg='s'):
    """
    Transform the array of states into the instantaneous energy basis specified
    by the hamiltonian callback H
    :param t_arr: Array of times with length T
    :param state_arr: Rank 2 (wavefunctions) or rank 3 (density matrices)
     The last dimension is treated as the time dimension and should have a length of T
    :param H: Hamiltonian callable
    :param num_energies: Number of energies to truncate to
    :param h_arg: Whether the callable H accepts s=t/t_f or i=0..T-1 as its argument

    :return: A (num_energies x T) or (num_energies x num_energies x T) array
    """
    arr_shape = state_arr.shape
    if len(arr_shape) == 2:
        new_shape = [num_energies, len(t_arr)]
        mat_transform = False
    elif len(arr_shape) == 3:
        new_shape = [num_energies, num_energies, len(t_arr)]
        mat_transform = True
    else:
        raise ValueError("Invalid shape {}".format(arr_shape))

    new_psi = np.zeros(new_shape, dtype=np.complex128)

    for i in range(len(t_arr)):
        if h_arg == 's':
            haml = H(t_arr[i])
        elif h_arg == 'i':
            haml = H(i)
        else:
            raise ValueError('Invalid h_arg {}'.format(h_arg))

        vals, vecs = eigh(haml)
        idx = np.argsort(vals)[0:num_energies]
        vecs = vecs[:, idx]  # n x k
        if mat_transform:
            new_psi[:, :, i] = np.conj(np.transpose(vecs)) @ state_arr[:, :, i] @ vecs
        else:
            new_psi[:, i] = np.conj(np.transpose(vecs)) @ state_arr[:, i]

    return new_psi


class TimeDependentOperator:
    """
    Wrapper class for the time dependent operator representation
     A = f0(t) A0 + f1(t) A1 + ...
    """
    constant = None
    constant_hermitian = lambda _: 1.0 + 0.0j

    def __init__(self, ops: List[OperatorTerm]):
        if len(ops) == 0:
            raise ValueError("Operator term list cannot be empty.")

        self.op_list = ops
        self.Dtype, self.op_type = _term_type(ops[0])

        for i, hk in enumerate(ops):
            if isinstance(hk, list) and len(hk) == 2:
                if not (isinstance(hk[0], self.op_type) and
                        isinstance(hk[1], types.FunctionType)):
                    raise TypeError("Could not recognize time-dependent operator term {}".format(i))
            elif isinstance(hk, self.op_type):
                pass
            else:
                raise TypeError("Could not recognize time-dependent operator term {}".format(i))

    def __call__(self, t, fmt='dense', sched_args: Dict =None) -> Union[Qobj, np.ndarray]:
        """
        Evaluate a time-dependent Hermitian operator
        where each fi is a scalar callback, or None if the term time independent
        H = f0(t) H0 + f1(t) H1 + ...

        Furthermore, if the type of fk is complex, the Hermitian conjugate of fk(t) Hk is also added,
        *unless* triangular is set to true. In this case, it is assumed that all Hamiltonian terms
        contribute to the same (upper or lower) triangular component.
        :param haml:
        :return:
        """
        #return evaluate_time_dep_operator(self.op_list, self.coef_fns, t, format=format, triangular=triangular)
        if sched_args is None:
            sched_args = {}

        if self.op_type is Qobj:
            return qt.Qobj.evaluate(self.op_list, t, sched_args)
        elif self.op_type is np.ndarray:
            return evaluate_time_dep_operator(self.op_list, t, fmt, sched_args=sched_args)
        else:
            raise TypeError()

    def ev(self, t, state, sched_args: Dict = None):
        if sched_args is None:
            sched_args = {}
        if self.op_type is Qobj:
            return qt.expect(qt.Qobj.evaluate(self.op_list, t, sched_args), state)
        elif self.op_type is ndarray:
            return _expect(evaluate_time_dep_operator(self.op_list, t, sched_args=sched_args), state)
        else:
            raise TypeError()

    def ev_den(self, t, rho, sched_args: Dict = None):
        if sched_args is None:
            sched_args = {}
        if self.op_type is Qobj:
            return qt.expect(qt.Qobj.evaluate(self.op_list, t, sched_args), rho)
        elif self.op_type is ndarray:
            return _expect_rho(evaluate_time_dep_operator(self.op_list, t, sched_args=sched_args), rho)
        else:
            raise TypeError()

    def eig(self, t, **kwargs):
        raise NotImplementedError()

    def into_dense_array_terms(self):
        if self.op_type is np.ndarray:
            return self
        elif self.op_type is Qobj:
            new_terms : List[NdArrayOperatorTerm] = []
            for term in self.op_list:
                new_terms.append(_qobj_term_to_array_term(term))
            return TimeDependentOperator(new_terms)
        else:
            raise TypeError()

    def transform(self, new_basis: Union[Qobj, np.ndarray]):
        """
        Returns a new TimeDependentOperator with respect to the transformed basis
        :param new_basis: a n x k array, where n
        :return:
        """
        if self.op_type is np.ndarray:
            new_op_list = transform_operators(self.op_list, new_basis)
        elif self.op_type is Qobj:
            new_op_list = transform_qobjs(self.op_list, new_basis)
        else:
            raise TypeError()

        return TimeDependentOperator(new_op_list)

