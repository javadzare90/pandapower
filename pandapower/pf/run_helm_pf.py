# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from time import time
from numpy import ones, conj, nonzero, any, exp, pi, hstack, real, concatenate, angle
from scipy.sparse import csr_matrix

from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, BR_STATUS, SHIFT, TAP, BR_R_ASYM, BR_X_ASYM
from pandapower.pypower.idx_bus import GS, BS, VM, VA
from pandapower.pf.ppci_variables import _store_results_from_pf_in_ppci
from pandapower.pf.run_newton_raphson_pf import _get_numba_functions, _get_pf_variables_from_ppci, _get_Y_bus, _get_Sbus, ppci_to_pfsoln, _store_internal
from pandapower.pypower.makeYbus import branch_vectors

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _run_helm_pf(ppci, options):
    from GridCal.Engine.Simulations.PowerFlow.helm_power_flow import helm_josep
    """
    Runs a Newton-Raphson power flow.

    INPUT
    ppci (dict) - the "internal" ppc (without out ot service elements and sorted elements)
    options(dict) - options for the power flow

    """
    makeYbus, pfsoln = _get_numba_functions(ppci, options)

    baseMVA, bus, gen, branch, ref, pv, pq, _, _, V0, ref_gens = _get_pf_variables_from_ppci(ppci)

    ppci, Ybus, Yf, Yt = _get_Y_bus(ppci, options, makeYbus, baseMVA, bus, branch)

    # compute complex bus power injections [generation - load]
    Sbus = _get_Sbus(ppci, options["recycle"])

    pvpq = concatenate([pv, pq])
    pvpq.sort()
    ## constants
    nb = bus.shape[0]  ## number of buses
    nl = branch.shape[0]  ## number of lines
    
    ## for each branch, compute the elements of the branch admittance matrix where
    ##
    ##      | If |   | Yff  Yft |   | Vf |
    ##      |    | = |          | * |    |
    ##      | It |   | Ytf  Ytt |   | Vt |
    ##
    Ytt, Yff, Yft, Ytf = branch_vectors(branch, nl)
    ## compute shunt admittance
    ## if Psh is the real power consumed by the shunt at V = 1.0 p.u.
    ## and Qsh is the reactive power injected by the shunt at V = 1.0 p.u.
    ## then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
    ## i.e. Ysh = Psh + j Qsh, so ...
    ## vector of shunt admittances
    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA
    
    ## build connection matrices
    f = real(branch[:, F_BUS]).astype(int)  ## list of "from" buses
    t = real(branch[:, T_BUS]).astype(int)  ## list of "to" buses
    ## connection matrix for line & from buses
    Cf = csr_matrix((ones(nl), (range(nl), f)), (nl, nb))
    ## connection matrix for line & to buses
    Ct = csr_matrix((ones(nl), (range(nl), t)), (nl, nb))
    
    ## build Yf and Yt such that Yf * V is the vector of complex branch currents injected
    ## at each branch's "from" bus, and Yt is the same for the "to" bus end
    i = hstack([range(nl), range(nl)])  ## double set of row indices
    
    Yf = csr_matrix((hstack([Yff, Yft]), (i, hstack([f, t]))), (nl, nb))
    Yt = csr_matrix((hstack([Ytf, Ytt]), (i, hstack([f, t]))), (nl, nb))
    # Yf = spdiags(Yff, 0, nl, nl) * Cf + spdiags(Yft, 0, nl, nl) * Ct
    # Yt = spdiags(Ytf, 0, nl, nl) * Cf + spdiags(Ytt, 0, nl, nl) * Ct
    
    ## build Ybus
    Ybus = Cf.T * Yf + Ct.T * Yt + \
           csr_matrix((Ysh, (range(nb), range(nb))), (nb, nb))
    Yseries = Cf.T * Yf + Ct.T * Yt
    V, success, norm_f, Scalc, iterations, et = helm_josep(Ybus=Ybus, V0=V0, 
                               pq=pq, pv=pv, sl=ref, Ysh0=Ysh, Yseries=Yseries,
                               S0=Sbus, pqpv=pvpq, max_coeff=100, tolerance=1e-8)
    ppci["bus"][:, VM] = abs(V)
    ppci["bus"][:, VA] = angle(V)
    ppci = _store_internal(ppci, {"bus": bus, "gen": gen, "branch": branch,
                                  "baseMVA": baseMVA, "V": V, "pv": pv, "pq": pq, "ref": ref, "Sbus": Sbus,
                                  "ref_gens": ref_gens, "Ybus": Ybus, "Yf": Yf, "Yt": Yt})
    bus, gen, branch = ppci_to_pfsoln(ppci, options)

    ppci = _store_results_from_pf_in_ppci(ppci, bus, gen, branch, success, iterations, et)
    return ppci
