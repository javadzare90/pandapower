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

    ## constants
    nb = bus.shape[0]  ## number of buses

    ## build Ybus
    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA
    Yseries = Ybus - csr_matrix((Ysh, (range(nb), range(nb))), (nb, nb))

    pvpq = concatenate([pv, pq])
    pvpq.sort()

    max_iter=options["max_iteration"]
    V, success, norm_f, Scalc, iterations, et = helm_josep(Ybus=Ybus, V0=V0, 
                               pq=pq, pv=pv, sl=ref, Ysh0=Ysh, Yseries=Yseries,
                               S0=Sbus, pqpv=pvpq, max_coeff=max_iter,
                               tolerance=1e-8)
    ppci["bus"][:, VM] = abs(V)
    ppci["bus"][:, VA] = angle(V)
    ppci = _store_internal(ppci, {"bus": bus, "gen": gen, "branch": branch,
                                  "baseMVA": baseMVA, "V": V, "pv": pv, "pq": pq, "ref": ref, "Sbus": Sbus,
                                  "ref_gens": ref_gens, "Ybus": Ybus, "Yf": Yf, "Yt": Yt})
    bus, gen, branch = ppci_to_pfsoln(ppci, options)

    ppci = _store_results_from_pf_in_ppci(ppci, bus, gen, branch, success, iterations, et)
    return ppci
