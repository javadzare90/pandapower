# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:01:40 2020

@author: Leon
"""

import pandapower as pp
import pandapower.networks as nw


def test_helm_nr_comparison():

    net = nw.case2869pegase()
    
    pp.runpp(net, "nr")
    nr = net.res_bus
    
    pp.runpp(net, "helm", max_iteration=100)
    helm = net.res_bus

    diff = (nr-helm).abs().max()
    
    #TODO: these tolerances are not yet good enough
    assert diff.vm_pu < 1e-4
    assert diff.p_mw < 0.1
    assert diff.q_mvar < 0.1

if __name__ == '__main__':
    test_helm_nr_comparison()
