# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:01:40 2020

@author: Leon
"""

import pandapower as pp
import pandapower.networks as nw

net = nw.case2869pegase()

net.bus.sort_index(inplace=True)
net.line.sort_index(inplace=True)

pp.runpp(net)
nr = net.res_bus

pp.runpp(net, "helm")
helm = net.res_bus

print((helm-nr).abs().sum())
