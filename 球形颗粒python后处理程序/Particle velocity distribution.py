# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:53:28 2020
@author: Eagle

# 功能介绍

# 变量说明

# 注意事项（针对不同工况，需调整的参数）

"""

import numpy as np
# import matplotlib.pyplot as plt
# import csv

Pi = np.arccos(-1.0)

Radius = 0.05  
Dp = 0.003
Rop = 1.0

Rot = 65  # rpm
Vwall = Rot*2*Pi/60*Radius

title_line = 63
time_start = 5.0
time_end = 9.5
time_step = 0.5

Xcenter = 0.0
Ycenter = 0.0

R0 = Radius - 0.0018
R1 = Radius - 0.00155
Qc = 60
Qs = 10

Q0 = (Qc - 0.5*Qs)/180.0*Pi
Q1 = (Qc + Qs)/180.0*Pi

output = open('result0.7_72000.dat', 'w', encoding='gbk')

outest = open('v-distribution_0.7_72000_1.55.dat', 'w', encoding='gbk')

with open('total data 0.7_72000_test.csv') as L:
    for i in range(title_line):
        title = L.readline()
    
    for time in np.arange(time_start, time_end+time_step, time_step):
        title = L.readline()
        
        line = L.readline()
        temp = line.split(',')
        print(temp[1])
        
        line = L.readline()
        ID = line.split(',')

        line = L.readline()
        X = line.split(',')
        
        line = L.readline()
        Y = line.split(',')

        line = L.readline()
        Z = line.split(',')

        line = L.readline()
        Vx = line.split(',')
        
        line = L.readline()
        Vy = line.split(',')

        line = L.readline()
        Vz = line.split(',')

        line = L.readline()
        Wx = line.split(',')
        
        line = L.readline()
        Wy = line.split(',')

        line = L.readline()
        Wz = line.split(',')
        
        sig_n  = 0
        sig_vt = 0.0
        sig_vz = 0.0
        
        sig_vx = 0.0
        sig_vy = 0.0
        sig_v2 = 0.0
        
        sig_wx = 0.0
        sig_wy = 0.0
        sig_wz = 0.0
        sig_w2 = 0.0
        
        if time > 4.99:
        
            num = len(ID) - 1        
            for n in range(num):
                i = n + 1
                         
                xp = float(X[i])
                yp = float(Y[i])
                zp = float(Z[i])
                vxp = float(Vx[i])
                vyp = float(Vy[i])
                vzp = float(Vz[i])
                wxp = float(Wx[i])
                wyp = float(Wy[i])
                wzp = float(Wz[i])  
                
                rad = np.sqrt(pow(xp-Xcenter,2) + pow(yp-Ycenter,2))
                if xp-Xcenter > 0:
                    theta = np.arccos((Ycenter-yp)/rad)
                else:
                    theta = -np.arccos((Ycenter-yp)/rad)
                
                # angle = theta/Pi*180
                # if 25 < angle and angle < 30 :
                #     print(rad/Radius, angle)
                
                if rad > R1:
                    
                    if Q0 < theta and theta < Q1:
                        
                        sig_n = sig_n + 1
                            
                        #1 tangential velocity
                        vt = ((Ycenter-yp)*vxp + (xp-Xcenter)*vyp)/rad
                        vn = ((xp-Xcenter)*vxp + (yp-Ycenter)*vyp)/rad
                        vs = Vwall - vt
                        
                        sig_vt = sig_vt + vs
                        sig_vz = sig_vz + vzp
                        
                        # translational granular temperature
                        v2 = vxp*vxp + vyp*vyp + vzp*vzp
                        sig_v2 = sig_v2 + v2
                        sig_vx = sig_vx + vxp
                        sig_vy = sig_vy + vyp
                        
                        # rotational granular temperature
                        sig_wx = sig_wx + wxp
                        sig_wy = sig_wy + wyp
                        sig_wz = sig_wz + wzp
                        sig_w2 = sig_w2 + wxp*wxp + wyp*wyp + wzp*wzp
                        
                        if 4.999<time and time < 10.01:
                            outest.write(str(vt)+str(',')+str(vn)+str(',')+str(vzp)+str(',')+str(wxp)+str(',')+str(wyp)+str(',')+str(wzp))
                            outest.write('\n')
                            print(time, vt)
                    
            # Statistics
            if sig_n > 0:
                
                vta = sig_vt/sig_n
                
                vxa = sig_vx/sig_n
                vya = sig_vy/sig_n
                vza = sig_vz/sig_n
                v2a = sig_v2/sig_n
                T = (v2a - vxa*vxa - vya*vya - vza*vza)/3.0
                
                wxa = sig_wx/sig_n
                wya = sig_wy/sig_n
                wza = sig_wz/sig_n
                w2a = sig_w2/sig_n
                R = (w2a - wxa*wxa - wya*wya - wza*wza)/3.0 * 0.1*Dp*Dp
        
                output.write(str(time)+str(',')+str(sig_n)+str(',')+str(vta)+str(',')+str(vza)+str(',')+str(T)+str(',')+str(R))
                output.write('\n')  
        
    output.close()
    
# f = open('result3.xls')                                                  # 8 -18 秒数据
# L = list(csv.reader(f))        
                
                
                


        
        
