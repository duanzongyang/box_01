import math
import numpy as np
import csv

# 速度归到网格中心
f = open('total data 0.7_72000_FORCE.csv')  # 8 -18 秒数据
L = list(csv.reader(f))
total_A_t_v = []

RAD = 50 / 1000.0
GRID = 1.25 / 1000.0
MESH = 2.5 / 1000.0
ANGLE = 30 / 180.0 * np.arccos(-1.0)  # degree

dist = np.arange(-RAD, RAD, MESH)
num = len(dist)
normal = [np.sin(ANGLE), -np.cos(ANGLE)]
total_A_t_v = []

Pi = np.arccos(-1.0)

Radius = 0.05
Dp = 0.003
Rop = 1.0

Rot = 65  # rpm
Vwall = 6.8 * Radius

title_line = 63
time_start = 3.0
time_end = 3.99
time_step = 0.01

Xcenter = 0.0
Ycenter = 0.0

R0 = Radius - 0.0025
Qc = 105
Qs = 5

Q0 = (Qc - 0.5 * Qs) / 180.0 * Pi
Q1 = (Qc + Qs) / 180.0 * Pi

output = open('SN_0.9_72000_105.dat', 'w', encoding='gbk')

line_init = 0

for time_index in range(int(len(L) / 6)):
#for time_index in range(200):
# for time_index in range(1):

    # 读取第一个统计时间的关键数据
    cx = L[1 + line_init]
    CX = []
    for i in cx:
        if i != '' and i != '\n':
            CX.append(i)

    cy = L[2 + line_init]
    CY = []
    for i in cy:
        if i != '' and i != '\n':
            CY.append(i)

    Nx = L[3 + line_init]
    N = []
    for i in Nx:
        if i != '' and i != '\n':
            N.append(i)

    Sx = L[4 + line_init]
    S = []
    for i in Sx:
        if i != '' and i != '\n':
            S.append(i)


    sig_n = 0
    total_NF = 0.0
    total_SF = 0.0




    ntotal = int(len(CX))
    for ip in range(ntotal):
        xp = float(CX[ip])
        yp = float(CY[ip])
        n = float(N[ip])
        s = float(S[ip])
        rad = np.sqrt(pow(xp - Xcenter, 2) + pow(yp - Ycenter, 2))
        if xp - Xcenter > 0:
            theta = np.arccos((Ycenter - yp) / rad)
        else:
            theta = -np.arccos((Ycenter - yp) / rad)

        # angle = theta/Pi*180
        # if 25 < angle and angle < 30 :
        #     print(rad/Radius, angle)

        if rad > R0:
            if Q0 < theta and theta < Q1:
                sig_n = sig_n + 1

                # 1 normal force
                NF = n
                total_NF = total_NF + NF

                # tangential force
                SF = s
                total_SF = total_SF + SF

    # Statistics
    if sig_n > 0:
        avg_NF = total_NF / sig_n

        avg_SF = total_SF / sig_n

    output.write(str(sig_n) + str(',') + str(avg_NF) + str(',') + str(avg_SF))
    output.write('\n')

    line_init += 6

output.close()
