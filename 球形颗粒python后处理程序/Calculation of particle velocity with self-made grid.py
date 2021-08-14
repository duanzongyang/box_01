import math
import numpy as np
import csv

# 速度归到网格中心
f = open('total data 0.7_210000_2.csv')  # 8 -18 秒数据
L = list(csv.reader(f))
total_A_t_v = []

RAD = 50 / 1000.0
GRID = 10 / 1000.0
MESH = 2 / 1000.0
ANGLE = 30 / 180.0 * np.arccos(-1.0)  # degree

dist = np.arange(-RAD, RAD, MESH)
num = len(dist)
normal = [np.sin(ANGLE), -np.cos(ANGLE)]
total_A_t_v = []


# output = open('result0.5.dat', 'w', encoding='gbk')

line_init = 0

for time_index in range(int(len(L) / 6)):
# for time_index in range(200):
# for time_index in range(1):

    # 读取第一个统计时间的关键数据
    posi_x = L[1 + line_init]
    posi_y = L[2 + line_init]
    velo_x = L[3 + line_init]
    velo_y = L[4 + line_init]

    pos_nor = []

    ntotal = int(len(posi_x))

    for ip in range(ntotal):
        xp = float(posi_x[ip])
        yp = float(posi_y[ip])
        dot = xp * normal[0] + yp * normal[1]
        pos_nor.append(dot)
        # print(dot)

    vector_x = []
    vector_y = []
    for dist_n in pos_nor:
        x_temp = -dist_n * normal[0]
        y_temp = -dist_n * normal[1]
        vector_x.append(x_temp)
        vector_y.append(y_temp)  # 得到向量x，y坐标

    velo_t = []
    dist_n = []
    for i in range(ntotal):
        x_temp = vector_x[i] + float(posi_x[i])
        y_temp = vector_y[i] + float(posi_y[i])
        dist_t = math.sqrt(x_temp * x_temp + y_temp * y_temp)
        if (dist_t < GRID):
            xp = float(posi_x[i])
            yp = float(posi_y[i])
            rxy = np.sqrt(xp * xp + yp * yp)
            tx = -yp / rxy
            ty = xp / rxy
            velo_temp = float(velo_x[i]) * tx + float(velo_y[i]) * ty
            velo_t.append(velo_temp)
            dist_n.append(pos_nor[i])
    # print(dist_n)
    pos_index = []
    for i in range(len(dist_n)):
        x = (RAD-float(dist_n[i]))/MESH
        pos_index.append(x)
    tangential_velocity = [[Tangential_velocity, pos] for Tangential_velocity, pos in zip(velo_t, pos_index)]
    grid_1 = []
    grid_2 = []
    grid_3 = []
    grid_4 = []
    grid_5 = []
    grid_6 = []
    grid_7 = []
    grid_8 = []
    grid_9 = []
    grid_10 = []
    grid_11 = []
    grid_12 = []
    grid_13 = []
    grid_14 = []
    grid_15 = []
    grid_16 = []
    grid_17 = []
    grid_18 = []
    grid_19 = []
    grid_20 = []
    grid_21 = []
    grid_22 = []
    grid_23 = []
    grid_24 = []
    grid_25 = []
    for i in range(len(tangential_velocity)):
        if 0 < float(tangential_velocity[i][1]) < 1:
            grid_1.append(tangential_velocity[i][0])
        elif 1 < float(tangential_velocity[i][1]) < 2:
            grid_2.append(tangential_velocity[i][0])
        elif 2 < float(tangential_velocity[i][1]) < 3:
            grid_3.append(tangential_velocity[i][0])
        elif 3 < float(tangential_velocity[i][1]) < 4:
            grid_4.append(tangential_velocity[i][0])
        elif 4 < float(tangential_velocity[i][1]) < 5:
            grid_5.append(tangential_velocity[i][0])
        elif 5 < float(tangential_velocity[i][1]) < 6:
            grid_6.append(tangential_velocity[i][0])
        elif 6 < float(tangential_velocity[i][1]) < 7:
            grid_7.append(tangential_velocity[i][0])
        elif 7 < float(tangential_velocity[i][1]) < 8:
            grid_8.append(tangential_velocity[i][0])
        elif 8 < float(tangential_velocity[i][1]) < 9:
            grid_9.append(tangential_velocity[i][0])
        elif 9 < float(tangential_velocity[i][1]) < 10:
            grid_10.append(tangential_velocity[i][0])
        elif 10 < float(tangential_velocity[i][1]) < 11:
            grid_11.append(tangential_velocity[i][0])
        elif 11 < float(tangential_velocity[i][1]) < 12:
            grid_12.append(tangential_velocity[i][0])
        elif 12 < float(tangential_velocity[i][1]) < 13:
            grid_13.append(tangential_velocity[i][0])
        elif 13 < float(tangential_velocity[i][1]) < 14:
            grid_14.append(tangential_velocity[i][0])
        elif 14 < float(tangential_velocity[i][1]) < 15:
            grid_15.append(tangential_velocity[i][0])
        elif 15 < float(tangential_velocity[i][1]) < 16:
            grid_16.append(tangential_velocity[i][0])
        elif 16 < float(tangential_velocity[i][1]) < 17:
            grid_17.append(tangential_velocity[i][0])
        elif 17 < float(tangential_velocity[i][1]) < 18:
            grid_18.append(tangential_velocity[i][0])
        elif 18 < float(tangential_velocity[i][1]) < 19:
            grid_19.append(tangential_velocity[i][0])
        elif 19 < float(tangential_velocity[i][1]) < 20:
            grid_20.append(tangential_velocity[i][0])
        elif 20 < float(tangential_velocity[i][1]) < 21:
            grid_21.append(tangential_velocity[i][0])
        elif 21 < float(tangential_velocity[i][1]) < 22:
            grid_22.append(tangential_velocity[i][0])
        elif 22 < float(tangential_velocity[i][1]) < 23:
            grid_23.append(tangential_velocity[i][0])
        elif 23 < float(tangential_velocity[i][1]) < 24:
            grid_24.append(tangential_velocity[i][0])
        elif 24 < float(tangential_velocity[i][1]) < 25:
            grid_25.append(tangential_velocity[i][0])
    grid = [grid_1, grid_2, grid_3, grid_4, grid_5, grid_6, grid_7, grid_8, grid_9, grid_10, grid_11, grid_12, grid_13,
            grid_14, grid_15, grid_16, grid_17, grid_18, grid_19, grid_20, grid_21, grid_22, grid_23, grid_24, grid_25]
    # print(grid)
    A_tangential_velocity = []
    for i in grid:
        try:
            a = sum(i) / len(i)                                                       # 网格平均速度
            A_tangential_velocity.append(a)
        except:
            A_tangential_velocity.append(0)
    # print(A_tangential_velocity)
    total_A_t_v.append(A_tangential_velocity)


    line_init += 6

# print(velocity)

# velocity_AVG = velocity.mean(axis=0)

# print(position)

# position_AVG = position.mean(axis=0)

# tangential_velocity = np.column_stack((position_AVG, velocity_AVG))

# print(tangential_velocity)

# =============================================================================
# velocity = np.delete(velocity, 0, 0)
# velocity = np.delete(velocity, 0, 0)
# position = np.delete(position, 0, 0)
# position = np.delete(position, 0, 0)
# =============================================================================


# output.write(str(tangential_velocity))
# output.write('\n')


lis1 = total_A_t_v[:]
output = open('0.7_2_210000_tangential.xls', 'w', encoding='gbk')
for i in range(len(lis1)):
    for j in range(len(lis1[i])):
        output.write(str(lis1[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
        output.write('\t')  # 相当于Tab一下，换一个单元格
    output.write('\n')  # 写完一行立马换行
output.close()