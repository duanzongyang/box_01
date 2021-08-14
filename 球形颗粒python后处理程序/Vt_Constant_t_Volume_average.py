import math
import numpy as np
import csv

# 速度归到网格中心
                                                  # 8 -18 秒数据

total_A_t_v = []
title_line = 39

RAD = 50/1000.0
RP = 3/2/1000.0
GRID = 10/1000.0
MESH = 2.5/1000.0
ANGLE = 30/180.0*np.arccos(-1.0) # degree

dist = np.arange(-RAD, RAD, MESH)
num = len(dist)
for i in range(num):
    dist[num-1-i] = RAD-MESH*(i+1)
dist[0] = - RAD

normal = [np.sin(ANGLE), -np.cos(ANGLE)]

   
line_init = 0
head_line = 41

output = open('0.7_11000_new_tangential_2.5.dat', 'w', encoding='gbk')
output_1 = open('0.7_3280_37.dat', 'w', encoding='gbk')
output_2 = open('0.7_3280_22.dat', 'w', encoding='gbk')

f = open('total data 0.7_11000.csv')                                                  # 8 -18 秒数据
L = list(csv.reader(f))

for time_index in range(int((len(L))/6)):
# for time_index in range(300):
        
   # 读取第一个统计时间的关键数据
    posi_x = L[1+line_init]
    posi_y = L[2+line_init]
    velo_x = L[3+line_init]
    velo_y = L[4+line_init]

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
        dist_t = math.sqrt(x_temp*x_temp + y_temp*y_temp)
        if (dist_t < GRID):
            xp = float(posi_x[i])
            yp = float(posi_y[i])
            rxy = np.sqrt(xp*xp + yp*yp)
            tx = -yp/rxy
            ty = xp/rxy
            # tx = - normal[1]
            # ty = normal[0]
            velo_temp = float(velo_x[i])*tx + float(velo_y[i])*ty
            velo_t.append(velo_temp)
            dist_n.append(pos_nor[i])
            
    
    vel = np.zeros(num)
    vip = np.zeros(num)
    for i in range(len(velo_t)):
        j = 0
        while dist[j] < dist_n[i]+RP:
            j += 1
            if j == num:
                break
        vp0 = 0.0
        while dist_n[i]-RP < dist[j-1]:
            h = dist_n[i] + RP - dist[j-1]
            vp = h*h*(3*RP-h)/4/RP/RP/RP         
            vel[j-1] += velo_t[i]*(vp-vp0)
            vip[j-1] += vp-vp0
            vp0 = vp
            j = j-1
        vel[j-1] += velo_t[i]*(1-vp)  
        vip[j-1] += 1-vp         
        
    for j in range(num):
        if vip[j]> 0:                
            vel[j] = vel[j]/vip[j]
              
    if (line_init == 0):
        velocity = vel.transpose()

    else:    
        vel_temp = vel.transpose()
        velocity = np.row_stack((velocity, vel_temp))

     
        
        
    line_init += 6
    
    # print(time_index)


#print(velocity)
       
velocity_AVG= velocity.mean(axis=0)



# =============================================================================
# velocity = np.delete(velocity, 0, 0)
# velocity = np.delete(velocity, 0, 0)
# position = np.delete(position, 0, 0)
# position = np.delete(position, 0, 0)
# =============================================================================


for i in range(num):
    output.write(str(float(dist[i]+MESH/2))+str(' , ')+str(float(velocity_AVG[i])))
    output.write('\n')
    
for i in range(velocity.shape[0]):
    output_1.write(str(i)+str(',')+str(velocity[i, 37]))
    output_1.write('\n')
for i in range(velocity.shape[0]):
    output_2.write(str(i)+str(',')+str(velocity[i, 22]))
    output_2.write('\n')

#grid_index = [i_3 for i_3 in np.arange(48.75, 1.25, -2.5)]
# print(grid_index)
# print(total_A_t_v)
#lis1 = total_A_t_v[:]
#output = open('0.5_3780_tangential.xls', 'w', encoding='gbk')
#for i in range(len(lis1)):
    #for j in range(len(lis1[i])):
        #output.write(str(lis1[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
        #output.write('\t')  # 相当于Tab一下，换一个单元格
    #output.write('\n')    # 写完一行立马换行
#output.close()