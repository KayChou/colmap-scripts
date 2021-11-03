from math import sqrt
import numpy as np
from scipy.spatial.transform import Rotation as R
import config

parasdir = config.created_dir

f1 = open(parasdir + "images.txt")
f2 = open(parasdir + "cameras.txt")

camnum = 10
Rs = []
Ts = []
Ks = []
wh = []
nam2cam = []
for i in range(camnum):
    Rs.append([])
    Ts.append([])
    Ks.append([])
    wh.append([])
    nam2cam.append([])

for i in range(camnum):
    line = f1.readline().split()
    camid = int(line[8])
    nameid = int(line[9].split('.')[0])

    nam2cam[nameid] = camid

    qw, qx, qy, qz = float(line[1]), float(line[2]), float(line[3]), float(line[4])
    r = R.from_quat([qx, qy, qz, qw]).as_matrix().tolist()

    tx, ty, tz = float(line[5]), float(line[6]), float(line[7])
    Rs[nameid] = [r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], r[2][0], r[2][1], r[2][2]]
    # Ts[nameid] = [tx, ty, tz]
    Ts[nameid] = [r[0][0]*tx+r[1][0]*ty+r[2][0]*tz, r[0][1]*tx+r[1][1]*ty+r[2][1]*tz, r[0][2]*tx+r[1][2]*ty+r[2][2]*tz ]
    line = f1.readline()

for i in range(camnum):
    line = f2.readline().split()

    w, h = int(line[2]), int(line[3])
    wh[i] = [w, h]

    fx, fy, cx, cy = float(line[4]), float(line[5]), float(line[6]), float(line[7])
    Ks[i] = [fx, fy, cx, cy]

f1.close()
f2.close()


f = open("./paras.txt", "w")

for i in range(10):
    print("camera_id "+str(i), file=f)
    print("resolution " + str(wh[nam2cam[i]-1][0]) + " " + str(wh[nam2cam[i]-1][1]), file=f)
    print("K_matrix "+ str(Ks[nam2cam[i]-1][0])+" "+str(Ks[nam2cam[i]-1][1])+" "+str(Ks[nam2cam[i]-1][2])+" "+str(Ks[nam2cam[i]-1][3]), file=f)

    print("R_matrix", end='', file=f)
    for j in range(9):
        print(" "+str(Rs[i][j]), end='', file=f)
    print(file=f)

    print("world_position "+str(-Ts[i][0])+" "+str(-Ts[i][1])+" "+str(-Ts[i][2]), file=f)

f.close()