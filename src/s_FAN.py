# -*- coding: utf-8 -*-
"""
 Script FAN
Created on Tue Sep 17 16:39:14 2019

@author: papalk
"""
from c_FAN import FAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define characteristics of the three systems in FAN 
# Blue/Green: PPT, black: TC
r_blue  =  [50, 80, 120, 140, 160, 180, 200, 220, 240, 260, 280, 320, 360, 400, 450, 500, 550, 600, 650, 700]
r_green =  [50, 80, 120, 140, 160, 180, 200, 220, 240, 260, 280, 320, 360, 400, 450, 500, 550, 600, 650, 700]
r_black =  [50, 80, 90, 100, 110, 120, 140, 160, 180, 200]

BLUE = FAN (  r_blue  , 15 , 5    , 4) # ranges, openingangle_hor, openingangle_ver, nr points)
GREEN = FAN ( r_green , 15 , 5    , 4)
BLACK = FAN ( r_black , 15 , 12.5 , 4)



fig = plt.figure()
ax = Axes3D(fig)
x , y ,z = BLUE.mode(mode='advanced', y_displ = -0.96 ,x_displ = -10.5,z_displ = 115+0.95,tilt = 0, yaw = 5)
ax.scatter(x,y,z,label='BLUE',color = 'blue')
x , y ,z = GREEN.mode(mode='advanced', y_displ = 0.96 ,x_displ = -10.5,z_displ = 115+0.95,tilt = 0, yaw = -5)
ax.scatter(x,y,z,label='GREEN',color = 'green')
x , y ,z = BLACK.mode(mode='advanced' ,z_displ = 115+1.1,x_displ = -10.5,tilt = 0 , yaw = 0)
ax.scatter(x,y,z,label='BLACK',color = 'black')
ax.plot([0,0],[0,0],[0,115],color = 'gray',label='AD8',linewidth=5.0)
ax.plot([398,398],[0,0],[0,115],color = 'r',label='MM',linewidth=5.0)


ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Height [m]')
ax.set_xlim([-20,700])
ax.set_ylim([-180,180])
ax.set_zlim([0,200])
ax.view_init(elev=-0, azim=90) #Reproduce view
ax.legend()
ax.grid(False)
ax.set_xlim(0,1000)
ax.set_zlim(0,300)
ax.set_ylim(-250,250)
plt.show

# plt.savefig(r'C:\Users\papalk\Desktop\layout_hubheight_90.png',bbox_inches='tight')
