import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

from numerical.pde.advection_diffusion_1d import *
from numerical.pde.advection_1d import *

"""
非線形移流拡散方程式を QUICK + FTCS 法で解く


支配方程式:
  du/dt + u * du/dx = k * ∂/∂x(∂u/∂x)

初期値・境界条件:
  u(0, t) = 0
  u(1, t) = 0
  u(x, 0) = 略

"""
# パラメータ
g      = 2.0
dx     = 0.005
dt_max = 5.0e-4
cfl    = 0.8

# dx 幅で刻んだ x 軸
x = np.arange(0.0, 1.0, dx)

# u の初期値
u0 = 0.1 * norm.pdf(x, 0.3, 0.025) + 1.0

# 境界条件
lbnd = 1.0
ubnd = 1.0

# solver に初期条件セット
f = lambda x : x
solver = AdvectionDiffusion1d(u0, dx, dt_max, lbnd, ubnd, f, g, cfl)
 
# 時間発展とプロット
fig = plt.figure()
ax = fig.add_subplot(111)

def update(frame):
  plt.cla()
  
  u, t = solver.update()
  
  ax.set_xlim(0.0, 1.0)
  ax.set_ylim(0.75, 2.75)
  plt.title("QUICK + FTCS")
  ax.set_xlabel("x")
  ax.set_ylabel("u")
  t_str = "time: {:.3f}".format(t[0])
  plt.plot(x, u[0], label=t_str, color="orange")
  ax.legend(loc='upper right')

ani = animation.FuncAnimation(fig, update, interval=100)
plt.show()
