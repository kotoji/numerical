import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

import numerical.pde.diffusion_1d as diffusion_1d

"""
線形拡散方程式を Crank–Nicolson 法で解く


支配方程式:
 ∂u/∂t = g * ∂/∂x(∂u/∂x)

初期値・境界条件:
  u(0, t) = 0
  u(1, t) = 0
  u(x, 0) = 略

定常状態:
  u(x, t_{inf}) = 0

"""
# パラメータ
g  = 0.1
dx = 0.005
dt = 0.001   

# d < 1/2 となっていた方が良い(陰解法なので大丈夫ではあるが)
d = g * dt / (dx**2)


# dx 幅で刻んだ x 軸
x = np.arange(0.0, 1.0, dx)

# u の初期値
u0 = 0.1 * norm.pdf(x, 0.5, 0.025)

# 境界条件
lbnd = 0.0
ubnd = 0.0

# solver に初期条件セット
solver = diffusion_1d.Diffusion1d(u0, dx, dt, lbnd, ubnd, g)

# 時間発展とプロット
fig = plt.figure()
ax = fig.add_subplot(111)

def update(frame):
  plt.cla()
  
  u, t = solver.update()
  
  ax.set_xlim(0.0, 1.0)
  ax.set_ylim(-0.2, 1.5)
  plt.title("Crank–Nicolson method")
  ax.set_xlabel("x")
  ax.set_ylabel("u")
  plt.plot(x, u[0], color="orange")
  plt.plot(x, np.zeros(len(u0)), color="gray")

ani = animation.FuncAnimation(fig, update, interval=100)
plt.show()
