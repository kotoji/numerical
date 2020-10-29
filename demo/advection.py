import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numerical.pde.advection_1d as advection_1d

"""
線形移流方程式を Lax–Wendorff 法で解く


支配方程式:
  du/dt + c * du/dx = 0

初期値・境界条件:
  u(0, t) = 1
  u(1, t) = 0
  u(x, 0) = H(0.1 - x)
ここで、c: 移流速度、H(x): Heaviside の単位ステップ関数である。

解析解:
  u(x, t) = H(ct - x + 0.1)

補足:
  Lax–Wendroff 法は空間 2 次精度であるためエイリアシングが発生する。

"""
# パラメータ
c   = 0.1
dx  = 0.005
cfl = 0.7

# dx 幅で刻んだ x 軸
x = np.arange(0.0, 1.0, dx)

# u の初期値
u0 = np.where(x < 0.1, 1.0, 0.0)
# 他の例(lbnd = ubnd = 0)
# u0 = 0.1 * (norm.pdf(x, 0.3, 0.05) + norm.pdf(x, 0.2, 0.05))

# 境界条件
lbnd = 1.0
ubnd = 0.0

# solver に初期条件セット
f = lambda _ : c
solver = advection_1d.Advection1d(u0, dx, lbnd, ubnd, f, cfl)

# 時間発展とプロット
fig = plt.figure()
ax = fig.add_subplot(111)

def update(frame):
  plt.cla()
  
  u, t = solver.update()
  
  ax.set_xlim(0.0, 1.0)
  ax.set_ylim(-0.2, 1.5)
  plt.title("Lax–Wendroff method")
  ax.set_xlabel("x")
  ax.set_ylabel("u")
  plt.plot(x, u[0], color="orange")
  plt.plot(x, np.where(x - c*t[0] < 0.1, 1.0, 0.0), color="gray")

ani = animation.FuncAnimation(fig, update, interval=100)
plt.show()
