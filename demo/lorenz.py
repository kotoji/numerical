import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

import numerical as mynum

def lorenz(t, u, p, b, r):
    x, y, z = u[0], u[1], u[2]
    dxdt = -p*x + p*y
    dydt = -x*z + r*x - y
    dzdt =  x*y - b*z
    return ([dxdt, dydt, dzdt])
fun = lambda t, u: lorenz(t, u, p=10.0, b=8.0/3.0, r=28)
u, t = mynum.integrate.solve_ivp(fun, (0.0, 40.0), [0.1, 0.1, 0.1], n_itr=5000)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(u[:,0], u[:,1], u[:,2])
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
plt.show()
