import numpy as np
import numerical.linalg as mylinalg

class Diffusion1d(object):
  """
  線形拡散方程式のソルバ(1d)


  支配方程式:
    ∂u/∂t =  g * ∂/∂x(∂u/∂x)
  手法:
    Crank–Nicolson 法

  Attributes
  ----------
  u0 : ndarray, shape (m,)
    初期条件．u(x, t_0) を意味する。
  t0 : float
    イテレーション開始時刻を保持する．
  h : float
    空間の刻み幅
  dt : float
    時間の刻み幅
  lbnd : float
    境界条件．u(x_0, t) の値．Dirichlet 境界条件において定数が与えられた場合に対応する．
  ubnd : float
    境界条件．u(x_1, t) の値．lbnd と同様.
  g : float
    拡散係数．支配方程式の g に対応．
  d : float
    拡散数．g * dt / (dx**2) で求まる．
  A : ndarray, shape (m, m)
    Crank–Nicolson 法において解く連立方程式の係数行列．

  See Also
  --------
  Crank–Nicolson Scheme for the Heat Equation:
    https://people.sc.fsu.edu/~jpeterson/5-CrankNicolson.pdf

  Notes
  -----
  連立一次方程式は Gauss–Seidel 法で解いている．上の PDF からも分かるように解くべき方程式系の係数行列は狭義の対角優位性を満たしているので Gauss–Seidel 法は収束する．

  """

  def __init__(self, u0, h, dt, lbnd, ubnd, g):
    """
    Parameters
    ----------
    u0 : ndarray, shape (n,)
      初期条件．u(x, t_0) を意味する。
    h : float
      空間の刻み幅
    dt : float
      時間の刻み幅
    lbnd : float
      境界条件．u(x_0, t) の値．Dirichlet 境界条件において定数が与えられた場合に対応する．
    ubnd : float
      境界条件．u(x_1, t) の値．lbnd と同様.
    g : float
      拡散係数
    
    """
    self.u0      = u0.copy()
    self.t0      = 0.0
    self.dt      = dt
    self.h       = h
    self.lbnd    = lbnd
    self.ubnd    = ubnd
    self.g       = g
    self.d       = g * dt / (h**2)
    d = self.d
    m = len(u0)
    self.A       = 2*(1+d) * np.eye(m) - d * np.eye(m, k=1) - d * np.eye(m, k=-1)


  def update(self, n_itr=1):
    """
    時間発展を行う

    Parameters
    ----------
    n_itr : int, default 1
      イテレーション回数

    Returns
    -------
    u : ndarray, shape(n_itr, m)
      u の数値解をイテレーション回数分まとめて返す．
    t : ndarray, shape(n_itr,)
      それぞれのイテレーションにおける時刻

    """
    h      = self.h
    dt     = self.dt
    lbnd   = self.lbnd
    ubnd   = self.ubnd
    d      = self.d
    A      = self.A

    # return value
    t = np.full(n_itr, self.t0)
    u = np.zeros((n_itr+1, len(self.u0)))
 
    # culc
    m = len(self.u0)
    u = np.zeros((n_itr+1, m))
    u[0] = self.u0.copy()
    # 時間発展
    for n in range(n_itr):
      t[n] += (n+1) * dt
      u_tmp = d * np.append(lbnd, np.append(u[n], ubnd))
      u[n+1] = 2*(1 - d)*u[n] + u_tmp[2:] + u_tmp[:-2]
      u[n+1, 0] += d * lbnd
      u[n+1, m-1] += d * ubnd
      # A u = y (今 y は u[n+1] に入れてある)を解いて u を更新
      u[n+1] = mylinalg.gauss_seidel(A, u[n+1])

    self.u0     = u[n_itr]
    self.t0     = t[n_itr-1]
    return u[1:], t
