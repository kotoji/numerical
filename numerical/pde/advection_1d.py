import numpy as np

class Advection1d(object):
  """
  非線形移流方程式のソルバ(1d)


  支配方程式:
    ∂u/∂t + f(u) * ∂u/∂x = 0
  手法:
    Lax–Wendroff 法

  Attributes
  ----------
  u0 : ndarray, shape (m,)
    初期条件．u(x, t_0) を意味する。
  u0_old : ndarray, shape (m,)
    前回の u0 の値を保持する．u(x, t_{-1}) を意味する．
  t0 : float
    イテレーション開始時刻を保持する．
  h : float
    空間の刻み幅
  lbnd : float
    境界条件．u(x_0, t) の値．Dirichlet 境界条件において定数が与えられた場合に対応する．
  ubnd : float
    境界条件．u(x_1, t) の値．lbnd と同様.
  f : callable
    支配方程式の f．u(x_i, t_n) をとって実数を返す関数．定数なら並進方程式，恒等関数なら Burgers 方程式に対応する．
  cfl : float
    CFL 条件．Courant 数がこの値以下になるように時間の刻み幅が選ばれる．

  """


  def __init__(self, u0, h, lbnd, ubnd, f, cfl=0.9):
    """
    Parameters
    ----------
    u0 : ndarray, shape (m,)
      初期条件．u(x, t_0) を意味する。
    h : float
      空間の刻み幅
    lbnd : float
      境界条件．u(x_0, t) の値．Dirichlet 境界条件において定数が与えられた場合に対応する．
    ubnd : float
      境界条件．u(x_1, t) の値．lbnd と同様.
    f : callable
      支配方程式の f．u(x_i, t_n) をとって実数を返す関数．与えられなければ 1.0 を返す定数関数になる．
    cfl : float, default 0.9
      CFL 条件．Courant 数がこの値以下になるように時間の刻み幅が選ばれる．

    """
    self.u0     = u0.copy()
    self.u0_old = u0.copy()
    self.t0     = 0.0
    self.h      = h
    self.lbnd   = lbnd
    self.ubnd   = ubnd
    self.f      = np.frompyfunc(f, 1, 1)
    self.cfl    = cfl
    

  def update(self, n_itr=1):
    """
    時間発展を行う


    Parameters
    ----------
    n_itr : int, default 1
      イテレーション回数

    Returns
    -------
    u : ndarray, shape(n_itr, n)
      u の数値解をイテレーション回数分まとめて返す．
    t : ndarray, shape(n_itr,)
      それぞれのイテレーションにおける時刻

    """
    h    = self.h
    lbnd = self.lbnd
    ubnd = self.ubnd
    f    = self.f

    # return value
    t = np.full(n_itr, self.t0)
    u = np.zeros((n_itr+2, self.u0.shape[0]))
 
    u[0] = self.u0_old.copy()
    u[1] = self.u0.copy()
    
    # culc
    for n in range(1, n_itr+1):
      dt = h * self.cfl / np.max(np.abs(f(u[n])))
      t[n-1] += dt * n

      u_mean = 0.5 * (np.append(lbnd, u[n]) + np.append(u[n], ubnd))
      fs = f(u_mean)
      u_tmp = np.append(lbnd, np.append(u[n], ubnd))

      u[n+1] = u[n-1] + dt/h * (fs[:-1]*u_tmp[:-2] - (fs[:-1] - fs[1:])*u[n] - fs[1:]*u_tmp[2:])

    self.u0_old = u[n_itr]
    self.u0     = u[n_itr+1]
    self.t0     = t[n_itr-1]
    return u[2:], t
