import numpy as np

class AdvectionDiffusion1d(object):
  """
  非線形移流拡散方程式のソルバ(1d)


  支配方程式:
    ∂u/∂t + f(u) * ∂u/∂x = g * ∂/∂x(∂u/∂x)
  手法:
    陽解法(非定常項: 陽的 Euler，移流項: QUICK，拡散項: 中心差分)

  Attributes
  ----------
  u0 : ndarray, shape (m,)
    初期条件．u(x, t_0) を意味する。
  t0 : float
    イテレーション開始時刻を保持する．
  h : float
    空間の刻み幅
  dt_max : float
    時間刻み幅の最大値．実際の刻み幅は CFL 条件によってこの値超えない範囲で定まる．拡散項による誤差が大きい場合この値で時間刻みを制御すると良い．
  lbnd : float
    境界条件．u(x_0, t) の値．Dirichlet 境界条件において定数が与えられた場合に対応する．
  ubnd : float
    境界条件．u(x_1, t) の値．lbnd と同様.
  f : callable
    支配方程式の f．u(x_i, t_n) をとって実数を返す関数．定数なら並進方程式，恒等関数なら Burgers 方程式に対応する．
  g : float
    拡散係数．支配方程式の g．
  cfl : float
    CFL 条件．Courant 数がこの値以下になるように時間の刻み幅が選ばれる．通常 1.0 以下の値となる．
  eps : float
    浮動小数点の同値を定める最小距離．絶対値がこの値を下回ったら 0 とみなす．

  See Also
  --------
  QUICK method:
    http://penguinitis.g1.xrea.com/study/note/upwind_difference.pdf
    棚橋隆彦『はじめての CFD ー移流拡散方程式ー』

  Notes
  -----
  3 次精度の風上差分であり制度は良いが安定性が低い．CFL 条件として通常の一次精度のものを課しているのみなので、発散する場合は，`cfl` あるいは `dt_max` で時間刻みの条件を強することで対応する.

  """


  def __init__(self, u0, h, dt_max, lbnd, ubnd, f, g, cfl=0.9, eps=1e-15):
    """
    Parameters
    ----------
    u0 : ndarray, shape (m,)
      初期条件．u(x, t_0) を意味する。
    h : float
      空間の刻み幅
    dt_max : float
    時間刻み幅の最大値．実際の刻み幅は CFL 条件によってこの値超えない範囲で定まる．
    lbnd : float
      境界条件．u(x_0, t) の値．Dirichlet 境界条件において定数が与えられた場合に対応する．
    ubnd : float
      境界条件．u(x_1, t) の値．lbnd と同様.
    f : callable
      支配方程式の f．u(x_i, t_n) をとって実数を返す関数．
    g : float
      拡散係数．支配方程式の g．
    cfl : float, default 0.9
      CFL 条件．Courant 数がこの値以下になるように時間の刻み幅が選ばれる．
    eps : float, default 1e-15
      浮動小数点の同値を定める最小距離．絶対値がこの値を下回ったら 0 とみなす．

    """
    self.u0     = u0.copy()
    self.t0     = 0.0
    self.h      = h
    self.dt_max = dt_max
    self.lbnd   = lbnd
    self.ubnd   = ubnd
    self.f      = np.frompyfunc(f, 1, 1)
    self.g      = g
    self.cfl    = cfl
    self.eps    = eps


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
    h     = self.h
    lbnd  = self.lbnd
    ubnd  = self.ubnd
    f     = self.f
    g     = self.g
    EPS   = self.eps

    # return value
    t = np.full(n_itr, self.t0)
    u = np.zeros((n_itr+1, len(self.u0)))
 
    u[0] = self.u0.copy()
    
    # culc
    for n in range(n_itr):
      # CFL 条件から dt を定める
      dt = self.dt_max
      f_max = np.max(np.abs(f(u[n])))
      if not (f_max < EPS):
        dt = min(dt, h * self.cfl / f_max)
      t[n] += dt * (n+1)

      ## 移流項の計算
      # 風上と風下を両方計算しておいて最後にどちらを選ぶか選択する
      u_mean = 0.5 * (np.append(lbnd, u[n]) + np.append(u[n], ubnd))
      f_mean = f(u_mean)
      u_tmp = np.append([lbnd, lbnd], np.append(u[n], [ubnd, ubnd]))
      u_up    = 3*u_tmp[2:-1] + 6*u_tmp[1:-2] - u_tmp[:-3]
      du_up   = 1/8 * (f_mean[1:] * u_up[1:] - f_mean[:-1] * u_up[:-1])
      u_down  = -1*u_tmp[3:] + 6*u_tmp[2:-1] + 3*u_tmp[1:-2]
      du_down = 1/8 * (f_mean[1:] * u_down[1:] - f_mean[:-1] * u_down[:-1])
      du = np.where(f(u[n]) >= 0.0, du_up, du_down)
      adv_term = - dt / h * du

      # 拡散項の計算
      u_tmp = u_tmp[1:-1]
      dif_term = g * dt / h * (u_tmp[2:] - 2*u_tmp[1:-1] + u_tmp[:-2])

      # 更新
      u[n+1] = u[n] + adv_term + dif_term

    self.u0     = u[n_itr]
    self.t0     = t[n_itr-1]
    return u[1:], t
