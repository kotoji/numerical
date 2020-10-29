from . import _ivp

def solve_ivp(fun, t_span, y0, n_itr=100):
  """
  常微分方程式の初期値問題を解く

  方程式系は以下のように初期値とともに与えられているとする．
    dy / dt = f(r, y)
    y(t0) = y0
  

  Parameters
  ----------
  fun : callable
    方程式系の右辺．fun(t, y) の形式で呼ばれる．
    ここで、t はスカラー値、y は array_like のベクトル値である．
    y の shape を (n,) とすると、fun の返り値の型は shape (n,) の array_like でなければならない．
  t_span : 2-tuple of floats
    t に関する積分区間 (t0, t1)．t0 < t1 でなければならない．
  y0 : array_like, shape (n,)
    解くべき関数 y の初期値
  n_itr : int. optional
    t_span の分割数．デフォルトで 100 が与えられる

  Returns
  -------
  y : ndarray, shape (n_points, n)
    t における y の数値解
  t : ndarray, shape (n_points,)
    参照時刻の点列

  """
  solver = _ivp.rk4
  _fun, _y0 = _ivp.wrap(fun, y0)
  y, t = solver(_fun, t_span, _y0, n_itr)
  return y, t
