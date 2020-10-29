import pytest
import numpy as np
import scipy.integrate
import math

from numerical import integrate

EPS = 1e-1

def approx(val, eps=EPS):
  return pytest.approx(val, rel=eps)

def sumdiff(x, y, ignore_sign=False):
  if ignore_sign:
    d = np.abs(np.abs(x) - np.abs(y))
  else:
    d = np.abs(x - y)
  return d.sum()
  
def test_solve_ivp():
  # QED 初期値問題ソルバのテスト
  f = lambda t, x: ([x[0]*(1.0 - x[0]/(1.0 + 0.5*math.sin(2.0*math.pi*t)))])
  t_span = (0.0, 10.0)
  y0 = [0.1]
  n_itr = 100
  y, t   = integrate.solve_ivp(f, t_span, y0, n_itr)
  res = scipy.integrate.solve_ivp(f, t_span, y0, t_eval=t)
  y_, t_ = res.y.T, res.t
  for i in range(n_itr-10, n_itr-1):
    assert approx(y[i, 0]) == y_[i, 0]
