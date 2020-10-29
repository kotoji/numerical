import numpy as np

# np.darray で包む
def wrap(fun, y0):
  y0 = np.array(y0)
  def fun_wrapped(t, y):
    return np.asarray(fun(t, y))
  return fun_wrapped, y0

# 4th order explicit Runge-Kutta method
def rk4(fun, t_span, y0, n_itr):
  if fun(t_span[0], y0).shape[0] != y0.shape[0]:
    raise ValueError("It not equal that the dimensions of integrand and initial value.")
  t = np.arange(t_span[0], t_span[1], (t_span[1]-t_span[0]) / float(n_itr))
  y_out = np.zeros((t.shape[0]-1, y0.shape[0]), dtype=y0.dtype)
  y_out = np.vstack((y0, y_out))
    
  for i in range(t.shape[0] - 1):
    h = t[i+1] - t[i]
    k1 = fun(t[i], y_out[i])
    k2 = fun(t[i] + h/2.0, y_out[i] + h*k1/2.0)
    k3 = fun(t[i] + h/2.0, y_out[i] + h*k2/2.0)
    k4 = fun(t[i] + h, y_out[i] + h*k3)
    # 更新
    y_out[i+1] = y_out[i] + (h/6.0) * (k1 + 2.0*k2 + 2.0*k3 + 2.0*k4)
  return y_out, t

