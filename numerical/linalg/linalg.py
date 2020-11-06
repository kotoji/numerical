from . import _linalg
import numpy as np

# 絶対値がこの値を下回る数は 0 とみなす
EPS = 1e-15

def lu(A, pivotting=False):
  """
  LU 分解
  
  手法として Doolittle 法を用いている．

  Parameters
  ----------
  A : ndarray, shape (n, n)
    正方行列
  pivotting : bool, optional
    ピボッティングを行うかどうかのフラグ．デフォルトで False

  Returns
  -------
  L : ndarray, shape (n, n)
    対角成分が 1 の下三角行列
  U : ndarray, shape (n, n)
    上三角行列
  row_ord : ndarray, shape (n,), optional
    ピボッティングを行った場合のみ返される．ピボッティングによる行の入れ替えの履歴を表す．
    例えば、3x3 行列において row_ord = [1, 2, 0] が返されたとき、本来の i 行目が row_ord[i] 行目に移っていること意味する(0-index で考えていることに注意)．

  Raises
  ------
  ValueError
    行列 A が正方行列でない場合
  ZeroDivisionError
    計算が発散した場合

  """
  A = np.array(A)
  m, n = A.shape
  if m != n:
    raise ValueError("The given matrix must be square.")
  row_ord = np.arange(m)

  for i in range(m):
    if pivotting:
      pv = np.argmax(np.abs(A[:,i][i:m])) + i
      if pv != i:
        row_ord[[pv, i]] = row_ord[[i, pv]]
        A[[pv, i]] = A[[i, pv]]
    if np.abs(A[i, i]) < EPS:
      if not pivotting:
        raise ZeroDivisionError("")
      continue
    for j in range(i+1, m):
      A[j, i] /= A[i, i]
      A[j, i+1:n] -= A[j, i] * A[i, i+1:n]


  L = np.tril(A, k=-1) + np.eye(m, n)
  U = np.triu(A)
  if pivotting:
    return L, U, row_ord
  return L, U


def solve(A, y):
  """
  連立一次方程式ソルバ

  方程式は以下のように与えられているとする:
    Ax = y

  ここで、A は係数行列、x は未知変数、y は定数である．
  
  Parameters
  ----------
  A : ndarray, shape (n, n)
    連立一次方程式の係数行列．正方行列である必要がある．
  y : ndarray, shape (n,)
    連立一次方程式の右辺値

  Returns
  -------
  x : ndarray, shape (n,)
    連立一次方程式の解
    
  Raises
  ------
  ValueError
    係数行列が正方行列でない場合
  ZeroDivisionError
    計算が発散した場合
  
  """
  A = np.array(A)
  y = np.array(y)
  m, n = A.shape
  if m != n:
    raise ValueError("The coefficient matrix must be square.")
  L, U, perm = lu(A, pivotting=True)
  y = np.asarray([y[perm[i]] for i in range(perm.shape[0])])
  for i in range(m):
    y[i+1:m] -= L[i+1:m,i] * y[i]
  for i in range(m-1, -1, -1):
    if np.abs(U[i, i]) < EPS:
      raise ZeroDivisionError("")
    y[i] /= U[i, i]
    for j in range(i-1, -1, -1):
      y[j] -= U[j, i] * y[i]
  return y


def gauss_seidel(A, y, x0=None, rsd=1e-10, max_itr=1000):
  """
  Gauss–Seidel 法による連立一次方程式ソルバ

  Parameters
  ----------
  A : ndarray, shape (n, n)
    係数行列
  y : ndarray, shape (n,)
    連立方程式の定数項
  x0 : ndarray, shpae (n,), optional
    解の推定値
  rsd : float64, optional
    許容残差(residual)．
    残差がこの値を下回れば収束したとみなす．
  max_itr: int, optional
    最大イテレーション数．すべてイテレーションが終了したら収束したとみなす．

  Returns
  -------
  x : ndarray, shape (n,)
    連立方程式の解
  
  Raises
  ------
  ValueError
    係数行列 A が正方行列でない場合
  ZeroDivisionError
    計算が発散した場合

  """
  m, n = A.shape
  if m != n:
    raise ValueError("")

  if x0 is None:
    x0 = y.copy()
  x  = np.zeros(y.shape[0])
  
  for _ in range(max_itr):
    for i in range(n):
      if np.abs(A[i, i]) < EPS:
        raise ZeroDivisionError("")
      x[i] = (y[i] - np.sum(A[i, 0:i] * x[0:i]) - np.sum(A[i, i+1:n] * x0[i+1:n])) / A[i, i]

    if np.linalg.norm(x - x0) / np.linalg.norm(x) < rsd:
      break
    x0 = x.copy()
  return x


def gmres(A, b, x0=None, rsd=1e-10, max_itr=1000):
  """
  GMRES 法による連立一次方程式ソルバ

  Parameters
  ----------
  A : ndarray, shape (n, n)
    係数行列
  b : ndarray, shape (n,)
    連立方程式の定数項
  x0 : ndarray, shpae (n,), optional
    解の推定値
  rsd : float64, default 1e-10
    許容残差(residual)．
    残差がこの値を下回れば収束したとみなす．
  max_itr: int, default 1000
    最大イテレーション数．すべてイテレーションが終了したら収束したとみなす．

  Returns
  -------
  x : ndarray, shape (n,)
    連立方程式の解
  
  Raises
  ------
  ValueError
    係数行列 A が正方行列でない場合

  """
  m, n = A.shape
  if m != n:
    raise ValueError("")

  max_itr = min(n, max_itr)

  if x0 is None:
    x0 = b

  r0 = b - A.dot(x0)

  # Krylov 部分空間の正規直交基底．各行ベクトルは基底ベクトル．
  Q = np.zeros([max_itr, n])
  Q[0] = r0 / np.linalg.norm(r0)

  # 上 Hessenberg 行列
  H = np.zeros([max_itr + 1, max_itr])

  x = np.array(x0, dtype=np.float)
  for k in range(max_itr):
    r = b - A.dot(x)
    # 残差が十分小さければ終了
    if np.linalg.norm(r) < rsd:
      break

    # Arnoldi 法で Krylov 部分空間の正規直交基底を作っていく
    v = A.dot(Q[k])
    for j in range(k+1):
      H[j, k] = np.dot(Q[j], v)
      v -= H[j, k] * Q[j]
    H[k+1, k] = np.linalg.norm(v)
    if (EPS < np.abs(H[k+1, k]) and k+1 < max_itr):
      Q[k+1] = v / H[k+1, k]

    # H * y - Q[0] のノルムを最小にするような y を最小二乗法で求める
    q = np.zeros(H.shape[0])
    q[0] = np.linalg.norm(r0)
    y = lstsq(H, q)
    
    # 推定値の更新
    x = x0 + Q.T.dot(y)
  return x


def lstsq(A, y):
  """
  過剰系の連立一次方程式の近似解を最小二乗法の意味で求める

  方程式は以下のように与えられているとする:
    Ac = b

  ここで、A は係数行列、c は未知変数、b は定数である．
  
  Parameters
  ----------
  A : ndarray, shape (m, n)
    連立一次方程式の係数行列．m >= n である必要がある．
  y : ndarray, shape (m,)
    連立一次方程式の右辺値

  Returns
  -------
  c : ndarray, shape (n,)
    近似解
    
  Raises
  ------
  ValueError
    係数行列の次元が不正な場合
  
  """
  m, n = A.shape
  if m < n:
    raise ValueError("")

  Q, R = qr(A, method='householder')
  Q = Q[:, 0:n]
  R = R[0:n]

  # 後退代入で R * c = Q^{T} * y を解いていく
  c = Q.T.dot(y)
  for i in range(n-1, -1, -1):
    if np.abs(R[i, i]) < EPS:
      c[i] = 0.0
      break
    c[i] /= R[i, i]
    c[0:i] -= R[0:i, i] * c[i]
  return c
 
 
def eigen(A, rsd=1e-10, max_itr=100):
  """
  行列の固有値を求める
  
  手法としてハウスホルダー変換による QR 分解を用いている

  Parameters
  ----------
  A : ndarray, shape(n, n)
    正方行列
  rsd : float64, optional
    許容残差(residual)．
    (対角成分を除く)下三角成分の最大値がこの値を下回れば収束したとみなす．
  max_itr: int, optional
    最大イテレーション数．すべてイテレーションが終了したら収束したとみなす．

  Returns
  -------
  values : ndarray, shape (n,)
    A の固有値

  """
  A = np.array(A)
  
  for _ in range(max_itr):
    # (対角成分を除く)下三角成分のみ残してすべての成分の絶対値をとる
    abstril = np.abs(np.tril(A, k=-1))
    # 下三角成分の最大値
    max_val = np.max(abstril)
    # 終了条件
    if max_val < rsd:
      break
    # 更新
    Q, R = qr(A, method='householder')
    A = np.dot(R, Q)
  return np.diag(A)


def qr(A, method='householder'):
  """
  QR 分解

  Parameters
  ----------
  A : ndarray. shape (m, n)
    行列
  method : string, optional
    QR 分解に用いる手法．デフォルトではハウスホルダー変換が使われる．
      'householder' : ハウスホルダー変換を使う
      'givens' : ギブンス回転を使う
  
  Returns
  -------
  Q : ndarray, shape (m, m)
    直交行列
  R : ndarray, shape (m, n)
    上三角行列

  """
  if method == 'givens':
    return _linalg.qr_givens(A)
  else:
    return _linalg.qr_householder(A)

