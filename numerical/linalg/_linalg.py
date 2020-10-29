import numpy as np
import scipy.linalg # qr_householderが依存している


def qr_givens(A):
  """
  ギブンス回転を使った QR 分解

  考え方:
    `(e_i)` を n 次元実ベクトル空間の自然な正規直交基底とする．また、`k, l` を `A` の対角成分を除く下三角成分の添字とする．
    このとき `A` の l 番目の列ベクトルについて、 k 番目の要素を 0 にする変換を求めたい．ギブンス回転は `(e_l, e_k)` 平面内での時計回りの回転を与える．適切に回転角を与えて列ベクトルが平面内で `e_k` と平行になるようにすればギブンス回転によって求める変換が与えられる．
    また、この変換は `(e_l, ..., e_n)` の張る部分空間内の変換であるため、`A` への左作用は `A` の右下の `(n-l-1)` 次部分正方行列のみにしか影響を与えない．したがって `A` の対角成分を除く下三角成分を左の列から順に、列は上から順に 0 にしていくことで求める分解の上三角行列 `R` が得られる．`Q` は手続きに使ったギブンス回転の合成によって求まる．

  Parameters
  ----------
  A : ndarray, shape (m, n)

  Returns
  -------
  Q : ndarray, shape (m, m)
    直交行列
  R : ndarray, shape (m, n)
    上三角行列

  """
  m, n = A.shape
    
  Q = np.identity(m)
  R = A.copy()

  for l in range(n):
    for k in range(l+1, m):
      # ギブンス回転行列の回転角を求める
      # l番目の列ベクトルのk成分を0にしたい。イメージとしては e_l, e_k 平面上で回転を行ってk成分を0にする。
      theta = np.arctan2(R[k, l], R[l, l])
      # ギブンス回転行列をつくる
      G = np.identity(m)
      G[k, k] = np.cos(theta)
      G[k, l] = -np.sin(theta)
      G[l, k] = np.sin(theta)
      G[l, l] = np.cos(theta)
      # Q, R を更新
      Q = np.dot(Q, G.T)
      R = np.dot(G, R)
  return Q, R


def qr_householder(A):
  """
  ハウスホルダー変換を使った QR 分解

  考え方:
    `x, y` をそれぞれ長さ(l^2 ノルム)の等しいベクトルとする．このとき、
      u = (x - y) / ||x - y||
      H = I - 2 * (u \tensor u)
    と置くと、`H` は x を y に変える直交行列であることが分かる．この `H` はハウスホルダー変換である．
    `A` の列ベクトルを `x` とし、`A` の対角成分を除く下三角成分が `0` になるような長さの等しいベクトルを `y` に選べば、ハウスホルダー変換 `H` によって `A` の選んだ列の対角成分を除く下三角成分が `0` になる．
    k 列の変換を行うとき、上のように選んだ `H` は `A` への作用に関して `A` の右下 `(n-k-1)` 次部分正方行列のみに影響を与える．従って、変換を `A` の列ベクトルに対して左から行っていくことで求める分解の上三角行列 `R` が得られる．`Q` は用いた変換 `H` の合成により求まる．
    
  Parameters
  ----------
  A : ndarray, shape (m, n)

  Returns
  -------
  Q : ndarray, shape (m, m)
    直交行列
  R : ndarray, shape (m, n)
    上三角行列

  """
  m, n = A.shape

  Q = np.identity(n)
  R = A.copy()
  for k in range(m-1):
    # k列目に関する Householder 行列を作る
    # H = [[I, 0], [0, I - u * u.T]]
    # where u = (x - y) / ||x - y||
    # x, y はコードの通りで H は x を y にする変換である
    x = R[:,k][k:n]
    y = np.zeros(n-k)
    y[0] = np.sign(-x[0]) * np.linalg.norm(x) # 符号は理論上どちらでも良いが x の第一成分の逆にすることで桁落ちを避けられる
    H = np.identity(n-k) - 2 * np.outer(x-y, x-y) / np.dot(x-y, x-y)
    if k != 0:
      H = scipy.linalg.block_diag(np.eye(k), H)
    # Q, R の更新
    Q = np.dot(Q, H.T)
    R = np.dot(H, R)
  return Q, R
