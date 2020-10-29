import pytest
import numpy as np
import scipy

from numerical import linalg

EPS = 1e-5

def approx(val, eps=EPS):
  return pytest.approx(val, abs=eps)


def sumdiff(x, y, ignore_sign=False):
  if ignore_sign:
    d = np.abs(np.abs(x) - np.abs(y))
  else:
    d = np.abs(x - y)
  return d.sum()
  

def test_lu():
  # LU 分解のテスト
  A = np.array([[1, 8, -2],
                [6, 4, 1],
                [3, 2, 0]], dtype=np.float)
  L, U, piv = linalg.lu(A, pivotting=True)
  lu, piv_ = scipy.linalg.lu_factor(A)
  L_, U_ = np.tril(lu, k=-1) + np.eye(3), np.triu(lu)
  assert approx(sumdiff(L, L_)) == 0.0
  assert approx(sumdiff(U, U_)) == 0.0


def test_solve():
  # 連立一次方程式ソルバのテスト
  A = np.array([[5, -4, 6],
                [7, -6, 10],
                [4, 9, 7]], dtype=np.float)
  y = np.array([8, 14, 74], dtype=np.float)
  x  = linalg.solve(A, y)
  x_ = np.linalg.solve(A, y)
  assert approx(sumdiff(x, x_)) == 0.0


def test_gauss_seidel():
  # Gauss–Seidel 法による連立一次方程式ソルバのテスト
  A = np.array([[17, -4, 6],
                [1, -19, 4],
                [4, 2, 18]], dtype=np.float)
  y = np.array([8, 14, 4], dtype=np.float)
  x  = linalg.gauss_seidel(A, y)
  x_ = np.linalg.solve(A, y)
  assert approx(sumdiff(x, x_)) == 0.0


def test_qr_givens():
  # ギブンス回転による QR 分解のテスト
  A = np.array([[5, -4, 6],
                [7, -6, 10],
                [4, 9, 7]], dtype=np.float)
  Q, R = linalg.qr(A, method='givens')
  Q_, R_ = np.linalg.qr(A)
  assert approx(sumdiff(Q, Q_, True)) == 0.0
  assert approx(sumdiff(R, R_, True)) == 0.0


def test_eigen():
  # 固有値を求める関数のテスト
  A = np.array([[1, 8, -2],
                [6, 4, 1],
                [3, 2, 0]], dtype=np.float)
  e_, _ = np.linalg.eig(A)
  e_ = np.sort(e_)
  e = np.sort(linalg.eigen(A))
  assert approx(sumdiff(e_, e)) == 0.0

