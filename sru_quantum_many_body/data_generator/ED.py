# -*- coding: utf-8 -*-
"""
@ zewang zhang
"""
import copy
import numpy as np
import numpy.linalg as LA
from scipy import sparse


def base_n(num, b, numerals="0123456789abcdefghijklmnopqrstuvwxyz"):
    """
    将数字num转化成以b进制表示的格式
    """
    return ((num == 0) and numerals[0]) or (base_n(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])


def get_hamr(Ns, Nl, Zj, Zg, Zh):
    """
    获取哈密顿量
    """

    H_i = []
    H_j = []
    Hv = []

    for i in range(Nl):

        # 填充到Ns位
        cfg = base_n(i, 2).zfill(Ns)

        # f(Zj, pos_i, pos_{i+1})
        for pos_i in range(Ns - 1):
            pos_next = pos_i + 1
            H_i.append(i)
            H_j.append(i)

            # Zj*(cfg[pos_i]*2-1)*(cfg[pos_next]*2-1)
            Hv.append(-Zj * (int(cfg[pos_i]) * 2 - 1)
                      * (int(cfg[pos_next]) * 2 - 1))

        # f(Zh, Z)
        for pos_i in range(Ns):
            H_i.append(i)
            H_j.append(i)
            Hv.append(-Zh * (int(cfg[pos_i]) * 2 - 1))

        # f(Zg, X)
        for pos_i in range(Ns):
            _cfg = copy.copy(cfg)  # shallow copy

            # 相当于取反操作
            if cfg[pos_i] == '0':
                _cfg = _cfg[:pos_i] + '1' + _cfg[pos_i + 1:]
            else:
                _cfg = _cfg[:pos_i] + '0' + _cfg[pos_i + 1:]

            j1 = int(_cfg, 2)
            H_i.append(j1)
            H_j.append(i)
            Hv.append(-Zg)

    # coo_matrix为稀疏矩阵，Hv为数据，H_i，H_j为相应的行列索引，形状为Nl*Nl
    # CSC是一种格式
    Hamr = sparse.coo_matrix((Hv, (H_i, H_j)), shape=(Nl, Nl)).tocsc()

    return Hamr


def get_psi0(Ns, Nl, mode, Pos=0):
    """
    设置psi的初始值
    """

    if mode == 'rand':
        psi = np.random.rand(Nl, 1) + 1j * np.random.rand(Nl, 1) - 0.5
    elif mode == 'site':
        psi = np.zeros((Nl, 1)) + 1j * np.zeros((Nl, 1))

        cfg = base_n(0, 2).zfill(Ns)
        _cfg = cfg[:Pos] + '1' + cfg[Pos + 1:]
        j1 = int(_cfg, 2)
        psi[j1] = 1.0
    else:
        raise ValueError('Unsupported mode')

    psi /= LA.norm(psi)  # default paras

    return psi  # (64, 1)


def get_coeff(Nl, psi, V):
    """
    获取系数
    """

    coeff = np.zeros((Nl), dtype=np.complex)

    for i in range(Nl):
        coeff[i] = np.dot(np.transpose(np.conj(V[:, i])), psi)  # V求逆-> V^-1 dot psi

    return coeff


def Evo(Nl, S, V, coeff, t):
    """
    psi随t的演化
    """

    psi = np.zeros((Nl, 1), dtype=np.complex)

    for i in range(Nl):
        psi += coeff[i] * V[:, i] * np.exp(-1j * S[i] * t)

    return psi  # (64, 1)


def calX(Ns, Nl, psi):
    """
    计算MagX
    """

    MagX = np.zeros(Ns)

    for Pos in range(Ns):
        PsiX = np.zeros(Nl) + 1j * np.zeros(Nl)
        for i in range(Nl):
            cfg = base_n(i, 2).zfill(Ns)
            if cfg[Pos] == '0':
                _cfg = cfg[:Pos] + '1' + cfg[Pos + 1:]
            else:
                _cfg = cfg[:Pos] + '0' + cfg[Pos + 1:]
            j1 = int(_cfg, 2)
            PsiX[j1] = psi[i, 0]
        MagX[Pos] = np.real(np.sum(np.conj(psi[:, 0]) * PsiX))

    return MagX


def calZ(Ns, Nl, psi):
    """
    计算MagZ
    """

    MagZ = np.zeros(Ns)

    for Pos in range(Ns):
        PsiZ = np.zeros(Nl) + 1j * np.zeros(Nl)
        for i in range(Nl):
            cfg = base_n(i, 2).zfill(Ns)
            PsiZ[i] = (int(cfg[Pos]) * 2 - 1) * psi[i, 0]
        MagZ[Pos] = np.real(np.sum(np.conj(psi[:, 0]) * PsiZ))

    return MagZ


def get_mag(t, Ns, Zg, Zh, Np=2, Zj=1.0):
    """
     获取t时刻的磁场
     """

    # 表示基底个数
    # 若5个原子，则有2**5=32种可能性
    Nl = Np ** Ns

    Hamr = get_hamr(Ns, Nl, Zj, Zg, Zh)

    # 返回特征值和特征向量
    S, V = LA.eigh(Hamr.todense())

    psi0 = get_psi0(Ns, Nl, 'rand')
    # psi0 = get_psi0(Ns, Nl, 'site', 1)

    coeff = get_coeff(Nl, psi0, V)

    psi = Evo(Nl, S, V, coeff, t)

    MagX = calX(Ns, Nl, psi)
    MagZ = calZ(Ns, Nl, psi)

    return MagX, MagZ


def get_mag_seq(time, step, Ns, Zg, Zh, Np=2, Zj=1.0):
    """
    产生磁场序列
    """

    Nl = Np ** Ns

    Hamr = get_hamr(Ns, Nl, Zj, Zg, Zh)

    S, V = LA.eigh(Hamr.todense())

    psi0 = get_psi0(Ns, Nl, 'rand')

    coeff = get_coeff(Nl, psi0, V)

    mag_x = []
    mag_z = []

    for t in np.arange(0, time, step):
        psi = Evo(Nl, S, V, coeff, t)

        mag_x.append(calX(Ns, Nl, psi))
        mag_z.append(calZ(Ns, Nl, psi))

    return np.array(mag_x), np.array(mag_z)


def get_psi_seq(time, step, Ns, Zg, Zh, Np=2, Zj=1.0):
    """
    获取psi序列
    """

    Nl = Np ** Ns  # 64

    Hamr = get_hamr(Ns, Nl, Zj, Zg, Zh)  # (64, 64)

    S, V = LA.eigh(Hamr.todense())  # S本征值,V本征矢

    psi0 = get_psi0(Ns, Nl, 'rand')  # (64, 1)

    coeff = get_coeff(Nl, psi0, V)  # (64)

    _seq = []

    for t in np.arange(0, time, step):
        _psi = Evo(Nl, S, V, coeff, t)  # (64, 1)
        _seq.append(_psi.reshape(Nl))

    return np.array(_seq)  # (200, 64)



