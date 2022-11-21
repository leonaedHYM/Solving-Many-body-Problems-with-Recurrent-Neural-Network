import os
import numpy as np
import numpy.linalg as LA
import scipy.linalg as LA1
import scipy.sparse.linalg as LAs
import Sub180221 as Sub
import math,sys,subprocess,glob,copy
import pylab as pl
import Model
import EvoMps
#----------------------------------------------------------------
def GetWave(T):
    Ns = len(T)
    Dp = np.shape(T[0])[1]

    A = T[0][0,:,:]
    for i in range(1,Ns-1):
        A = Sub.NCon([A,T[i]],[[-1,1],[1,-2,-3]])
        A = Sub.Group(A,[[0,1],[2]])
    A = Sub.NCon([A,T[-1][:,:,0]],[[-1,1],[1,-2]])
    Wave = np.reshape(A,[Dp**Ns])

    return Wave

def GetHBond(Ns,Ham):
    Dp = np.shape(Ham)[0]
    Nl = Dp**Ns
    Nop = len(np.shape(Ham))//2
    H = np.reshape(Ham,[Dp**2,Dp**2])
    S0 = np.eye(Dp)
    HBond = np.zeros((Nl,Nl)) + 1j*np.zeros((Nl,Nl))

    A = copy.copy(H)
    for i in range(Ns-Nop):
        A = np.kron(A,S0)
    HBond += A

    for m in range(1,Ns-Nop+1):
        A = copy.copy(S0)
        for i in range(1,m):
            A = np.kron(A,S0)
        A = np.kron(A,H)
        for i in range(Ns-Nop-m):
            A = np.kron(A,S0)
        HBond += A

    return HBond

def GetHSite(OpList):
    Ns = len(OpList)

    HSite = copy.copy(OpList[0])
    for i in range(1,Ns):
        HSite = np.kron(HSite,OpList[i])

    return HSite

def GetCoeff(HamAll,Wave,icheck=0):
    Nl = np.shape(HamAll)[0]
    E,W = LA.eigh(HamAll)
    C = np.zeros((Nl)) + 1j*np.zeros((Nl))

    for i in range(Nl):
        C[i] = np.sum(np.conj(W[:,i])*Wave)

    if icheck == 1:
        Wave1 = Sub.NCon([W,C],[[-1,1],[1]])
        print(LA.norm(Wave1-Wave))


    return E,W,C

def EvoState(E,W,C,Zt):
    CE = C*np.exp(-1j*E*Zt)
    Wave = Sub.NCon([W,CE],[[-1,1],[1]])
    return Wave

def CalOpSite(Ns,Dp,Wave,Type):
    Val = np.zeros((Ns)) + 1j*np.zeros((Ns))

    S0,Sp,Sm,Sz,Sx,Sy = Sub.SpinOper(Dp)
    Sx *= 2.0
    Sy *= 2.0
    Sz *= 2.0

    if Type == 'x':
        Op = copy.copy(Sx)
    if Type == 'y':
        Op = copy.copy(Sy)
    if Type == 'z':
        Op = copy.copy(Sz)
    if Type == 'I':
        Op = copy.copy(S0)

    for i in range(Ns):
        OpList = [S0]*Ns
        OpList[i] = Op
        HSite = GetHSite(OpList)
        Val[i] = Sub.NCon([np.conj(Wave),HSite,Wave],[[1],[1,2],[2]])

    Val = np.real(Val)

    return Val
#----------------------------------------------------------------
def Evolution(Ham,T,Ns,OpType,Zdt,Nt,Zt0,iid,save_dir):
    N0 = Ns//2
    Dp = np.shape(T[0])[1]
    print('Dp:', Dp)
    fmt = '%0.8f \t'*3 + '\n'

    # File = open('ED-M'+OpType+'-N'+str(Ns)+'_'+str(iid)+'.dat','w')
    HamAll = GetHBond(Ns,Ham)
    Wave0 = GetWave(T)
    E,W,C = GetCoeff(HamAll,Wave0)

    result = []
    for it in range(Nt+1):
        Zt = Zdt*it+Zt0
        Wave = EvoState(E,W,C,Zt)
        print(np.shape(Wave))
        break
        Mz = CalOpSite(Ns,Dp,Wave,OpType)
        result.append(Mz)
    filename = 'ED-M'+OpType+'-N'+str(Ns)+'_'+str(iid)
    np.save(os.path.join(save_dir, filename), np.array(result, dtype=np.float32))
    print("Generated {} sequence by ED".format(iid))

    return None

def getMz(Ns, Wave, Dp=2, OpType='z'):
    """一个时刻"""
    Mz = CalOpSite(Ns,Dp,Wave,OpType)

    return Mz


if __name__ == '__main__':
    Dp = 2
    Ns = 6
    Ds = 3
    OpType = 'z'
    Zdt = 0.001
    Zt0 = 0.0
    Nuse = 5
    Nt=201
    Seed=0
    Ham = Model.GetHam_Heisenberg(Dp)
    T = EvoMps.GetInitT(Dp,Ns,Ds,'Rand',Seed)
    save_dir='./'
    iid = 0
    Evolution(Ham,T,Ns,OpType,Zdt,Nt,Zt0,iid,save_dir)
