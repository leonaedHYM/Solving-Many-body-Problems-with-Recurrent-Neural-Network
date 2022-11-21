import os
import numpy as np
import numpy.linalg as LA
import scipy.linalg as LA1
import scipy.sparse.linalg as LAs
import Sub180221 as Sub
import math,sys,subprocess,glob,copy
import pylab as pl
#----------------------------------------------------------------
def GetEvoMpo_Tebd(Ns,Ham,Zdt):
    Dp = np.shape(Ham)[0]

    H = np.reshape(Ham,[Dp**2,Dp**2])
    H = LA1.expm(-1j*H*Zdt)
    H = np.reshape(H,[Dp,Dp,Dp,Dp])

    A = Sub.Group(H,[[0,2],[1,3]])
    U,S,V,Dc = Sub.SplitSvd(A,Dp**2,1)
    U = np.reshape(U,[1,Dp,Dp,Dc])
    V = np.reshape(V,[Dc,Dp,Dp,1])
    II = np.reshape(np.eye(Dp),[1,Dp,Dp,1])

    Mpo = [None]*Ns
    Mpo0 = [None]*Ns
    Mpo1 = [None]*Ns

    Mpo0[0] = copy.copy(II)
    Mpo0[-1] = copy.copy(II)
    for i in range(1,Ns-1,2):
        # print i
        Mpo0[i] = copy.copy(U)
        Mpo0[i+1] = copy.copy(V)
    for i in range(0,Ns,2):
        # print i
        Mpo1[i] = copy.copy(U)
        Mpo1[i+1] = copy.copy(V)

    for i in range(Ns):
        # print i,np.shape(Mpo0[i]),np.shape(Mpo1[i])
        Mpo[i] = Sub.NCon([Mpo1[i],Mpo0[i]],[[-1,-3,1,-5],[-2,1,-4,-6]])
        Mpo[i] = Sub.Group(Mpo[i],[[0,1],[2],[3],[4,5]])
        # print i,np.shape(Mpo[i])

    return Mpo
#----------------------------------------------------------------
def GetNormTT(TA,TB):
    Ns = len(TA)

    A = Sub.NCon([np.conj(TB[0]),TA[0]],[[1,2,-1],[1,2,-2]])
    for i in range(1,Ns-1):
        A = Sub.NCon([A,np.conj(TB[i]),TA[i]],[[1,2],[1,3,-1],[2,3,-2]])
    Norm = Sub.NCon([A,np.conj(TB[-1]),TA[-1]],[[1,2],[1,3,4],[2,3,4]])

    return Norm

def GetNormTHT(TA,Mpo,TB):
    Ns = len(TA)

    A = np.ones((1,1,1))
    for i in range(Ns):
        A = Sub.NCon([A,np.conj(TB[i]),Mpo[i],TA[i]],[[1,2,4],[1,3,-1],[2,3,5,-2],[4,5,-3]])
    Norm = Sub.NCon([A,np.ones((1,1,1))],[[1,2,3],[1,2,3]])

    return Norm

def NormalizeT(T):
    Ns = len(T)

    Norm = GetNormTT(T,T)
    Ratio = (Norm)**(1.0/float(Ns*2))
    # print 'Ratio=',Ratio
    for i in range(Ns):
        T[i] /= Ratio

    return T

def TtoData(T):
    Ns = len(T)
    Data = []

    for i in range(Ns):
        Dall = np.prod(np.shape(T[i]))
        # print i,Dall
        A = copy.copy(T[i]).flatten()
        Data = np.append(Data,A)

    return Data

def DatatoT(Dp,Ns,Ds,Data):
    T = [None]*Ns
    # print len(Data)
    i0 = 0

    for i in range(Ns):
        Dl = min(Dp**i,Dp**(Ns-i),Ds)
        Dr = min(Dp**(i+1),Dp**(Ns-1-i),Ds)
        Dall = Dl*Dp*Dr
        T[i] = np.reshape(Data[i0:i0+Dall],[Dl,Dp,Dr])
        i0 += Dall

    return T
#----------------------------------------------------------------
def GetInitT(Dp,Ns,Ds,Type,Seed):
    T = [None]*Ns
    N0 = Ns//2
    np.random.seed(Seed)
    if Type == 'Flip':
        for i in range(Ns):
            Dl = min(Dp**i,Dp**(Ns-i),Ds)
            Dr = min(Dp**(i+1),Dp**(Ns-1-i),Ds)
            T[i] = (np.random.rand(Dl,Dp,Dr)-0.5)*1.0e-10 + 1j*np.zeros((Dl,Dp,Dr))
            T[i][0,0,0] = 1.0

        T[N0][0,0,0] = 0.0
        T[N0][0,1,0] = 1.0

    if Type == 'Rand':
        for i in range(Ns):
            Dl = min(Dp**i,Dp**(Ns-i),Ds)
            Dr = min(Dp**(i+1),Dp**(Ns-1-i),Ds)
            T[i] = (np.random.rand(Dl,Dp,Dr)-0.5) + 1j*(np.zeros((Dl,Dp,Dr))-0.5)

    T = NormalizeT(T)

    return T

def PrepareT(T,Orient='rl'):
    Ns = len(T)
    Dp = np.shape(T[0])[1]
    Ds = np.shape(T[0])[2]

    if Orient == 'rl':
        for i in range(Ns-1,0,-1):
            # print i,i-1
            U,T[i] = Sub.Mps_LQ0P(T[i])
            U /= U[0,0]
            T[i-1] = np.tensordot(T[i-1],U,(2,0))
        T[0] /= LA.norm(T[0])

    if Orient == 'lr':
        for i in range(Ns-1):
            # print i,i+1
            T[i],U = Sub.Mps_QR0P(T[i])
            U /= U[0,0]
            T[i+1] = np.tensordot(U,T[i+1],(1,0))
        T[-1] /= LA.norm(T[-1])

    return T

def InitHLR(Mpo,TA,TB,Orient='rl'):
    Ns = len(TA)
    HL = [None]*Ns
    HR = [None]*Ns

    HL[0] = np.zeros((1,1,1))
    HL[0][0,0,0] = 1.0
    HR[-1] = np.zeros((1,1,1))
    HR[-1][0,0,0] = 1.0

    if Orient == 'rl':
        for i in range(Ns-1,0,-1):
            HR[i-1] = Sub.NCon([HR[i],TA[i],Mpo[i],np.conj(TB[i])],[[1,3,5],[-1,2,1],[-2,4,2,3],[-3,4,5]])

    if Orient == 'lr':
        for i in range(Ns-1):
            HL[i+1] = Sub.NCon([HL[i],np.conj(TB[i]),Mpo[i],TA[i]],[[1,2,4],[1,3,-1],[2,3,5,-2],[4,5,-3]])

    return HL,HR

def OptTSite(Mpo,HL,HR,TA):
    TB = Sub.NCon([HL,TA,Mpo,HR],[[-1,2,1],[1,3,4],[2,-2,3,5],[4,5,-3]])
    Nom = LA.norm(TB)

    return TB,Nom

def OptT(Mpo,TA,Orient='rl',Prec=1.0e-10,icheck=0):
    Ns = len(TA)
    Dp = np.shape(TA[0])[1]
    Ds = np.shape(TA[0])[2]
    Nom0 = np.ones((Ns))
    Nom1 = np.ones((Ns))

    if icheck == 1:
        TB0 = [None]*Ns
        for i in range(Ns):
            TB0[i] = Sub.NCon([Mpo[i],TA[i]],[[-1,-3,1,-4],[-2,1,-5]])
            TB0[i] = Sub.Group(TB0[i],[[0,1],[2],[3,4]])
        NormAA = GetNormTT(TB0,TB0)
        print('NormAA=',NormAA)

    TB = copy.deepcopy(TA)
    HL,HR = InitHLR(Mpo,TA,TB,Orient=Orient)

    if icheck == 1:
        Norm0 = GetNormTT(TB0,TB)
        # print Norm0
        if Orient == 'rl':
            i = 0
        else:
            i = Ns-1
        Norm1 = Sub.NCon([HL[i],np.conj(TB[i]),Mpo[i],TA[i],HR[i]],[[1,2,4],[1,3,8],[2,3,5,7],[4,5,6],[6,7,8]])
        print('InitHLR',Norm1-Norm0)

    if Orient == 'rl':
        for r in range(1000):
            for i in range(Ns-1):
                TB[i],Nom1[i] = OptTSite(Mpo[i],HL[i],HR[i],TA[i])
                # print i,Nom1[i]
                TB[i],U = Sub.Mps_QR0P(TB[i])
                HL[i+1] = Sub.NCon([HL[i],np.conj(TB[i]),Mpo[i],TA[i]],[[1,2,4],[1,3,-1],[2,3,5,-2],[4,5,-3]])
                TB[i+1] = np.tensordot(U,TB[i+1],(1,0))

            for i in range(Ns-1,0,-1):
                TB[i],Nom1[i] = OptTSite(Mpo[i],HL[i],HR[i],TA[i])
                # print i,Nom1[i]
                U,TB[i] = Sub.Mps_LQ0P(TB[i])
                HR[i-1] = Sub.NCon([HR[i],TA[i],Mpo[i],np.conj(TB[i])],[[1,3,5],[-1,2,1],[-2,4,2,3],[-3,4,5]])
                TB[i-1] = np.tensordot(TB[i-1],U,(2,0))

            Err = Nom1[1]-Nom0[1]
            # print r,Nom1[1],Err
            if abs(Err) < Prec:
                break
            Nom0 = copy.copy(Nom1)

    if Orient == 'lr':
        for r in range(1000):
            for i in range(Ns-1,0,-1):
                TB[i],Nom1[i] = OptTSite(Mpo[i],HL[i],HR[i],TA[i])
                # print i,Nom1[i]
                U,TB[i] = Sub.Mps_LQ0P(TB[i])
                HR[i-1] = Sub.NCon([HR[i],TA[i],Mpo[i],np.conj(TB[i])],[[1,3,5],[-1,2,1],[-2,4,2,3],[-3,4,5]])
                TB[i-1] = np.tensordot(TB[i-1],U,(2,0))

            for i in range(Ns-1):
                TB[i],Nom1[i] = OptTSite(Mpo[i],HL[i],HR[i],TA[i])
                # print i,Nom1[i]
                TB[i],U = Sub.Mps_QR0P(TB[i])
                HL[i+1] = Sub.NCon([HL[i],np.conj(TB[i]),Mpo[i],TA[i]],[[1,2,4],[1,3,-1],[2,3,5,-2],[4,5,-3]])
                TB[i+1] = np.tensordot(U,TB[i+1],(1,0))

            Err = Nom1[1]-Nom0[1]
            # print r,Nom1[1],Err
            if abs(Err) < Prec:
                break
            Nom0 = copy.copy(Nom1)

    TB = NormalizeT(TB)

    if icheck == 1:
        NormBB = GetNormTT(TB,TB)
        print('NormBB =',NormBB)

    NormAB = GetNormTHT(TA,Mpo,TB)
    Err = 1.0-np.abs(NormAB)

    return TB,Err
#----------------------------------------------------------------
def CalOpSite(T,Type):
    Ns = len(T)
    Dp = np.shape(T[0])[1]
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
        TO = copy.deepcopy(T)
        TO[i] = Sub.NCon([TO[i],Op],[[-1,1,-3],[-2,1]])
        Val[i] = GetNormTT(TO,T)

    Val = np.real(Val)

    return Val

def Evolution(Ham,T,Ns,Ds,OpType,Zdt,Nt,Zt0,iid,save_dir,mode):
    x_filename = os.path.join(save_dir, 'data_{}_x'.format(iid))  
    y_filename = os.path.join(save_dir, 'data_{}_y'.format(iid))  

    N0 = Ns//2
    fmt = '%0.8f \t'*3 + '\n'

    Mpo = GetEvoMpo_Tebd(Ns,Ham,Zdt)

    result = []
    T = PrepareT(T)
    Mz = CalOpSite(T,OpType)

    for it in range(1,Nt):
        Zt = Zdt*it+Zt0
        T,Err = OptT(Mpo,T)
        data = TtoData(T)
        if mode == 'mps_wavefunction':
            result.append(np.array(data, dtype=np.complex64))
        elif mode == 'mps_magnetic':
            T=DatatoT(2,Ns,Ds,data)
            Mz = CalOpSite(T,OpType)
            result.append(Mz)
    if mode == 'mps_wavefunction':
        np.save(x_filename, np.array(result[:-1], dtype=np.complex64))
        np.save(y_filename, np.array(result[1:], dtype=np.complex64))
    elif mode == 'mps_magnetic':
        np.save(x_filename, np.array(result[:-1], dtype=np.float32))
        np.save(y_filename, np.array(result[1:], dtype=np.float32))
    elif mode == 'mps_magnetic_xyz':
        pass
    print("Generated {} sequence by MPS, shape: {}".format(iid, np.shape(result[:-1])))

    return None
