import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as LAs
from scipy import linalg
import itertools

"""

"""
#---------------------------------------------------------------------------
def Group(A,shapeA):
    """ transpose + reshape """
    dimA = np.asarray(np.shape(A))
    rankA = len(shapeA)

    shapeB = []
    for i in range(0,rankA):
        shapeB += [np.prod(dimA[shapeA[i]])]

    orderB = sum(shapeA,[])
    A = np.reshape(np.transpose(A,orderB),shapeB)
    return A

def NCon(Tensor,Index):
    ConList = range(1,max(sum(Index,[]))+1)

    while len(ConList) > 0:

        Icon = []
        for i in range(len(Index)):
            if ConList[0] in Index[i]:
                Icon.append(i)
                if len(Icon) == 2:
                    break

        if len(Icon) == 1:
            IndCommon = list(set([x for x in Index[Icon[0]] if Index[Icon[0]].count(x)>1]))

            for icom in range(len(IndCommon)):
                Pos = sorted([i for i,x in enumerate(Index[Icon[0]]) if x==IndCommon[icom]])
                Tensor[Icon[0]] = np.trace(Tensor[Icon[0]],axis1=Pos[0],axis2=Pos[1])
                Index[Icon[0]].pop(Pos[1])
                Index[Icon[0]].pop(Pos[0])

        else:
            IndCommon = list(set(Index[Icon[0]]) & set(Index[Icon[1]]))
            Pos = [[],[]]
            for i in range(2):
                for ind in range(len(IndCommon)):
                    Pos[i].append(Index[Icon[i]].index(IndCommon[ind]))
            A = np.tensordot(Tensor[Icon[0]],Tensor[Icon[1]],(Pos[0],Pos[1]))

            for i in range(2):
                for ind in range(len(IndCommon)):
                    Index[Icon[i]].remove(IndCommon[ind])
            Index[Icon[0]] = Index[Icon[0]]+Index[Icon[1]]
            Index.pop(Icon[1])
            Tensor[Icon[0]] = A
            Tensor.pop(Icon[1])

        ConList = list(set(ConList)^set(IndCommon))

    while len(Index) > 1:

        Tensor[0] = np.multiply.outer(Tensor[0],Tensor[1])
        Tensor.pop(1)
        Index[0] = Index[0]+Index[1]
        Index.pop(1)

    Index = Index[0]
    if len(Index) > 0:
        Order = sorted(range(len(Index)),key=lambda k:Index[k])[::-1]
        Tensor = np.transpose(Tensor[0],Order)
    else:
        Tensor = Tensor[0]

    return Tensor

def BaseN(num,b,numerals="0123456789abcdefghijklmnopqrstuvwxyz"):
    return ((num == 0) and numerals[0]) or (BaseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])

def SpinOper(ss):
    spin = (ss-1)/2.0
    dz = np.zeros(ss)
    mp = np.zeros(ss-1)

    for i in range(ss):
        dz[i] = spin-i
    for i in range(ss-1):
        mp[i] = np.sqrt((2*spin-i)*(i+1))

    S0 = np.eye(ss)
    Sp = np.diag(mp,1)
    Sm = np.diag(mp,-1)
    Sx = 0.5*(Sp+Sm)
    Sy = -0.5j*(Sp-Sm)
    Sz = np.diag(dz)

    return S0,Sp,Sm,Sz,Sx,Sy
#---------------------------------------------------------------------------
def GetMapT(Shape,Parity,PrintEq=0):
    N = len(Shape)

    StrX = "x0"
    for i in range(1,N):
        StrX += ",x"+str(i)
    # print StrX

    if Parity == 0:
        StrY = "1-np.mod("
        for i in range(N):
            StrY += "+x"+str(i)
        StrY += ",2)"
    if Parity == 1:
        StrY = "np.mod("
        for i in range(N):
            StrY += "+x"+str(i)
        StrY += ",2)"

    Fun = "def GetMap("+StrX+"):\n\treturn "+StrY
    if PrintEq == 1:
        print(Fun)
    exec(Fun)

    MapT = np.fromfunction(GetMap,Shape)

    return MapT

def GetFermiT(Shape,Type):
    if Type == 'R':
        T = np.random.rand(*Shape)
    if Type == 'C':
        T = np.random.rand(*Shape) + 1j*np.random.rand(*Shape)

    MapT = GetMapT(Shape,0)
    T = T*MapT

    return T

def GetSignT(T,Order,PrintEq=0):
    N = T.ndim

    StrX = "x0"
    for i in range(1,N):
        StrX += ",x"+str(i)
    # print StrX

    StrY = "(-1)**np.mod(0"
    Ind = range(N)
    for i,Op1 in enumerate(Order):
        i1 = Ind.index(Op1)
        if i1 > i:
            StrY += "+x"+str(Op1)+"*("
            for j in range(i,i1):
                Op2 = Ind[j]
                StrY += "+x"+str(Op2)
            StrY += ")"
            Ind.remove(Op1)
            Ind.insert(i,Op1)
    StrY += ",2)"
    # print StrY

    Fun = "def GetSign("+StrX+"):\n\treturn "+StrY
    if PrintEq == 1:
        print(Fun)
    exec(Fun)

    SignT = np.fromfunction(GetSign,np.shape(T))

    return SignT

def GetSignTT(T,Tf,Order,M,PrintEq=0):
    N = T.ndim

    StrX = "x0"
    for i in range(1,N):
        StrX += ",x"+str(i)
    # print StrX

    StrY = "(-1)**np.mod(0"
    for i,Op1 in enumerate(Order[-M:]):
        if Tf[Op1] == 1:
            StrY += "+x"+str(Op1)
    StrY += ",2)"
    # print StrY

    Fun = "def GetSign("+StrX+"):\n\treturn "+StrY
    if PrintEq == 1:
        print(Fun)
    exec(Fun)

    SignTT = np.fromfunction(GetSign,np.shape(T))

    return SignTT

def Ftranspose(T,Tf,Order,PrintEq=0):
    SignT = GetSignT(T,Order,PrintEq=PrintEq)
    T1 = np.transpose(T*SignT,Order)
    Tf1 = Tf[Order]
    return T1,Tf1

def Fconjtranspose(T,Tf):
    N = T.ndim
    Order = range(N)[::-1]
    T1 = np.transpose(np.conj(T),Order)
    Tf1 = Tf[Order]
    return T1,Tf1

def Ftensordot(TA,TAf,TB,TBf,ConList,PrintEq=0):
    NA = TA.ndim
    NB = TB.ndim
    IndA = range(NA)
    IndB = range(NB)

    try:
        M = len(ConList[0])
        # print M,ConList[0],ConList[1]
        for i in range(M):
            if TAf[ConList[0][i]]+TBf[ConList[1][i]] != 1:
                print('Error Ftensordot')

        for i in range(M-1,-1,-1):
            Op = ConList[0][i]
            IndA.remove(Op)
            IndA.append(Op)
        # print 'IndA',IndA

        for i in range(M-1,-1,-1):
            Op = ConList[1][i]
            IndB.remove(Op)
            IndB.insert(0,Op)
        # print 'IndB',IndB

        TTf = np.append(TAf[IndA[:NA-M]],TBf[IndB[M:]])
        # print 'TTf',TTf

    except TypeError as Err:
        M = 1
        # print M,ConList[0],ConList[1]
        if TAf[ConList[0]]+TBf[ConList[1]] != 1:
            print('Error Ftensordot')

        Op = ConList[0]
        IndA.remove(Op)
        IndA.append(Op)
        # print 'IndA',IndA

        Op = ConList[1]
        IndB.remove(Op)
        IndB.insert(0,Op)
        # print 'IndB',IndB

        TTf = np.append(TAf[IndA[:NA-1]],TBf[IndB[1:]])
        # print 'TTf',TTf

    SignTA = GetSignT(TA,IndA,PrintEq=PrintEq)
    SignTB = GetSignT(TB,IndB,PrintEq=PrintEq)
    SignTT = GetSignTT(TA,TAf,IndA,M,PrintEq=PrintEq)
    TT = np.tensordot(TA*SignTA*SignTT,TB*SignTB,ConList)

    return TT,TTf

def FNCon(Tensor,Tensorf,Index,PrintEq=0):
    ConList = range(1,max(sum(Index,[]))+1)

    while len(ConList) > 0:

        Icon = []
        for i in range(len(Index)):
            if ConList[0] in Index[i]:
                Icon.append(i)
                if len(Icon) == 2:
                    break

        if len(Icon) == 1:
            print("Error: no self trace")

        else:
            IndCommon = list(set(Index[Icon[0]]) & set(Index[Icon[1]]))
            Pos = [[],[]]
            for i in range(2):
                for ind in range(len(IndCommon)):
                    Pos[i].append(Index[Icon[i]].index(IndCommon[ind]))
            A,Af = Ftensordot(Tensor[Icon[0]],Tensorf[Icon[0]],Tensor[Icon[1]],Tensorf[Icon[1]],(Pos[0],Pos[1]),PrintEq=PrintEq)

            for i in range(2):
                for ind in range(len(IndCommon)):
                    Index[Icon[i]].remove(IndCommon[ind])
            Index[Icon[0]] = Index[Icon[0]]+Index[Icon[1]]
            Index.pop(Icon[1])
            Tensor[Icon[0]] = A
            Tensor.pop(Icon[1])
            Tensorf[Icon[0]] = Af
            Tensorf.pop(Icon[1])

        ConList = list(set(ConList)^set(IndCommon))

    while len(Index) > 1:

        Tensor[0] = np.multiply.outer(Tensor[0],Tensor[1])
        Tensor.pop(1)
        Tensorf[0] = np.append(Tensorf[0],Tensorf[1])
        Tensorf.pop(1)
        Index[0] = Index[0]+Index[1]
        Index.pop(1)

    Index = Index[0]
    if len(Index) > 0:
        Order = sorted(range(len(Index)),key=lambda k:Index[k])[::-1]
        Tensor,Tensorf = Ftranspose(Tensor[0],Tensorf[0],Order,PrintEq=PrintEq)
    else:
        Tensor = Tensor[0]
        Tensorf = Tensorf[0]

    return Tensor,Tensorf

def GetMix(D):
    if np.mod(D,2) == 0:
        Mix = np.zeros((D,D,D**2))
        for ix in range(D):
            for iy in range(D):
                iz = (ix*(D/2)+iy/2)*2+np.mod(ix+iy,2)
                Mix[ix,iy,iz] = 1.0
    else:
        Mix = np.eye(D**2)
        Mix = np.reshape(Mix,[D,D,D**2])

    return Mix
#---------------------------------------------------------------------------
def OutputT(T,prec=1.0e-10):
    """ output nonzero elements """
    pos = np.nonzero(np.abs(T)>prec)
    val = T[pos]
    pos = np.transpose(pos)
    for i in range(len(val)):
        print(pos[i],val[i])

def OutputTFile(file,T,prec=1.0e-10):
    """ output nonzero elements to file """
    pos = np.nonzero(np.abs(T)>prec)
    val = T[pos]
    pos = np.transpose(pos)
    val = np.reshape(val,[len(val),1])

    if LA.norm(np.imag(T)) > prec:
        out = np.concatenate((pos+1,np.real(val),np.imag(val)),axis=1)
        np.savetxt(file,out,fmt = '%d\t'*np.shape(pos)[1] + '%0.6f \t %0.6f')
    else:
        out = np.concatenate((pos,np.real(val)),axis=1)
        np.savetxt(file,out,fmt = '%d\t'*np.shape(pos)[1] + '%0.6f')
#---------------------------------------------------------------------------
def GetIndexEO(Shape):
    Dall = np.prod(Shape)
    Ind = [None]*2
    Ind[0] = np.nonzero(np.reshape(GetMapT(Shape,0),[Dall]))[0]
    Ind[1] = np.nonzero(np.reshape(GetMapT(Shape,1),[Dall]))[0]
    return Ind

def GetEO(T,Igroup):
    N = T.ndim
    DT = np.shape(T)

    if Igroup == 0:
        Dall = np.prod(DT)
        Tall = np.reshape(T,Dall)

        Ind = GetIndexEO(DT)

        TEO = [None]*2
        for i in range(2):
            TEO[i] = Tall[Ind[i]]

    if (Igroup > 0) & (Igroup < N):
        Dall = [np.prod(DT[slice(0,Igroup)]),np.prod(DT[slice(Igroup,N)])]
        Tall = np.reshape(T,Dall)

        Ind = [None]*2
        Ind[0] = GetIndexEO(DT[slice(0,Igroup)])
        Ind[1] = GetIndexEO(DT[slice(Igroup,N)])

        IndM = [None]*4
        IndM = np.reshape(IndM,[2,2])
        for i in range(2):
            for j in range(2):
                IndM[i,j] = np.meshgrid(Ind[0][i],Ind[1][j])

        TEO = [None]*4
        TEO = np.reshape(TEO,[2,2])
        for i in range(2):
            for j in range(2):
                TEO[i,j] = np.transpose(Tall[IndM[i,j][0],IndM[i,j][1]])
                # print i,j,np.shape(TEO[i,j]),TEO[i,j]

    return TEO,Ind

def BackEO(TEO,Ind,DT,Igroup):
    N = len(DT)

    if Igroup == 0:
        Dall = np.prod(DT)

        if np.any(np.imag(TEO[0])) | np.any(np.imag(TEO[1])):
            T = np.zeros((Dall),dtype=np.complex)
        else:
            T = np.zeros(Dall)

        for i in range(2):
            T[Ind[i]] = np.copy(TEO[i])

    if (Igroup > 0) & (Igroup < N):
        Dall = [np.prod(DT[slice(0,Igroup)]),np.prod(DT[slice(Igroup,N)])]

        if np.any(np.imag(TEO[0,0])) | np.any(np.imag(TEO[0,1])) | np.any(np.imag(TEO[1,0])) | np.any(np.imag(TEO[1,1])):
            T = np.zeros((Dall),dtype=np.complex)
        else:
            T = np.zeros(Dall)

        IndM = [None]*4
        IndM = np.reshape(IndM,[2,2])
        for i in range(2):
            for j in range(2):
                IndM[i,j] = np.meshgrid(Ind[0][i],Ind[1][j])

        for i in range(2):
            for j in range(2):
                T[IndM[i,j][0],IndM[i,j][1]] = np.transpose(TEO[i,j])

    T = np.reshape(T,DT)

    return T
#---------------------------------------------------------------------------
def SplitEig(A,Dcut,prec=1.0e-12,safe=3):
    """ Eig, Lapack + Arpack """
    D = np.min(np.shape(A))

    if Dcut >= D-1:
        S,V = LA.eig(A)
    else:
        k_ask = min(Dcut+safe,D-2)
        S,V = LAs.eigs(A,k=k_ask)

    idx = np.isfinite(S)
    if idx.any() != False:
        S = S[idx]
        V = V[:,idx]
    order = np.argsort(abs(S))[::-1]
    S = S[order]
    V = V[:,order]

    S = S[abs(S/S[0])>prec]
    Dc = min(len(S),Dcut)
    S = S[:Dc]
    V = V[:,:Dc]

    return V,S,Dc

def SplitEigh(A,Dcut,mode='P',prec=1.0e-12,safe=3,icheck=0):
    """ Eig, Hermitian, Lapack + Arpack """
    """ P = perserve, C = cut """
    D = np.min(np.shape(A))

    if Dcut >= D-1:
        S,V = LA.eigh(A)
    else:
        k_ask = min(Dcut+safe,D-2)
        S,V = LAs.eigsh(A,k=k_ask)

    idx = np.isfinite(S)
    if idx.any() != False:
        S = S[idx]
        V = V[:,idx]
    order = np.argsort(abs(S))[::-1]
    S = S[order]
    V = V[:,order]

    if mode == 'P':
        S = S[abs(S/S[0])>1.0e-15]
        Dc = min(len(S),Dcut)
        S = S[:Dc]
        V = V[:,:Dc]
        S[abs(S/S[0])<prec] = prec*np.sign(S[abs(S/S[0])<prec])

    if mode == 'C':
        S = S[abs(S/S[0])>prec]
        Dc = min(len(S),Dcut)
        S = S[:Dc]
        V = V[:,:Dc]

    if icheck == 1:
        print(LA.norm(np.dot(V,np.dot(np.diag(S),np.transpose(np.conj(V))))-A)/LA.norm(A))

    return V,S,Dc

def SplitEigh_Deg(A,Dcut,mode='P',prec=1.0e-12,tol=1.0e-11,iadd0=10,safe=3,icheck=0):
    """ Eig, Hermitian, Lapack + Arpack, keep all degenerate states """
    """ P = perserve, C = cut """
    D = np.min(np.shape(A))
    ienough = 0
    iadd = iadd0

    while ienough == 0:
        if Dcut+iadd >= D-1:
            S,V = LA.eigh(A)
        else:
            k_ask = min(Dcut+iadd+safe,D-2)
            S,V = LAs.eigsh(A,k=k_ask)

        idx = np.isfinite(S)
        if idx.any() != False:
            S = S[idx]
            V = V[:,idx]
        order = np.argsort(abs(S))[::-1]
        S = S[order]
        V = V[:,order]

        if mode == 'P':
            S = S[abs(S/S[0])>1.0e-15]
        if mode == 'C':
            S = S[abs(S/S[0])>prec]
        Dc = len(S)
        S = S[:Dc]
        V = V[:,:Dc]

        if Dc <= Dcut:
            ienough = 1
        else:
            Sadd = S[Dcut:]
            idx = abs(abs(Sadd)-abs(S[Dcut-1]))/abs(S[0]) < tol
            idx = np.append([True]*Dcut,idx)
            S = S[idx]
            V = V[:,idx]
            Dc = len(S)

            if Dc < Dcut+iadd:
                ienough = 1
            else:
                iadd += iadd0

    if mode == 'P':
        S[abs(S/S[0])<prec] = prec*np.sign(S[abs(S/S[0])<prec])

    if icheck == 1:
        print(LA.norm(np.dot(V,np.dot(np.diag(S),np.transpose(np.conj(V))))-A)/LA.norm(A))

    return V,S,Dc

def SplitEigh_Z2Deg(Ae,Ao,Dcut,appro='T',mode='P',prec=1.0e-12,tol=1.0e-11,iadd0=10,safe=3,icheck=0):
    """ Eig, Hermitian, Lapack + Arpack, keep all degenerate states """
    """ appro: T = together, S = seperate """
    """ P = perserve, C = cut """

    if appro == 'S':
        Ve,Se,Dce = SplitEigh_Deg(Ae,(Dcut+1)/2,mode=mode,prec=prec,tol=tol,iadd0=iadd0,safe=safe,icheck=icheck)
        Vo,So,Dco = SplitEigh_Deg(Ao,(Dcut+1)/2,mode=mode,prec=prec,tol=tol,iadd0=iadd0,safe=safe,icheck=icheck)

    if appro == 'T':
        Ve,Se,Dce = SplitEigh_Deg(Ae,(Dcut+1)/2+iadd0,mode=mode,prec=prec,tol=tol,iadd0=iadd0,safe=safe,icheck=icheck)
        Vo,So,Dco = SplitEigh_Deg(Ao,(Dcut+1)/2+iadd0,mode=mode,prec=prec,tol=tol,iadd0=iadd0,safe=safe,icheck=icheck)
        Sall = np.append(Se,So)
        order = np.argsort(np.abs(Sall))[::-1]
        Sall = Sall[order]

        if len(Sall) > Dcut:
            S0 = max(Se[0],So[0])
            Scut = Sall[Dcut-1]
            idxe =~ ((abs(Se)-abs(Scut))/abs(S0) < -tol)
            idxo =~ ((abs(So)-abs(Scut))/abs(S0) < -tol)
            Se = Se[idxe]
            So = So[idxo]
            Ve = Ve[:,idxe]
            Vo = Vo[:,idxo]
            Dce = len(Se)
            Dco = len(So)

    return Ve,Vo,Se,So,Dce,Dco

def SplitEigh_Z2Ten(T,Dcut,Igroup,appro='T',mode='P',prec=1.0e-12,tol=1.0e-11,iadd0=10,safe=3,icheck=0):
    N = T.ndim
    DT = np.shape(T)
    TEO,Ind = GetEO(T,Igroup)

    Ve,Vo,Se,So,Dce,Dco = SplitEigh_Z2Deg(TEO[0,0],TEO[1,1],Dcut,appro=appro,mode=mode,prec=prec,tol=tol,iadd0=iadd0,safe=safe)
    Dc = max(Dce*2-1,Dco*2)
    # print Dce,Dco,Dc

    if np.any(np.imag(T)):
        V = np.zeros((np.prod(DT[slice(0,Igroup)]),Dc),dtype=np.complex)
    else:
        V = np.zeros((np.prod(DT[slice(0,Igroup)]),Dc))
    S = np.zeros(Dc)

    V[Ind[0][0],0:Dce*2-1:2] = Ve
    V[Ind[0][1],1:Dco*2:2] = Vo
    S[0:Dce*2-1:2] = Se
    S[1:Dco*2:2] = So

    V = np.reshape(V,list(DT[slice(0,Igroup)])+[Dc])

    if icheck == 1:
        A = NCon([V,np.diag(S),np.conj(V)],[list(-np.arange(1,Igroup+1))+[1],[1,2],list(-np.arange(Igroup+1,N+1))+[2]])
        print(LA.norm(A-T)/LA.norm(T))

    return V,S,Dc
#---------------------------------------------------------------------------
def SplitSvd_Lapack(A,Dcut,iweight,mode='P',prec=1.0e-12,icheck=0):
    """ SVD, Lapack only """
    """ P = perserve, C = cut """

    U,S,V = LA.svd(A,full_matrices=0)

    S = np.abs(S)
    idx = np.isfinite(S)
    if idx.any() != False:
        S = S[idx]
        U = U[:,idx]
        V = V[idx,:]
    order = np.argsort(S)[::-1]
    S = S[order]
    U = U[:,order]
    V = V[order,:]

    if mode == 'P':
        S = S[S/S[0]>1.0e-15]
        Dc = min(len(S),Dcut)
        S = S[:Dc]
        U = U[:,:Dc]
        V = V[:Dc,:]
        S[S/S[0]<prec] = prec

    if mode == 'C':
        S = S[S/S[0]>prec]
        Dc = min(len(S),Dcut)
        S = S[:Dc]
        U = U[:,:Dc]
        V = V[:Dc,:]

    if iweight == 1:
        U = np.dot(U,np.diag(np.sqrt(S)))
        V = np.dot(np.diag(np.sqrt(S)),V)

    if icheck == 1:
        if iweight == 0:
            print(LA.norm(np.dot(U,np.dot(np.diag(S),V))-A)/LA.norm(A))
        else:
            print(LA.norm(np.dot(U,V)-A)/LA.norm(A))

    return U,S,V,Dc
#---------------------------------------------------------------------------
def SplitSvd(A,Dcut,iweight,mode='P',prec=1.0e-12,safe=3,icheck=0):
    """ SVD, Lapack + Arpack """
    """ P = perserve, C = cut """
    D = np.min(np.shape(A))

    if Dcut >= D-1:
        U,S,V = LA.svd(A,full_matrices=0)
    else:
        k_ask = min(Dcut+safe,D-2)
        U,S,V = LAs.svds(A,k=k_ask)

    S = np.abs(S)
    idx = np.isfinite(S)
    if idx.any() != False:
        S = S[idx]
        U = U[:,idx]
        V = V[idx,:]
    order = np.argsort(S)[::-1]
    S = S[order]
    U = U[:,order]
    V = V[order,:]

    if mode == 'P':
        S = S[S/S[0]>1.0e-15]
        Dc = min(len(S),Dcut)
        S = S[:Dc]
        U = U[:,:Dc]
        V = V[:Dc,:]
        S[S/S[0]<prec] = prec

    if mode == 'C':
        S = S[S/S[0]>prec]
        Dc = min(len(S),Dcut)
        S = S[:Dc]
        U = U[:,:Dc]
        V = V[:Dc,:]

    if iweight == 1:
        U = np.dot(U,np.diag(np.sqrt(S)))
        V = np.dot(np.diag(np.sqrt(S)),V)

    if icheck == 1:
        if iweight == 0:
            print(LA.norm(np.dot(U,np.dot(np.diag(S),V))-A)/LA.norm(A))
        else:
            print(LA.norm(np.dot(U,V)-A)/LA.norm(A))

    return U,S,V,Dc

def SplitSvd_Deg(A,Dcut,iweight,mode='P',prec=1.0e-12,tol=1.0e-11,iadd0=10,safe=3,icheck=0):
    """ SVD, Lapack + Arpack, keep all degenerate states """
    """ P = perserve, C = cut """
    D = np.min(np.shape(A))
    ienough = 0
    iadd = iadd0

    while ienough == 0:
        if Dcut+iadd >= D-1:
            U,S,V = LA.svd(A,full_matrices=0)
        else:
            k_ask = min(Dcut+iadd+safe,D-2)
            U,S,V = LAs.svds(A,k=k_ask)

        S = np.abs(S)
        idx = np.isfinite(S)
        if idx.any() != False:
            S = S[idx]
            U = U[:,idx]
            V = V[idx,:]
        order = np.argsort(S)[::-1]
        S = S[order]
        U = U[:,order]
        V = V[order,:]

        if mode == 'P':
            S = S[S/S[0]>1.0e-15]
        if mode == 'C':
            S = S[S/S[0]>prec]
        Dc = len(S)
        S = S[:Dc]
        U = U[:,:Dc]
        V = V[:Dc,:]

        if Dc <= Dcut:
            ienough = 1
        else:
            Sadd = S[Dcut:]
            idx = abs(Sadd-S[Dcut-1])/S[0] < tol
            idx = np.append([True]*Dcut,idx)
            S = S[idx]
            U = U[:,idx]
            V = V[idx,:]
            Dc = len(S)

            if Dc < Dcut+iadd:
                ienough = 1
            else:
                iadd += iadd0

    if mode == 'P':
        S[S/S[0]<prec] = prec

    if iweight == 1:
        U = np.dot(U,np.diag(np.sqrt(S)))
        V = np.dot(np.diag(np.sqrt(S)),V)

    if icheck == 1:
        if iweight == 0:
            print(LA.norm(np.dot(U,np.dot(np.diag(S),V))-A)/LA.norm(A))
        else:
            print(LA.norm(np.dot(U,V)-A)/LA.norm(A))

    return U,S,V,Dc

def SplitSvd_Z2Deg(Ae,Ao,Dcut,iweight,appro='T',mode='P',prec=1.0e-12,tol=1.0e-11,iadd0=10,safe=3,icheck=0):
    """ SVD, Lapack + Arpack, keep all degenerate states """
    """ appro: T = together, S = seperate """
    """ mode: P = perserve, C = cut """

    if appro == 'S':
        if LA.norm(Ae) > prec:
            Ue,Se,Ve,Dce = SplitSvd_Deg(Ae,(Dcut+1)/2,iweight,mode=mode,prec=prec,tol=tol,iadd0=iadd0,safe=safe,icheck=icheck)
        else:
            Ue = []
            Se = []
            Ve = []
            Dce = 0
        if LA.norm(Ao) > prec:
            Uo,So,Vo,Dco = SplitSvd_Deg(Ao,(Dcut+1)/2,iweight,mode=mode,prec=prec,tol=tol,iadd0=iadd0,safe=safe,icheck=icheck)
        else:
            Uo = []
            So = []
            Vo = []
            Dco = 0

    if appro == 'T':
        if LA.norm(Ae) > prec:
            Ue,Se,Ve,Dce = SplitSvd_Deg(Ae,(Dcut+1)/2+iadd0,iweight,mode=mode,prec=prec,tol=tol,iadd0=iadd0,safe=safe,icheck=icheck)
        else:
            Ue = []
            Se = []
            Ve = []
            Dce = 0
        if LA.norm(Ao) > prec:
            Uo,So,Vo,Dco = SplitSvd_Deg(Ao,(Dcut+1)/2+iadd0,iweight,mode=mode,prec=prec,tol=tol,iadd0=iadd0,safe=safe,icheck=icheck)
        else:
            Uo = []
            So = []
            Vo = []
            Dco = 0

        Sall = np.sort(np.append(Se,So))[::-1]

        if len(Sall) > Dcut:
            if len(Se) == 0:
                S0 = So[0]
            if len(So) == 0:
                S0 = Se[0]
            if (len(Se) > 0) & (len(So) > 0):
                S0 = max(Se[0],So[0])
            Scut = Sall[Dcut-1]
            idxe =~ ((Se-Scut)/S0 < -tol)
            idxo =~ ((So-Scut)/S0 < -tol)
            if len(idxe) > 0:
                Se = Se[idxe]
                Ue = Ue[:,idxe]
                Ve = Ve[idxe,:]
                Dce = len(Se)
            if len(idxo) > 0:
                So = So[idxo]
                Uo = Uo[:,idxo]
                Vo = Vo[idxo,:]
                Dco = len(So)

    return Ue,Uo,Se,So,Ve,Vo,Dce,Dco

def SplitSvd_Z2Ten(T,Dcut,iweight,Igroup,appro='T',mode='P',prec=1.0e-12,tol=1.0e-11,iadd0=10,safe=3,icheck=0):
    N = T.ndim
    DT = np.shape(T)
    TEO,Ind = GetEO(T,Igroup)

    Ue,Uo,Se,So,Ve,Vo,Dce,Dco = SplitSvd_Z2Deg(TEO[0,0],TEO[1,1],Dcut,iweight,appro=appro,mode=mode,prec=prec,tol=tol,iadd0=iadd0,safe=safe)
    Dc = max(Dce*2-1,Dco*2)
    # print Dce,Dco,Dc

    if np.any(np.imag(T)):
        U = np.zeros((np.prod(DT[slice(0,Igroup)]),Dc),dtype=np.complex)
        V = np.zeros((Dc,np.prod(DT[slice(Igroup,N)])),dtype=np.complex)
    else:
        U = np.zeros((np.prod(DT[slice(0,Igroup)]),Dc))
        V = np.zeros((Dc,np.prod(DT[slice(Igroup,N)])))
    S = np.zeros(Dc)

    if Dce > 0:
        U[Ind[0][0],0:Dce*2-1:2] = Ue
        V[0:Dce*2-1:2,Ind[1][0]] = Ve
        S[0:Dce*2-1:2] = Se
    if Dco > 0:
        U[Ind[0][1],1:Dco*2:2] = Uo
        V[1:Dco*2:2,Ind[1][1]] = Vo
        S[1:Dco*2:2] = So

    U = np.reshape(U,list(DT[slice(0,Igroup)])+[Dc])
    V = np.reshape(V,[Dc]+list(DT[slice(Igroup,N)]))

    if icheck == 1:
        if iweight == 0:
            A = NCon([U,np.diag(S),V],[list(-np.arange(1,Igroup+1))+[1],[1,2],[2]+list(-np.arange(Igroup+1,N+1))])
        else:
            A = NCon([U,V],[list(-np.arange(1,Igroup+1))+[1],[1]+list(-np.arange(Igroup+1,N+1))])
        print(LA.norm(A-T)/LA.norm(T))

    return U,S,V,Dc
#---------------------------------------------------------------------------
def Mps_QRP(UL,T,icheck=0):
    """ (0-UL-1)(0-T-L) -> (0-Tnew-L)(0-UR-1) """
    shapeT = np.asarray(np.shape(T))
    rankT = len(shapeT)

    A = np.tensordot(UL,T,(1,0))
    A = np.reshape(A,[np.prod(shapeT[:-1]),shapeT[-1]])
    Tnew,UR = linalg.qr(A,mode = 'economic')
    Sign = np.diag(np.sign(np.diag(UR)))
    Tnew = np.dot(Tnew,Sign)
    UR = np.dot(Sign,UR)
    Tnew = np.reshape(Tnew,shapeT)

    if icheck == 1:
        A = np.reshape(A,shapeT)
        B = np.tensordot(Tnew,UR,(rankT-1,0))
        print(LA.norm(A-B)/LA.norm(A))
        A = np.tensordot(np.conj(Tnew),Tnew,(range(0,rankT-1),range(0,rankT-1)))
        print(LA.norm(A-np.eye(shapeT[-1])))

    return Tnew,UR

def Mps_LQP(T,UR,icheck=0):
    """ (0-T-L)(0-UR-1) -> (0-UL-1)(0-Tnew-L) """
    shapeT = np.asarray(np.shape(T))
    rankT = len(shapeT)

    A = np.tensordot(T,UR,(rankT-1,0))
    A = np.reshape(A,[shapeT[0],np.prod(shapeT[1:])])
    UL,Tnew = linalg.rq(A,mode = 'economic')
    Sign = np.diag(np.sign(np.diag(UL)))
    UL = np.dot(UL,Sign)
    Tnew = np.dot(Sign,Tnew)
    Tnew = np.reshape(Tnew,shapeT)

    if icheck == 1:
        A = np.reshape(A,shapeT)
        B = np.tensordot(UL,Tnew,(1,0))
        print(LA.norm(A-B)/LA.norm(A))
        A = np.tensordot(np.conj(Tnew),Tnew,(range(1,rankT),range(1,rankT)))
        print(LA.norm(A-np.eye(shapeT[0])))

    return UL,Tnew

def Mps_QR0P(T,icheck=0):
    """ (0-T-L) -> (0-Tnew-L)(0-UR-1) """
    shapeT = np.asarray(np.shape(T))
    rankT = len(shapeT)

    A = np.reshape(T,[np.prod(shapeT[:-1]),shapeT[-1]])
    Tnew,UR = linalg.qr(A,mode = 'economic')
    Sign = np.diag(np.sign(np.diag(UR)))
    Tnew = np.dot(Tnew,Sign)
    UR = np.dot(Sign,UR)
    Tnew = np.reshape(Tnew,shapeT)

    if icheck == 1:
        B = np.tensordot(Tnew,UR,(rankT-1,0))
        print(LA.norm(T-B)/LA.norm(T))
        A = np.tensordot(np.conj(Tnew),Tnew,(range(0,rankT-1),range(0,rankT-1)))
        print(LA.norm(A-np.eye(shapeT[-1])))

    return Tnew,UR

def Mps_LQ0P(T,icheck=0):
    """ (0-T-L) -> (0-UL-1)(0-Tnew-L) """
    shapeT = np.asarray(np.shape(T))
    rankT = len(shapeT)

    A = np.reshape(T,[shapeT[0],np.prod(shapeT[1:])])
    UL,Tnew = linalg.rq(A,mode = 'economic')
    Sign = np.diag(np.sign(np.diag(UL)))
    UL = np.dot(UL,Sign)
    Tnew = np.dot(Sign,Tnew)
    Tnew = np.reshape(Tnew,shapeT)

    if icheck == 1:
        B = np.tensordot(UL,Tnew,(1,0))
        print(LA.norm(T-B)/LA.norm(T))
        A = np.tensordot(np.conj(Tnew),Tnew,(range(1,rankT),range(1,rankT)))
        print(LA.norm(A-np.eye(shapeT[0])))

    return UL,Tnew
#---------------------------------------------------------------------------
if __name__ == "__main__":
    Test = {}
    Test['Group'] = 0
    Test['NCon'] = 0
    Test['CheckEO'] = 0
    Test['SplitEig'] = 0
    Test['SplitSvd'] = 0
    Test['Mps_QRP'] = 0

    if Test['Group'] == 1:
        A = np.random.rand(3,2,4,3)
        A = Group(A,[[0,2],[3,1]])
        print(A)

    if Test['NCon'] == 1:
        T1 = np.random.rand(2,3,4,3)
        T2 = np.random.rand(4,2,3,5)
        T3 = np.random.rand(3,3,2,4)

        A = np.tensordot(T1,T2,([0,3],[1,2]))
        A = np.transpose(A,[2,0,3,1])
        B = NCon([T1,T2],[[1,-2,-4,2],[-1,1,2,-3]])
        print(LA.norm(A-B)/LA.norm(A))

        A = np.tensordot(T1,T2,(3,2))
        A = np.transpose(A,[1,3,4,2,0,5])
        B = NCon([T1,T2],[[-5,-1,-4,1],[-2,-3,1,-6]])
        print(LA.norm(A-B)/LA.norm(A))

        A = np.tensordot(T1,T2,([0,3],[1,2]))
        A = np.tensordot(A,T3,([0,1],[1,3]))
        A = np.transpose(A,[3,0,1,2])
        B = NCon([T1,T2,T3],[[1,3,4,2],[-2,1,2,-3],[-4,3,-1,4]])
        print(LA.norm(A-B)/LA.norm(A))

        T1 = np.random.rand(2,3)
        T2 = np.random.rand(3,4)
        T3 = np.random.rand(4,2)

        A = np.kron(T1,T2)
        A = np.reshape(A,[2,3,3,4])
        B = NCon([T1,T2],[[-1,-3],[-2,-4]])
        print(LA.norm(A-B)/LA.norm(A))

        A = np.kron(T1,T2)
        A = np.kron(A,T3)
        A = np.reshape(A,[2,3,4,3,4,2])
        A = np.transpose(A,[1,0,5,4,2,3])
        B = NCon([T1,T2,T3],[[-2,-6],[-1,-4],[-5,-3]])
        print(LA.norm(A-B)/LA.norm(A))

        T = np.random.rand(3,2,4,3,2)

        A = np.eye(3)
        A = np.tensordot(T,A,([0,3],[0,1]))
        A = np.transpose(A,[2,0,1])
        B = NCon([T],[[1,-2,-3,1,-1]])
        print(LA.norm(A-B)/LA.norm(A))

        A = np.eye(3*2)
        A = np.reshape(A,[3,2,3,2])
        A = np.tensordot(T,A,([0,1,3,4],[0,1,2,3]))
        B = NCon([T],[[1,2,-1,1,2]])
        print(LA.norm(A-B)/LA.norm(A))

        T1 = np.random.rand(2,2)
        T2 = np.random.rand(2,2)
        T3 = np.random.rand(2,2)

        A = np.tensordot(T1,T2,(1,0))
        A = np.kron(A,T3)
        A = np.reshape(A,[2,2,2,2])
        B = NCon([T1,T2,T3],[[-1,1],[1,-3],[-2,-4]])
        print(LA.norm(A-B)/LA.norm(A))

        T1 = np.random.rand(3,3)
        T2 = np.random.rand(3,1,3)
        T3 = np.random.rand(3,3,1)
        T4 = np.random.rand(3,3,3,3)

        A = np.tensordot(T1,T2,(1,0))
        A = np.tensordot(A,T3,(1,2))
        A = np.tensordot(A,T4,([0,1,2,3],[0,1,2,3]))
        B = NCon([T1,T2,T3,T4],[[3,1],[1,2,4],[5,6,2],[3,4,5,6]])
        print(LA.norm(A-B)/LA.norm(A))

    if Test['CheckEO'] == 1:
        T = np.random.rand(3,4,2) + 1j*np.random.rand(3,4,2)
        TEO,Ind = GetEO(T,0)
        T1 = BackEO(TEO,Ind,np.shape(T),0)
        print(LA.norm(T1-T))

        T = np.random.rand(3,3,2,4,2,2) + 1j*np.random.rand(3,3,2,4,2,2)
        Igroup = 2
        DT = np.shape(T)
        TEO,Ind = GetEO(T,Igroup)
        T1 = BackEO(TEO,Ind,DT,Igroup)
        print(LA.norm(T1-T))

    if (Test['SplitEig'] == 1) | (Test['SplitSvd'] == 1):
        T = np.zeros((3,3,3))
        T[1,1,1] = 1
        T[1,0,2] = 1
        T[0,2,1] = 1
        T[2,1,0] = 1
        T[1,2,0] = -1
        T[2,0,1] = -1
        T[0,1,2] = -1

        G = np.zeros((3,2,3))
        G[1,1,0] = 1
        G[0,1,1] = 1
        G[1,0,2] = 1
        G[2,0,1] = 1

        A = np.tensordot(T,G,(0,0))
        A = np.tensordot(A,G,(0,0))
        A = np.tensordot(A,G,(0,0))
        A = np.tensordot(A,np.conj(A),([0,2,4],[0,2,4]))
        STA = Group(A,[[0,3],[1,4],[2,5]])

        A = np.reshape(T,[1,3,3,3])
        A = np.tensordot(A,np.conj(A),(0,0))
        STB = Group(A,[[0,3],[1,4],[2,5]])

        A = np.tensordot(STA,STB,([0],[0]))
        ST0 = np.transpose(A,[3,0,1,2])

    if Test['SplitEig'] == 1:
        ST = Group(ST0,[[3,0],[2,1]])
        STh = (ST+np.transpose(ST))/2
        ST0h = np.reshape(STh,[9,9,9,9])
        Dfull = np.shape(ST)[0]
        print(Dfull)

        print('full eig')
        V,S,Dc = SplitEig(STh,Dfull)
        Sfull = S
        print(Sfull,len(Sfull))

        print('partial eig')
        for Dcut in range(1,Dfull+1):
            V,S,Dc = SplitEig(STh,Dcut)
            print(Dcut,Dc,LA.norm(abs(S)-abs(Sfull[:Dc])))

        print('full eigh')
        V,S,Dc = SplitEigh(STh,Dfull,icheck=1)
        Sfull = S
        print(Sfull,len(Sfull))

        print('partial eigh')
        for Dcut in range(1,Dfull+1):
            V,S,Dc = SplitEigh(STh,Dcut,icheck=1)
            print(Dcut,Dc,LA.norm(abs(S)-abs(Sfull[:Dc])))

        print('degenerate eigh')
        for Dcut in range(1,Dfull+1):
            V,S,Dc = SplitEigh_Deg(STh,Dcut,icheck=1)
            print(Dcut,Dc,LA.norm(abs(S)-abs(Sfull[:Dc])))

        print('tensor eigh')
        for Dcut in range(1,12+1):
            print(Dcut)
            V,S,Dc = SplitEigh_Deg(STh,Dcut,icheck=1)
            V,S,Dc = SplitEigh_Z2Ten(ST0h,Dcut,2,icheck=1)

    if Test['SplitSvd'] == 1:
        ST = Group(ST0,[[0,1],[2,3]])
        ST1 = Group(ST0,[[0],[1],[2,3]])
        ST2 = Group(ST0,[[0,1],[2],[3]])
        Dfull = np.shape(ST)[0]
        print(Dfull)

        print('full svd')
        for iweight in range(2):
            U,S,V,Dc = SplitSvd(ST,Dfull,iweight,icheck=1)
        Sfull = S
        print(Sfull,len(Sfull))

        print('partial svd')
        for Dcut in range(1,Dfull+1):
            for iweight in range(2):
                U,S,V,Dc = SplitSvd(ST,Dcut,iweight,mode='C',icheck=1)
                print(Dcut,iweight,Dc,LA.norm(S-Sfull[:Dc]))

        print('degenerate svd')
        for Dcut in range(1,Dfull+1):
            for iweight in range(2):
                U,S,V,Dc = SplitSvd_Deg(ST,Dcut,iweight,mode='C',icheck=1)
                print(Dcut,iweight,Dc,LA.norm(S-Sfull[:Dc]))

        print('Z2 degenerate svd')
        TEO,Ind = GetEO(T,2)
        Ae = TEO[0,0]
        Ao = TEO[1,1]

        for Dcut in range(1,Dfull+1):
            for iweight in range(2):
                Ue,Uo,Se,So,Ve,Vo,Dce,Dco = SplitSvd_Z2Deg(Ae,Ao,Dcut,iweight,appro='S',mode='C')
                S = np.append(Se,So)
                S = np.sort(S)[::-1]
                Dc = len(S)
                print(Dcut,iweight,'S',Dce,Dco,LA.norm(S-Sfull[:Dc]))

                Ue,Uo,Se,So,Ve,Vo,Dce,Dco = SplitSvd_Z2Deg(Ae,Ao,Dcut,iweight,appro='T',mode='C')
                S = np.append(Se,So)
                S = np.sort(S)[::-1]
                Dc = len(S)
                print(Dcut,iweight,'T',Dce,Dco,LA.norm(S-Sfull[:Dc]))

        print('tensor svd')
        for Dcut in range(1,Dfull+1):
            for iweight in range(2):
                print(Dcut,iweight)
                U,S,V,Dc = SplitSvd_Deg(ST,Dcut,iweight,icheck=1)
                U,S,V,Dc = SplitSvd_Z2Ten(ST0,Dcut,iweight,2,icheck=1)
                U,S,V,Dc = SplitSvd_Z2Ten(ST1,Dcut,iweight,2,icheck=1)
                U,S,V,Dc = SplitSvd_Z2Ten(ST2,Dcut,iweight,1,icheck=1)

    if Test['Mps_QRP'] == 1:
        print('no Z2, rank 3')
        for D1,D2,D3 in itertools.product(range(3,5),range(3,5),range(3,5)):
            D = [D1,D2,D3]
            T0 = np.random.rand(D[0],D[1],D[2]) + 1j*np.random.rand(D[0],D[1],D[2])
            UL = np.random.rand(D[0],D[0]) + 1j*np.random.rand(D[0],D[0])
            UR = np.random.rand(D[2],D[2]) + 1j*np.random.rand(D[2],D[2])

            print(D1,D2,D3)
            T,UR = Mps_QRP(UL,T0,icheck=1)
            UL,T = Mps_LQP(T0,UR,icheck=1)
            T,UR = Mps_QR0P(T0,icheck=1)
            UL,T = Mps_LQ0P(T0,icheck=1)

        print('no Z2, rank 4')
        for D1,D2,D3,D4 in itertools.product(range(3,5),range(3,5),range(3,5),range(3,5)):
            D = [D1,D2,D3,D4]
            T0 = np.random.rand(D[0],D[1],D[2],D[3]) + 1j*np.random.rand(D[0],D[1],D[2],D[3])
            UL = np.random.rand(D[0],D[0]) + 1j*np.random.rand(D[0],D[0])
            UR = np.random.rand(D[3],D[3]) + 1j*np.random.rand(D[3],D[3])

            print(D1,D2,D3,D4)
            T,UR = Mps_QRP(UL,T0,icheck=1)
            UL,T = Mps_LQP(T0,UR,icheck=1)
            T,UR = Mps_QR0P(T0,icheck=1)
            UL,T = Mps_LQ0P(T0,icheck=1)

