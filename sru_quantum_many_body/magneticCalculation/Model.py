import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as LAs
import Sub180221 as Sub
import math,sys,subprocess,glob,copy
import pylab as pl
#----------------------------------------------------------------
def GetHam_Heisenberg(Dp):
	S0,Sp,Sm,Sz,Sx,Sy = Sub.SpinOper(Dp)
	Sx *= 2.0
	Sy *= 2.0
	Sz *= 2.0

	Ham = Sub.NCon([Sx,Sx],[[-1,-3],[-2,-4]]) + Sub.NCon([Sy,Sy],[[-1,-3],[-2,-4]]) + Sub.NCon([Sz,Sz],[[-1,-3],[-2,-4]])

	return Ham
