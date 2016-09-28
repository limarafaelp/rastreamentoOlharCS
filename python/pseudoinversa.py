import scipy
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, dct, idct
from scipy.optimize import linprog, minimize
import time
img = cv2.imread('eye_thumbnail.jpg',0)
#img = cv2.imread('eye_thumbnail_mini.jpg',0)

#return np.bmat([[Ak_i - g.T*c],[c]])

def pinv(A, iA = None):
    # Calcula a pseudoinversa da matriz A = [a1 | ... | ak]
    # onde iA eh a pseudoinversa de [a1| ... |ak-1]

    # so funciona para A matrix de tipo float
    # Entao ainda nao funciona para array
    eps = 1e-10
    m, n = A.shape
    if n == 1:
        x = np.linalg.norm(A)**2
        if x == 0:
            return A.T
        else:
            return (1./x)*A.T

    #A1 = A[:,:n-1]
    #a  = A[:,n-1] #ultima coluna

    if iA == None:
		k0 = 1
		iA = pinv(A[:,0])						
    else:
        k0 = n-1
        
    for k in range(k0,n):
        A1 = A[:,:k]
        a  = A[:,k]
        g = iA*a
        gg = (g.T*g)[0,0]

        if np.abs(A1*g - a).max() < eps:
            c = (1./(1 + gg))*g.T*iA        
        else:
            c = pinv(a - A1*g)
        iA = np.bmat([[iA - g*c],[c]])
    return iA
        

def pinvCol(A, I, iA, i):
    # Calcula a pseudo-inversa de A[:,J]
    # onde J = [I, i]
    # iA eh a pseudo-inversa de A[:,I]
    # I: lista ordenada de indices
    if i in I:
        return I, iA
    
    if len(I) == 0:
        iA = pinv(A[:,i])
        I = [i]
        return I, iA
    
    B = np.bmat([A[:,I], A[:,i]])
    iB = pinv(B, iA)
    I.append(i)
    J = np.argsort(I)
    iB = iB[J,:]
    I.sort()
    return I, iB

def pinv_Iuv(u, v):
    # Calcula a pseudo-inversa de I + uv.T
    # onde I eh a matriz identidade e
    # u, v sao vetores de mesma dimensao
    # u, v devem ser matrix

    B = u*v.T
    B[np.diag_indices_from(B)] += 1.

    print B

    s = np.linalg.norm(B, axis = 0)**2
    I = np.nonzero(s)[0]
    s[I] = 1./s[I]

    print s
    return np.multiply(B.T,s).T
    #return (s*B).T
    

    
#u = np.mat([2.,1.,3.]).T
#v = np.mat([-1.,0.,1.]).T
#B = np.mat(np.eye(3)) + u*v.T

#A = np.mat(np.random.randn(10,100))
