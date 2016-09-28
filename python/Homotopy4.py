# Homotopy

#import scipy
#import cv2
import numpy as np
#from matplotlib import pyplot as plt
#from scipy.fftpack import fft, dct, idct
#from scipy.optimize import linprog, minimize
import time

from pseudoinversa import *
eps = 1e-10

def sgn(c):
    r = np.matrix(c.shape[0]*[0]).T
    for i in range(c.shape[0]):
        if c[i,0] > 0:
            r[i,0] = 1
        else:
            r[i,0] = -1
    return r

def homotopy(A, y):
	A = np.mat(A)
	y = np.mat(y)
	if y.shape[0] == 1:
		y = y.T
	m, n = A.shape
	
	x = np.matrix(np.array([0.]*n)).T

	TI = time.time()
	T = []

	c = A.T*y
	lambd = np.abs(c).max()
	i = np.argmax(c)
	I = [i]
	d = np.matrix([0.]*n).T

	d[I,:] = pinv(A[:,I].T*A[:,I])*sgn(c[I,:])
	print d.max()
	lambd_ant = 1e+11
	iA = pinv(A[:,I])
	ATA = A.T*A
	while lambd > eps and lambd_ant > lambd:
		t1 = time.time()
		tt1 = time.time()
		d[I,:] = iA*iA.T*sgn(c[I,:])
    
		tt2 = time.time()
		#print "Tempo calculando d:"+str(tt2 - tt1)
		L = []
		tt1 = time.time()
		#z = np.matrix([0]*n).T
		z = ATA[:,I]*d[I,:]
		tt2 = time.time()
		#print "tempo calculando z:"+str(tt2 - tt1)
		
		tt1 = time.time()
		v = np.array([1e+11]*n)
		
		I1 = np.where(d > eps)[0].tolist()
		I1 = np.intersect1d(I, I1).tolist()
		v[I1] = np.divide(-x[I1,0],d[I1,0]).T.tolist()[0]
		
		v1 = np.array([1e+11]*n)
		v2 = np.array([1e+11]*n)
		
		I2 = np.where(np.abs(z) < 1 - eps)[0].tolist()
		I2remove = np.intersect1d(I2, I)
		I2 = np.delete(I2, I2remove).tolist()
       
		v1[I2] = np.divide(lambd + c[I2,:], 1 + z[I2,:]).T.tolist()[0]
		v2[I2] = np.divide(lambd - c[I2,:], 1 - z[I2,:]).T.tolist()[0]
		v[I2]  = np.minimum(v1[I2], v2[I2])
		
		I_neg = [i for i in range(n) if v[i] < eps]
		v[I_neg] = 1e+11
    
		j = np.argmin(v) # <<< tem que ser positivo
		g = v[j]
		tt2 = time.time()
		tt2 = time.time()
		#print "Tempo no calculando g: "+str(tt2 - tt1)
		
		x_ant = x
		x = x + g*d

		tt1 = time.time()
		flag = False
		if j in I:
			I.remove(j)
			iA = pinv(A[:,I])
		else:
			I, iA = pinvCol(A, I, iA, j)
			flag = True
			#I.append(j)
		tt2 = time.time()
		#if flag:
		#	print "tempo calculando iA:"+str(tt2 - tt1)+", pinv rapido"
		#else:
		#	print "tempo calculando iA:"+str(tt2 - tt1)+", pinv demorado"
    
		tt1 = time.time()    
		c = A.T*(y - A*x)
		tt2 = time.time()
		#print "tempo calculando c:"+str(tt2 - tt1)
		lambd_ant = lambd
		lambd = np.abs(c).max()
		t2 = time.time()
		#print "lambda = "+str(lambd)
		#print "iteracao "+str(len(T))+", n = "+str(n)+", tempo = "+str(t2 - t1)
		#print "-----------------------------------------"
		T.append(t2 - t1)
	print "Tempo total minimizando l1: "+str(np.sum(T))
	return x

#A = np.random.randn(4,10)
#y = np.array([1,2,3,4])
#x = homotopy(A,y)
#print x
#print "------"
#print A*x
#print "------"
#print y