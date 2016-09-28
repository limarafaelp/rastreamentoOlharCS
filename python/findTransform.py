import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt

from Homotopy4 import homotopy

#f0 = cv2.imread('imagens/F17.jpg',0)
f0 = cv2.imread('imagens/eye_thumbnail.jpg',0)
f1 = cv2.imread('imagens/eye_thumbnail.jpg',0)
#f1 = cv2.imread('imagens/F16.jpg',0)

#print f1.shape

M, N = 50,50 #numero de elementos da grade -1
step = np.divide(f1.shape, (M,N))
#print step
y = [i*step[0] for i in range(M+1)]
x = [i*step[1] for i in range(N+1)]

if y[M] == f1.shape[0]:
	y[M] -= 1
if x[N] == f1.shape[1]:
	x[N] -= 1

n = 15
A = []

#print y
#print x
#print f1.shape
coordList = []
for j in range(len(x)):
	for i in range(len(y)):
		coordList.append((y[i],x[j]))
		#print y[i], x[j]
		if  j <= (N+1)/2:
			u = f1[y[i], x[j]:x[j]+n]
			f1[y[i], x[j]:x[j]+n] = 255
		else:
			u = f1[y[i], x[j] -n + 1:x[j]+1]
			f1[y[i], x[j] -n + 1:x[j]+1] = 255
		if i <= (M+1)/2:
			v = f1[y[i]:y[i]+n, x[j]]
			f1[y[i]:y[i]+n, x[j]] = 255
		else:
			v = f1[y[i] -n +1:y[i]+1, x[j]]
			f1[y[i] -n +1:y[i]+1, x[j]] = 255
		u = np.matrix(u).T
		v = np.matrix(v).T
		uv = np.bmat([u, v])
		if len(A) == 0:
			A = uv
		else:
			A = np.bmat([A, uv])

#print uv.shape
#print A.shape

Id = np.mat(np.eye(n))
A = np.bmat([A, Id])
sample = np.mat(f0[15:30, 15:30])
#print "shape a1"+str(a1_x.shape)
#a1_x = f0[20,20:n+20]
print A.shape
#print a1_x.shape

#a1_x = f0[15:25, 15:25]
c = np.mat(np.zeros((A.shape[1],1)))
for col in range(sample.shape[1]):
	s = sample[:, col]
	print str(s.shape) + "eh a shape de s"
	print str(A.shape) + "eh a shape de A"
	cs = homotopy(A, s)
	c = c + np.abs(cs)

print "o valor maximo em modulo de c eh "+str(np.max(np.abs(c)))
j = np.argmax(np.abs(c[:-n])) /2 #pois empilhamos dois vetores por coordenada. Excluindo elementos 'cross'
print "indice j = "+str(j)+", coordList tem "+str(len(coordList))+"elementos..."

if j < len(coordList):
	xj, yj = coordList[j]
	x_c = [xj + k for k in range(-10,9) if 0 <= k <= f1.shape[1]]
	y_c = [yj + k for k in range(-10,9) if 0 <= k <= f1.shape[0]]
	#coord_c = [(ky, kx) for kx in x_c for ky in y_c]
	#print coord_c
	#f1[ymax - 40:ymax+40, xmax - 40:xmax+40] = 2
	#print y_c
	#print x_c
	for xcc in x_c:
		for ycc in y_c:
			f1[ycc, xcc] = 200
plt.imshow(f1, cmap='gray')
plt.show()

print "10 elementos de c:"
c = np.abs(c)
c.sort()
print c[:10]

f0[15:25, 15:25] = 200
plt.imshow(f0, cmap='gray')
plt.show()