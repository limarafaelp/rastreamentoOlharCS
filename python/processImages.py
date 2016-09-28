import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from random import randint

from Homotopy4 import homotopy
def getTargetSamples(i, j, nsamples, npyrDown, isRand = False):
    file = 'video'+str(i)+'_'+str(j)+'.avi'
    cap = cv2.VideoCapture("../data/"+file)

    if not cap.isOpened():
        print "video ("+file+")file could not be opened!"
        quit()
        #return -1

    nframes = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    #fwidth  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    #fheight = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    
    #width   = fwidth /(2**npyrDown)
    #height  = fheight/(2**npyrDown)

    #w = width/2
    if isRand:
        samples = [randint(0, nframes - 1) for i in range(nsamples)]
    else:
        step = nframes/nsamples
        samples = [i * step for i in range(nsamples)]
        
    #Right = np.mat(np.zeros((height*w            ,1)))
    #Left  = np.mat(np.zeros((height*(width - w),  1)))
    Right = []
    Left  = []
    for ind in range(nsamples):
        #ind = i * step
        cap.set(1, ind)
        flag, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for k in range(npyrDown):
            frame = cv2.pyrDown(frame)
        
        w = frame.shape[1]/2
        xr = np.mat(frame[:,:w].ravel()).T
        xl = np.mat(frame[:,w:].ravel()).T
        
        if len(Right) == 0:
            Right = np.bmat([xr])
            Left  = np.bmat([xl])
        else:
            Right = np.bmat([Right, xr])
            Left  = np.bmat([Left , xl])
    
    cap.release()
    #Right = Right[:,1:]
    #Left  = Left[:,1:]
    return (Right, Left)

def processVideos(m, n, nsamples, npyrDown):
    #R, L = processVideoTarget(0,0)
    R = []
    L = []
    for i in range(m):
        for j in range(n):
            r, l = getTargetSamples(i, j, nsamples, npyrDown)
            if len(R) == 0:
                R = r
                L = l
            else:
                R = np.bmat([R, r])
                L = np.bmat([L, l])
    print R.shape
    Id = np.mat(np.eye(R.shape[0]))
    R = np.bmat([R, Id])
    Id = np.mat(np.eye(L.shape[0]))
    L = np.bmat([L, Id])
    return (R, L)

def getRandomSample(m, n, npyrDown):
    i = randint(0,m - 1)
    j = randint(0,n - 1)
    R, L = getTargetSamples(i, j, 1, npyrDown, isRand = True)
    return (i,j, R, L)

def estimateTarget(m, n, A, a):
    c = homotopy(R, r)
    likelihoodEye = [np.max(np.abs(c[i*nsamples:(i+1)*nsamples,0])) for i in range(m*n)]
    #likelihoodNon = [np.max(c[nsamples*m*n:,0])]
    ind = np.argmax(likelihoodEye)
    i = ind/m
    j = ind % m
    return i,j
    
nsamples = 5
npyrDown = 3
R, L = processVideos(4,4, nsamples, npyrDown)
#print R.shape
#print L.shape

i, j, r, l = getRandomSample(4,4,npyrDown)
print "amostra aleatoria de "+str((i,j))

#c = homotopy(R, r)
#print "norma infinito de c = "+str(np.max(np.abs(c)))
m = n = 4

print "Posicao estimada olho direito:  "+str(estimateTarget(m,n,R,r))
print "Posicao estimada olho esquerdo: "+str(estimateTarget(m,n,L,l))
