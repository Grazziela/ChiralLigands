import numpy as np
import numpy.linalg.linalg as LA

def train(x, y, indices, alpha=0, beta=1, wt_threshold=1e-3):
    # add a column of 1's to the X variable matrix
    # Zero intercept method wrong and the proper method is in dispute.
    # Marquardt and Snee 1974; Maddala 1977; Gordon 1981).
    indices2 = ['Intercept'] + list(indices)
    H = np.ones((x.shape[0], x.shape[1]+1), float)
    H[:,1:] = x
    r, c = H.shape
    HT = np.transpose(H)
    HTy = np.dot(HT,y)
    Ainv = LA.pinv(np.dot(HT ,H))
    w = np.dot(Ainv,HTy)
    nw = 0
    for i in range(c):
        if np.absolute(w[i])>wt_threshold: nw+=1
    #yPred = np.dot(H,w)
    #yDiff = yPred -y
    ssd2 = 1#np.sqrt(np.dot(np.transpose(yDiff),yDiff)/r)
    nw2 = c
    #EM algorithm
    itera = 0
    change = 1.
    while itera<300 and change>0.01 and ssd2>1.e-15 and nw2>5:
        nw1 = nw2
        indices1 = indices2
        U = np.eye(nw1)
        Ic = np.eye(nw1)
        itera += 1
        ssd1 = ssd2
        U =  U*np.abs(w)
        Ic = Ic * (alpha + beta*ssd1*ssd1)
        R1 = np.dot(np.dot(HT ,H),U)
        R2 = np.dot(U,R1)
        R3 = LA.pinv(Ic + R2)
        R4 = np.dot(U,R3)
        R5 = np.dot(U,HTy)
        w = np.dot(R4,R5)
        yPred = np.dot(H,w)
        yDiff = yPred - y
        ssd2 =  np.sqrt(np.dot(np.transpose(yDiff),yDiff)/r)
        change = 100*np.absolute(ssd2-ssd1)/ssd1
        # reduce matrix size
        nw2 = 0
        for i in range(nw1):
            if np.absolute(w[i])>wt_threshold: nw2+=1
        ww = np.zeros(nw2,float).reshape(-1,1)
        HH = np.zeros((r,nw2),float)
        k = -1
        indices2 = []
        for i in range(nw1):
            if np.absolute(w[i]) > wt_threshold:
                k+=1
                ww[k] = w[i]
                indices2.append(indices1[i])
                for j in range(r):
                    HH[j,k] = H[j,i]                  
        H = HH
        w = ww
        HT = np.transpose(H)
        HTy = np.dot(HT,y)

    weights = w   
    features = H
    ypred = np.dot(features,weights)
    nw = weights.shape[0]

    C = LA.pinv(np.dot(np.transpose(features),features)).diagonal()

    if (y.shape[0] - nw) <= 0:
        division = 1
    else:
        division = (y.shape[0] - nw)
    se = np.sqrt((np.sum(((y - ypred)*(y - ypred)))/(division)))
    
    CI = np.sqrt(C*se*se).reshape(-1,1)
    s = np.random.standard_t(nw, size=10000)
    tvalue = np.abs(np.divide(weights,CI)).reshape(-1,1)
    pvalue = np.sum(s>tvalue,axis=1) / float(len(s))

    
    return w, indices2, pvalue