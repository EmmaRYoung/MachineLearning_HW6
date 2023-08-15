import scipy.io
mat = scipy.io.loadmat('mnist_all.mat')
import numpy as np
import random
import numpy.linalg as LA
import math 
import matplotlib.pyplot as plt


keep0 = 1000 #Training data has same number of samples for each class (easier)
train0 = mat["train0"]
train0 = train0[0:keep0,:]
train0 = train0/255  

train1 = mat["train1"]
train1 = train1[0:keep0,:]
train1 = train1/255  

train2 = mat["train2"]
train2 = train2[0:keep0,:]
train2 = train2/255  

train3 = mat["train3"]
train3 = train3[0:keep0,:]
train3 = train3/255  

train4 = mat["train4"]
train4 = train4[0:keep0,:]
train4 = train4/255  

train5 = mat["train5"]
train5 = train5[0:keep0,:]
train5 = train5/255  

train6 = mat["train6"]
train6 = train6[0:keep0,:]
train6 = train6/255  

train7 = mat["train7"]
train7 = train7[0:keep0,:]
train7 = train7/255  

train8 = mat["train8"]
train8 = train8[0:keep0,:]
train8 = train8/255  

train9 = mat["train9"]
train9 = train9[0:keep0,:]
train9 = train9/255  

#randomly sample 25% and then combine
percentage = 0.25
samples = int(keep0*percentage)
sequence = list(range(keep0))
ind0 = random.sample(sequence, samples)
ind1 = random.sample(sequence, samples)
ind2 = random.sample(sequence, samples)
ind3 = random.sample(sequence, samples)
ind4 = random.sample(sequence, samples)
ind5 = random.sample(sequence, samples)
ind6 = random.sample(sequence, samples)
ind7 = random.sample(sequence, samples)
ind8 = random.sample(sequence, samples)
ind9 = random.sample(sequence, samples)

#stack data up
trainALL = np.vstack((train0[ind0,:], train1[ind1,:], train2[ind2,:], train3[ind3,:], train4[ind4,:], train5[ind5,:], train6[ind6,:], train7[ind7,:], train8[ind8,:], train9[ind9,:]))
yall = np.vstack

def DistanceDecider(trainALL, cent, m):
    numCluster = len(cent)
    numSample = len(trainALL)
    
    distances = np.empty((numSample, numCluster))
    for i in range(numCluster):
        for j in range(numSample):
            distances[j,i] = LA.norm(cent[i]-trainALL[j,:], m)
            
            
    #find smallest distance
    indStorage = np.argmin(distances,axis=1)
  

    return indStorage


def EntropyPurity(clusterAssignment, k):
    numSamples = len(clusterAssignment)
    numClusters = k
    entropy = 0
    pureStore = []
    for i in range(numClusters):
        aa = len(np.where(clusterAssignment==i)[0])
        Pi = aa/numSamples
        entropy = entropy + Pi*math.log2(Pi)
        pureStore.append(Pi)
        
    entropy = entropy*-1
    purity = np.max(pureStore)
    return entropy, purity


    
    
 


#randomly choose k data points to be the initial centroids, cluster centers
k = [7, 10, 12, 7, 10, 12]
m = [2, 2, 2, 1, 1, 1]
numFeatures = np.shape(trainALL)[1]
stopCalc = 0.01  
storage = []

#this is for euchludean distance 
entropyStore = []
purityStore = []
varStore = np.empty((np.sum(k), numFeatures))
meanStore = np.empty((np.sum(k), numFeatures))
for i in range(len(k)):
    stoppingCriteria = False
    count = 0
    while(stoppingCriteria == False):
        if count==0:
            ind = random.sample(sequence, k[i]) #initial random sample is the centroid
            cent = trainALL[ind,:]
        print(count)
        count = count + 1
            
        #assign each data point to the closest centroid
        clusterAssignment = DistanceDecider(trainALL,cent,m[i])
        
        #re-compute the centroids using the current cluster assignments
        oldCent = cent
        
        cent = np.zeros((k[i], numFeatures))
        for j in range(k[i]):
            currInd = np.where(clusterAssignment == j)
            currData = trainALL[currInd,:][0]
            mean = np.mean(currData, axis=0)
            cent[j,:] = np.transpose(mean)
        
        #look at change in location of centroids            
        centroidDiff = LA.norm(cent - oldCent, axis=1)
        
        if centroidDiff.all() < stopCalc:
            stoppingCriteria = True
            print("stopping")
            #calculate purity
            
            #calulate entropy
            [E, P] = EntropyPurity(clusterAssignment, k[i])
            entropyStore.append(E)
            purityStore.append(P)
            
            #calculate mean and variance of each cluster, and plot each cluster
            for ii in range(k[i]):
                currInd = np.where(clusterAssignment==ii)
                currData = trainALL[currInd,:][0]
                var = np.var(currData,axis=0)
                mean = np.mean(currData,axis=0)
                
                varStore[i,:] = var
                meanStore[i,:] = mean
                '''
                plt.figure()
                reshapedImage = np.reshape(currData,(len(currData), 28, 28,1))
                meanImage = np.mean(reshapedImage, axis=0)
                plt.imshow(meanImage)
                plt.title("Cluster number " + str(ii) + " for k =" + str(k[i]) + "  for norm:" + str(m[i]))
                plt.show()
                '''
                
                plt.figure()
                reshapedImage = np.reshape(mean, (28, 28, 1))
                plt.imshow(reshapedImage)
                plt.title("Cluster number " + str(ii) + " for k =" + str(k[i]) + "  for norm:" + str(m[i]) + " Mean")
                plt.show()
                
                plt.figure()
                reshapedImage = np.reshape(var, (28, 28, 1))
                plt.imshow(reshapedImage)
                plt.title("Cluster number " + str(ii) + " for k =" + str(k[i]) + "  for norm:" + str(m[i]) + " Var")
                plt.show()
            '''
            image=np.concatenate((x_train_gray[1],x_train_gray[1],x_train_gray[1]),axis = 2)
            plt.imshow(image)
            plt.colorbar()
            plt.grid(False)
            plt.show()
            '''

    




