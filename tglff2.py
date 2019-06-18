# -*- coding: utf-8 -*-
"""
Created on Tuesday June 06 12:05:47 2019
@author: Dr Clement Etienam
Supervisor: Professor Kody Law
Parralel CCR
Logic:
1) Determine optimum number of clusters of X and y pair
2) Do K-means custering of the X,y pair obtain cluster label d
3) Create Folder equal to number of clusters determined from elbow in step 1
4) Build classifier of X and d, save the classifier model
5) Build regressors in parallel for each cluster of X and y and save model using Joblib
6) Predict for test data in series

Important steps for CCR to be succesful
1) Make each column of the training input matrix (X) to follow a 'Gaussian distribution'
2) During clustering when concatenating the X,y matrix together, make a copy of y (new_y) and increase magnitude of ' new_y' 
3) During regression resclae 'y' to be between 0 and 1, to make the regression easier
4) Dont forget to backtransform the prediction to original scale
"""
from __future__ import print_function
print(__doc__)

from sklearn.neural_network import MLPClassifier
import numpy as np
import scipy.io as sio
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from numpy import linalg as LA
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.layers import Dense, Activation
from keras.models import Sequential 
import os
import shutil
import multiprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

## This section is to prevent Windows from sleeping when executing the Python script
class WindowsInhibitor:
    '''Prevent OS sleep/hibernate in windows; code from:
    https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/master/preventsuspendplus/core.py
    API documentation:
    https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx'''
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self):
        pass

    def inhibit(self):
        import ctypes
        #Preventing Windows from going to sleep
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS | \
            WindowsInhibitor.ES_SYSTEM_REQUIRED)

    def uninhibit(self):
        import ctypes
        #Allowing Windows to go to sleep
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS)


osSleep = None
# in Windows, prevent the OS from sleeping while we run
if os.name == 'nt':
    osSleep = WindowsInhibitor()
    osSleep.inhibit()
##------------------------------------------------------------------------------------

## Start of Programme

print( 'Parallel CCR for TGLF ')


oldfolder = os.getcwd()
cores = multiprocessing.cpu_count()
print(' ')
print(' This computer has %d cores, which will all be utilised in parallel '%cores)
#print(' The number of cores to be utilised can be changed in runeclipse.py and writefiles.py ')
print(' ')

start = datetime.datetime.now()
print(str(start))

print('Determine the optimum number of clusers using Gap statistics') # You may ignore this step, i noticed 10-15 clusters was enough

def optimalK(data, nrefs, maxClusters):
    """
    Calculates KMeans optimal K using Gap Statistic 
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k,n_jobs=-1)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k,n_jobs=-1)
        km.fit(data)
        
        origDisp = km.inertia_
        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal


filename2= 'Claffiermodel1.asv'
modelclass=MLPClassifier(solver= 'lbfgs',max_iter=5000)
def run_model(model,xxx,yyy,inputtest,filename2):
    # build the model on training data
    model.fit(xxx, yyy )
    print('Save classifier')
    pickle.dump(model, open(filename2, 'wb'))

    # make predictions for test data
    labelDA = model.predict(inputtest)

    return labelDA
#------------------Begin Code-------------------------------------------------------------------#
print('Build the machine for output 1')
sgsim = open("orso.out") #533051 by 28
sgsim = np.fromiter(sgsim,float)

data = np.reshape(sgsim,(385747,28), 'F')
print('Standardize and normalize the input data')
input1=data[:,0:22]
from scipy.stats import rankdata, norm
newbig=np.zeros((385747,22))
for i in range(22):
    input11=input1[:,i]
    newX = norm.ppf(rankdata(input11)/(len(input11) + 1))
    newbig[:,i]=newX.T

input1=newbig
scaler = MinMaxScaler()
(scaler.fit(input1))
input1=(scaler.transform(input1))

output=data[:,22:29]
input11=input1
numrows=len(input1)    # rows of inout
numcols = len(input1[0])
inputtrain=(input1[0:300000,:]) #select the first 300000 for training
inputtest=(input1[300001:numrows,:]) #use the remaining data for testing
numruth = len(input1[0])
print('For the first output')
outputtrain=np.reshape((output[0:300000,0]),(-1,1)) #select the first 300000 for training
ydami=outputtrain

ydamir=outputtrain
scaler1 = MinMaxScaler()
(scaler1.fit(ydamir))
ydamir=(scaler1.transform(ydamir))

outputtest1=np.reshape((output[300001:numrows,0]),(-1,1))

numrowstest=len(outputtest1)    # rows of inout
numcolstest = len(outputtest1[0])

matrix=np.concatenate((inputtrain,ydami), axis=1)
#ruuth = np.int(input("Enter the maximum number of clusters you want to accomodate for the CCR machine: ") )
ruuth=30 # 10-15 clusters was enough
nclusters=ruuth
print('Do the K-means clustering of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=ruuth,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
dd=dd.T
dd=np.reshape(dd,(-1,1))


def deep_learningclass(inputtrainclass, outputtrainclasss,filename):
    np.random.seed(7)
    modelDNN = Sequential()

# Adding the input layer and the first hidden layer
    modelDNN.add(Dense(200, activation = 'relu', input_dim = numruth))

# Adding the second hidden layer
    modelDNN.add(Dense(units = 420, activation = 'relu'))
    #modelDNN.add(Dense(units = 620, activation = 'relu'))
# Adding the third hidden layer
    modelDNN.add(Dense(units = 21, activation = 'relu'))

# Adding the output layer

    #modelDNN.add(Dense(units = 1))
    modelDNN.add(Dense(nclusters, activation='softmax'))

    modelDNN.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting the ANN to the Training set
    modelDNN.fit(inputtrainclass, outputtrainclasss,validation_split=0.01, batch_size = 50, epochs = 200)
    pickle.dump(modelDNN, open(filename, 'wb'))
    # make predictions for test data
    labelDA = modelDNN.predict(inputtest)
    return labelDA

#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))

print(' Learn the classifer from the predicted labels from Kmeans')

outputtrainclasss = to_categorical(outputtrainclass)

labelDA=deep_learningclass(inputtrainclass, outputtrainclasss,filename2)
#labelDA=run_model(modelclass,inputtrainclass, outputtrainclass,inputtest,filename2)
labelDA=np.argmax(labelDA, axis=-1) 


X_train=inputtrain
X_test=inputtest
y_train=dd
y_test=dd


np.savetxt('y_train1.out',y_train, fmt = '%4.6f', newline = '\n')
np.savetxt('X_train1.out',X_train, fmt = '%4.6f', newline = '\n')
np.savetxt('y_traind1.out',ydamir, fmt = '%4.6f', newline = '\n')
np.savetxt('labelDA1.out',labelDA, fmt = '%d', newline = '\n')
np.savetxt('inputtest1.out',inputtest, fmt = '%4.6f', newline = '\n')

a = open('numrowstest1.out', 'w')
#a.write('%4.6f',numrowstest)
a.write("%i \n" % (numrowstest))
a.close()

#-------------------Regression---------------------------------------------------#
print('Learn regression of the clusters with different labels from k-means ' )

#clementanswer=np.zeros((numrowstest,1))

print('Start the regression')

oldfolder = os.getcwd()
os.chdir(oldfolder)

print('')

for j in range(nclusters):
    folder = 'CCRmodel1_%d'%(j)
    if os.path.isdir(folder): # value of os.path.isdir(directory) = True
        shutil.rmtree(folder)      
    os.mkdir(folder)
    shutil.copy2('y_train1.out',folder)
    shutil.copy2('X_train1.out',folder)
    shutil.copy2('y_traind1.out',folder)


    shutil.copy2('numrowstest1.out',folder)

print( ' Solving the prediction problem' )
print('')
os.chdir(oldfolder)
##
import multiprocessing
from sklearn.neural_network import MLPRegressor
import numpy as np
#from sklearn.neural_network import MLPRegressor

import os
from joblib import Parallel, delayed

filename1='regressor1.asv'
#
def parad(j):
    
    oldfolder = os.getcwd()
    folder = 'CCRmodel1_%d'%(j)
    os.chdir(folder)
    y_train=np.genfromtxt("y_train1.out", dtype='float')

    y_train = np.reshape(y_train,(-1,1), 'F') 
    
    
    X_train = np.genfromtxt("X_train1.out", dtype='float') #533051 by 28
   
    X_train = np.reshape(X_train,(-1,numruth), 'F') 
    
    y_traind = np.genfromtxt("y_traind1.out", dtype='float') #533051 by 28
   
    y_traind = np.reshape(y_traind,(-1,1), 'F') 

    label0=(np.asarray(np.where(y_train == j))).T    
#    model0 = RandomForestRegressor(n_estimators=2000)
    model0 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
    a0=X_train[label0,:]
    a0=np.reshape(a0,(-1,numruth),'F')

    b0=y_traind[label0,:]
    b0=np.reshape(b0,(-1,1),'F')
    
    model0.fit(a0, b0)
    pickle.dump(model0, open(filename1, 'wb'))
    os.chdir(oldfolder)
    print(" cluster %d has been processed"%(j))
    
print('')
print('cluster in parallel')
print('')
clementanswer1=np.zeros((numrowstest,1))    
num_cores = multiprocessing.cpu_count()
number_of_realisations = range(nclusters)
Parallel(n_jobs=nclusters, verbose=50)(delayed(
    parad)(j)for j in number_of_realisations)


print('')
print('predict in series')
print('')
for i in range(nclusters):
    folder = 'CCRmodel1_%d'%(i)
    os.chdir(folder)    
    loaded_model = pickle.load(open(filename1, 'rb'))

    
    labelDA0=(np.asarray(np.where(labelDA == i))).T


##----------------------##------------------------##
    a00=inputtest[labelDA0,:]
    a00=np.reshape(a00,(-1,numruth),'F')
    if a00.shape[0]!=0:
        clementanswer1[np.ravel(labelDA0),:]=np.reshape(loaded_model .predict(a00),(-1,1))
    os.chdir(oldfolder)

clementanswer1=scaler1.inverse_transform(clementanswer1)
print('End of training the machine for output 1')

##----------------------------Output 2--------------------------------------------------------##

print('Build the machine for output 2')
print('')
outputtrain=np.reshape((output[0:300000,1]),(-1,1)) #select the first 300000 for training
ydami=outputtrain

ydamir=outputtrain
scaler2 = MinMaxScaler()
(scaler2.fit(ydamir))
ydamir=(scaler2.transform(ydamir))
#outputtrain=np.arcsinh(outputtrain)
outputtest2=np.reshape((output[300001:numrows,1]),(-1,1))

numrowstest=len(outputtest2)    # rows of inout
numcolstest = len(outputtest2[0])

matrix=np.concatenate((inputtrain,ydami), axis=1)
nclusters=ruuth
print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=ruuth,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
dd=dd.T
dd=np.reshape(dd,(-1,1))

filename2= 'Claffiermodel2.asv'

#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))

print(' Learn the classifer from the predicted labels from Kmeans')

outputtrainclasss = to_categorical(outputtrainclass)

labelDA=deep_learningclass(inputtrainclass, outputtrainclasss,filename2)
labelDA=np.argmax(labelDA, axis=-1) 
#labelDA=run_model(modelclass,inputtrainclass, outputtrainclass,inputtest,filename2)

X_train=inputtrain
X_test=inputtest
y_train=dd
y_test=dd

np.savetxt('y_train2.out',y_train, fmt = '%4.6f', newline = '\n')
np.savetxt('X_train2.out',X_train, fmt = '%4.6f', newline = '\n')
np.savetxt('y_traind2.out',ydamir, fmt = '%4.6f', newline = '\n')
np.savetxt('labelDA2.out',labelDA, fmt = '%d', newline = '\n')
np.savetxt('inputtest2.out',inputtest, fmt = '%4.6f', newline = '\n')

a = open('numrowstest2.out', 'w')
#a.write('%4.6f',numrowstest)
a.write("%i \n" % (numrowstest))
a.close()

#-------------------Regression-----------------------------------------------------------#
print('Learn regression of the clusters with different labels from k-means ' )

#clementanswer=np.zeros((numrowstest,1))

print('Start the regression')

oldfolder = os.getcwd()
os.chdir(oldfolder)

print('')
##Creating Folders and Copying Simulation Datafiles
#print(' Creating the folders and copying the simulation files for the forward problem ')
print('')
for j in range(nclusters):
    folder = 'CCRmodel2_%d'%(j)
    if os.path.isdir(folder): # value of os.path.isdir(directory) = True
        shutil.rmtree(folder)      
    os.mkdir(folder)
    shutil.copy2('y_train2.out',folder)
    shutil.copy2('X_train2.out',folder)
    shutil.copy2('y_traind2.out',folder)


    shutil.copy2('numrowstest2.out',folder)

print( ' Solving the prediction problem' )
print('')
os.chdir(oldfolder)
##
import multiprocessing

import numpy as np
#from sklearn.neural_network import MLPRegressor

import os
from joblib import Parallel, delayed

filename2='regressor2.asv'
#
def parad2(j):
    
    oldfolder = os.getcwd()
    folder = 'CCRmodel2_%d'%(j)
    os.chdir(folder)
    y_train=np.genfromtxt("y_train2.out", dtype='float')

    y_train = np.reshape(y_train,(-1,1), 'F') 
    
    
    X_train = np.genfromtxt("X_train2.out", dtype='float') #533051 by 28
   
    X_train = np.reshape(X_train,(-1,numruth), 'F') 
    
    y_traind = np.genfromtxt("y_traind2.out", dtype='float') #533051 by 28
   
    y_traind = np.reshape(y_traind,(-1,1), 'F') 

    label0=(np.asarray(np.where(y_train == j))).T    
#    model0 = RandomForestRegressor(n_estimators=2000)
    model0 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
    a0=X_train[label0,:]
    a0=np.reshape(a0,(-1,numruth),'F')

    b0=y_traind[label0,:]
    b0=np.reshape(b0,(-1,1),'F')
    
    model0.fit(a0, b0)
    pickle.dump(model0, open(filename2, 'wb'))
    os.chdir(oldfolder)
    print(" cluster %d has been processed"%(j))
    
print('')
print('cluster in parallel')
print('')
clementanswer2=np.zeros((numrowstest,1))    
num_cores = multiprocessing.cpu_count()
number_of_realisations = range(nclusters)
Parallel(n_jobs=nclusters, verbose=50)(delayed(
    parad2)(j)for j in number_of_realisations)


print('')
print('predict in series')
print('')
for i in range(nclusters):
    folder = 'CCRmodel2_%d'%(i)
    os.chdir(folder)    
    loaded_model = pickle.load(open(filename2, 'rb'))

    
    labelDA0=(np.asarray(np.where(labelDA == i))).T


##----------------------##------------------------##
    a00=inputtest[labelDA0,:]
    a00=np.reshape(a00,(-1,numruth),'F')
    if a00.shape[0]!=0:
        clementanswer2[np.ravel(labelDA0),:]=np.reshape(loaded_model .predict(a00),(-1,1))
    os.chdir(oldfolder)


clementanswer2=scaler2.inverse_transform(clementanswer2)
print('End of Building the machine for output 2')
##----------------------------Output 3--------------------------------------------------------

print('Build the machine for output 3')
outputtrain=np.reshape((output[0:300000,2]),(-1,1)) #select the first 300000 for training
ydami=outputtrain

ydamir=outputtrain
scaler3 = MinMaxScaler()
(scaler3.fit(ydamir))
ydamir=(scaler3.transform(ydamir))
#outputtrain=np.arcsinh(outputtrain)
outputtest3=np.reshape((output[300001:numrows,2]),(-1,1))

numrowstest=len(outputtest3)    # rows of inout
numcolstest = len(outputtest3[0])

matrix=np.concatenate((inputtrain,ydami), axis=1)
#ruuth = np.int(input("Enter the maximum number of clusters you want to accomodate for 3rd machine: ") )

nclusters=ruuth
print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=ruuth,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
dd=dd.T
dd=np.reshape(dd,(-1,1))

filename2= 'Claffiermodel3.asv'
#ddtest=np.reshape(ddtest,(-1,1))
#-------------------#----------------------------------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))

print(' Learn the classifer from the predicted labels from Kmeans')

outputtrainclasss = to_categorical(outputtrainclass)

labelDA=deep_learningclass(inputtrainclass, outputtrainclasss,filename2)
labelDA=np.argmax(labelDA, axis=-1) 
#labelDA=run_model(modelclass,inputtrainclass, outputtrainclass,inputtest,filename2)

X_train=inputtrain
X_test=inputtest
y_train=dd
y_test=dd


np.savetxt('y_train3.out',y_train, fmt = '%4.6f', newline = '\n')
np.savetxt('X_train3.out',X_train, fmt = '%4.6f', newline = '\n')
np.savetxt('y_traind3.out',ydamir, fmt = '%4.6f', newline = '\n')
np.savetxt('labelDA3.out',labelDA, fmt = '%d', newline = '\n')
np.savetxt('inputtest3.out',inputtest, fmt = '%4.6f', newline = '\n')

a = open('numrowstest3.out', 'w')
#a.write('%4.6f',numrowstest)
a.write("%i \n" % (numrowstest))
a.close()

#-------------------Regression----------------#
print('Learn regression of the clusters with different labels from k-means ' )

#clementanswer=np.zeros((numrowstest,1))

print('Start the regression')

oldfolder = os.getcwd()
os.chdir(oldfolder)

print('')
##Creating Folders and Copying Simulation Datafiles
#print(' Creating the folders and copying the simulation files for the forward problem ')
print('')
for j in range(nclusters):
    folder = 'CCRmodel3_%d'%(j)
    if os.path.isdir(folder): # value of os.path.isdir(directory) = True
        shutil.rmtree(folder)      
    os.mkdir(folder)
    shutil.copy2('y_train3.out',folder)
    shutil.copy2('X_train3.out',folder)
    shutil.copy2('y_traind3.out',folder)


    shutil.copy2('numrowstest3.out',folder)

print( ' Solving the prediction problem' )
print('')
os.chdir(oldfolder)
##
import multiprocessing

import numpy as np
#from sklearn.neural_network import MLPRegressor

import os
from joblib import Parallel, delayed

filename3='regressor3.asv'
#
def parad3(j):
    
    oldfolder = os.getcwd()
    folder = 'CCRmodel3_%d'%(j)
    os.chdir(folder)
    y_train=np.genfromtxt("y_train3.out", dtype='float')

    y_train = np.reshape(y_train,(-1,1), 'F') 
    
    
    X_train = np.genfromtxt("X_train3.out", dtype='float') #533051 by 28
   
    X_train = np.reshape(X_train,(-1,numruth), 'F') 
    
    y_traind = np.genfromtxt("y_traind3.out", dtype='float') #533051 by 28
   
    y_traind = np.reshape(y_traind,(-1,1), 'F') 

    label0=(np.asarray(np.where(y_train == j))).T    
#    model0 = RandomForestRegressor(n_estimators=2000)
    model0 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
    a0=X_train[label0,:]
    a0=np.reshape(a0,(-1,numruth),'F')

    b0=y_traind[label0,:]
    b0=np.reshape(b0,(-1,1),'F')
    
    model0.fit(a0, b0)
    pickle.dump(model0, open(filename3, 'wb'))
    os.chdir(oldfolder)
    print(" cluster %d has been processed"%(j))
    
print('')
print('cluster in parallel')
print('')
clementanswer3=np.zeros((numrowstest,1))    
num_cores = multiprocessing.cpu_count()
number_of_realisations = range(nclusters)
Parallel(n_jobs=nclusters, verbose=50)(delayed(
    parad3)(j)for j in number_of_realisations)


print('')
print('predict in series')
print('')
for i in range(nclusters):
    folder = 'CCRmodel3_%d'%(i)
    os.chdir(folder)    
    loaded_model = pickle.load(open(filename3, 'rb'))

    
    labelDA0=(np.asarray(np.where(labelDA == i))).T


##----------------------##------------------------##
    a00=inputtest[labelDA0,:]
    a00=np.reshape(a00,(-1,numruth),'F')
    if a00.shape[0]!=0:
        clementanswer3[np.ravel(labelDA0),:]=np.reshape(loaded_model .predict(a00),(-1,1))
    os.chdir(oldfolder)


clementanswer3=scaler3.inverse_transform(clementanswer3)
print('End of building machine for output 3')


##----------------------------Output 4--------------------------------------------------------

print('Build the machine for output 4')
outputtrain=np.reshape((output[0:300000,3]),(-1,1)) #select the first 300000 for training
ydami=outputtrain

ydamir=outputtrain
scaler4 = MinMaxScaler()
(scaler4.fit(ydamir))
ydamir=(scaler4.transform(ydamir))
#outputtrain=np.arcsinh(outputtrain)
outputtest4=np.reshape((output[300001:numrows,3]),(-1,1))

#(scaler.fit(ydami))
#ydami=(scaler.transform(ydami))

numrowstest=len(outputtest4)    # rows of inout
numcolstest = len(outputtest4[0])

matrix=np.concatenate((inputtrain,ydami), axis=1)
#ruuth = np.int(input("Enter the maximum number of clusters you want to accomodate for 4th machine: ") )

nclusters=ruuth
print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=ruuth,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
dd=dd.T
dd=np.reshape(dd,(-1,1))

filename2= 'Claffiermodel4.asv'

#ddtest=np.reshape(ddtest,(-1,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))

print(' Learn the classifer from the predicted labels from Kmeans')

outputtrainclasss = to_categorical(outputtrainclass)

labelDA=deep_learningclass(inputtrainclass, outputtrainclasss,filename2)
labelDA=np.argmax(labelDA, axis=-1) 
#labelDA=run_model(modelclass,inputtrainclass, outputtrainclass,inputtest,filename2)

X_train=inputtrain
X_test=inputtest
y_train=dd
y_test=dd


np.savetxt('y_train4.out',y_train, fmt = '%4.6f', newline = '\n')
np.savetxt('X_train4.out',X_train, fmt = '%4.6f', newline = '\n')
np.savetxt('y_traind4.out',ydamir, fmt = '%4.6f', newline = '\n')
np.savetxt('labelDA4.out',labelDA, fmt = '%d', newline = '\n')
np.savetxt('inputtest4.out',inputtest, fmt = '%4.6f', newline = '\n')

a = open('numrowstest4.out', 'w')
#a.write('%4.6f',numrowstest)
a.write("%i \n" % (numrowstest))
a.close()

#-------------------Regression---------------------------------------------------------------#
print('Learn regression of the clusters with different labels from k-means ' )

#clementanswer=np.zeros((numrowstest,1))

print('Start the regression')

oldfolder = os.getcwd()
os.chdir(oldfolder)

print('')
##Creating Folders and Copying Simulation Datafiles
#print(' Creating the folders and copying the simulation files for the forward problem ')
print('')
for j in range(nclusters):
    folder = 'CCRmodel4_%d'%(j)
    if os.path.isdir(folder): # value of os.path.isdir(directory) = True
        shutil.rmtree(folder)      
    os.mkdir(folder)
    shutil.copy2('y_train4.out',folder)
    shutil.copy2('X_train4.out',folder)
    shutil.copy2('y_traind4.out',folder)


    shutil.copy2('numrowstest4.out',folder)

print( ' Solving the prediction problem' )
print('')
os.chdir(oldfolder)
##
import multiprocessing

import numpy as np
#from sklearn.neural_network import MLPRegressor

import os
from joblib import Parallel, delayed

filename4='regressor4.asv'
#
def parad4(j):
    
    oldfolder = os.getcwd()
    folder = 'CCRmodel4_%d'%(j)
    os.chdir(folder)
    y_train=np.genfromtxt("y_train4.out", dtype='float')

    y_train = np.reshape(y_train,(-1,1), 'F') 
    
    
    X_train = np.genfromtxt("X_train4.out", dtype='float') #533051 by 28
   
    X_train = np.reshape(X_train,(-1,numruth), 'F') 
    
    y_traind = np.genfromtxt("y_traind4.out", dtype='float') #533051 by 28
   
    y_traind = np.reshape(y_traind,(-1,1), 'F') 

    label0=(np.asarray(np.where(y_train == j))).T    
#    model0 = RandomForestRegressor(n_estimators=2000)
    model0 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
    a0=X_train[label0,:]
    a0=np.reshape(a0,(-1,numruth),'F')

    b0=y_traind[label0,:]
    b0=np.reshape(b0,(-1,1),'F')
    
    model0.fit(a0, b0)
    pickle.dump(model0, open(filename4, 'wb'))
    os.chdir(oldfolder)
    print(" cluster %d has been processed"%(j))
    
print('')
print('cluster in parallel')
print('')
clementanswer4=np.zeros((numrowstest,1))    
num_cores = multiprocessing.cpu_count()
number_of_realisations = range(nclusters)
Parallel(n_jobs=nclusters, verbose=50)(delayed(
    parad4)(j)for j in number_of_realisations)


print('')
print('predict in series')
print('')
for i in range(nclusters):
    folder = 'CCRmodel4_%d'%(i)
    os.chdir(folder)    
    loaded_model = pickle.load(open(filename4, 'rb'))

    
    labelDA0=(np.asarray(np.where(labelDA == i))).T


##----------------------##------------------------##
    a00=inputtest[labelDA0,:]
    a00=np.reshape(a00,(-1,numruth),'F')
    if a00.shape[0]!=0:
        clementanswer4[np.ravel(labelDA0),:]=np.reshape(loaded_model .predict(a00),(-1,1))
    os.chdir(oldfolder)

    
clementanswer4=scaler4.inverse_transform(clementanswer4)
print('End of building machine for output 4')

##----------------------------Output 5--------------------------------------------------------

print('Build the machine for output 5')
outputtrain=np.reshape((output[0:300000,4]),(-1,1)) #select the first 300000 for training
ydami=outputtrain
ydamir=outputtrain
scaler5 = MinMaxScaler()
(scaler5.fit(ydamir))
ydamir=(scaler5.transform(ydamir))
#outputtrain=np.arcsinh(outputtrain)
outputtest5=np.reshape((output[300001:numrows,4]),(-1,1))

#(scaler.fit(ydami))
#ydami=(scaler.transform(ydami))

numrowstest=len(outputtest5)    # rows of inout
numcolstest = len(outputtest5[0])

matrix=np.concatenate((inputtrain,ydami), axis=1)
#ruuth = np.int(input("Enter the maximum number of clusters you want to accomodate for 5th machine: ") )

nclusters=ruuth
print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=ruuth,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
dd=dd.T
dd=np.reshape(dd,(-1,1))

filename2= 'Claffiermodel5.asv'
#deep_learningclass(inputtrainclass, outputtrainclasss,filename2)


#ddtest=np.reshape(ddtest,(-1,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))

print(' Learn the classifer from the predicted labels from Kmeans')

outputtrainclasss = to_categorical(outputtrainclass)

labelDA=deep_learningclass(inputtrainclass, outputtrainclasss,filename2)
labelDA=np.argmax(labelDA, axis=-1) 
#labelDA=run_model(modelclass,inputtrainclass, outputtrainclass,inputtest,filename2)

X_train=inputtrain
X_test=inputtest
y_train=dd
y_test=dd


np.savetxt('y_train5.out',y_train, fmt = '%4.6f', newline = '\n')
np.savetxt('X_train5.out',X_train, fmt = '%4.6f', newline = '\n')
np.savetxt('y_traind5.out',ydamir, fmt = '%4.6f', newline = '\n')
np.savetxt('labelDA5.out',labelDA, fmt = '%d', newline = '\n')
np.savetxt('inputtest5.out',inputtest, fmt = '%4.6f', newline = '\n')

a = open('numrowstest5.out', 'w')
#a.write('%4.6f',numrowstest)
a.write("%i \n" % (numrowstest))
a.close()

#-------------------Regression----------------#
print('Learn regression of the clusters with different labels from k-means ' )

#clementanswer=np.zeros((numrowstest,1))

print('Start the regression')

oldfolder = os.getcwd()
os.chdir(oldfolder)

print('')
##Creating Folders and Copying Simulation Datafiles
#print(' Creating the folders and copying the simulation files for the forward problem ')
print('')
for j in range(nclusters):
    folder = 'CCRmodel5_%d'%(j)
    if os.path.isdir(folder): # value of os.path.isdir(directory) = True
        shutil.rmtree(folder)      
    os.mkdir(folder)
    shutil.copy2('y_train5.out',folder)
    shutil.copy2('X_train5.out',folder)
    shutil.copy2('y_traind5.out',folder)


    shutil.copy2('numrowstest5.out',folder)

print( ' Solving the prediction problem' )
print('')
os.chdir(oldfolder)
##
import multiprocessing

import numpy as np
#from sklearn.neural_network import MLPRegressor

import os
from joblib import Parallel, delayed

filename5='regressor5.asv'
#
def parad5(j):
    
    oldfolder = os.getcwd()
    folder = 'CCRmodel5_%d'%(j)
    os.chdir(folder)
    y_train=np.genfromtxt("y_train5.out", dtype='float')

    y_train = np.reshape(y_train,(-1,1), 'F') 
    
    
    X_train = np.genfromtxt("X_train5.out", dtype='float') #533051 by 28
   
    X_train = np.reshape(X_train,(-1,numruth), 'F') 
    
    y_traind = np.genfromtxt("y_traind5.out", dtype='float') #533051 by 28
   
    y_traind = np.reshape(y_traind,(-1,1), 'F') 

    label0=(np.asarray(np.where(y_train == j))).T    
#    model0 = RandomForestRegressor(n_estimators=2000)
    model0 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
    a0=X_train[label0,:]
    a0=np.reshape(a0,(-1,numruth),'F')

    b0=y_traind[label0,:]
    b0=np.reshape(b0,(-1,1),'F')
    
    model0.fit(a0, b0)
    pickle.dump(model0, open(filename5, 'wb'))
    os.chdir(oldfolder)
    print(" cluster %d has been processed"%(j))
    
print('')
print('cluster in parallel')
print('')
clementanswer5=np.zeros((numrowstest,1))    
num_cores = multiprocessing.cpu_count()
number_of_realisations = range(nclusters)
Parallel(n_jobs=nclusters, verbose=50)(delayed(
    parad5)(j)for j in number_of_realisations)


print('')
print('predict in series')
print('')
for i in range(nclusters):
    folder = 'CCRmodel5_%d'%(i)
    os.chdir(folder)    
    loaded_model = pickle.load(open(filename5, 'rb'))

    
    labelDA0=(np.asarray(np.where(labelDA == i))).T


##----------------------##------------------------##
    a00=inputtest[labelDA0,:]
    a00=np.reshape(a00,(-1,numruth),'F')
    if a00.shape[0]!=0:
        clementanswer5[np.ravel(labelDA0),:]=np.reshape(loaded_model .predict(a00),(-1,1))
    os.chdir(oldfolder)

    
clementanswer5=scaler5.inverse_transform(clementanswer5)
print('End of building machine for output 5')

##----------------------------Output 6--------------------------------------------------------

print('Build the machine for output 6')
outputtrain=np.reshape((output[0:300000,5]),(-1,1)) #select the first 300000 for training
ydami=outputtrain
ydamir=outputtrain
scaler6 = MinMaxScaler()
(scaler6.fit(ydamir))
ydamir=(scaler6.transform(ydamir))
#outputtrain=np.arcsinh(outputtrain)
outputtest6=np.reshape((output[300001:numrows,5]),(-1,1))

#(scaler.fit(ydami))
#ydami=(scaler.transform(ydami))

numrowstest=len(outputtest6)    # rows of inout
numcolstest = len(outputtest6[0])

matrix=np.concatenate((inputtrain,ydami), axis=1)
#ruuth = np.int(input("Enter the maximum number of clusters you want to accomodate for 6th machine: ") )

nclusters=ruuth
print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=ruuth,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
dd=dd.T
dd=np.reshape(dd,(-1,1))

filename2= 'Claffiermodel6.asv'
#ddtest=np.reshape(ddtest,(-1,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))

print(' Learn the classifer from the predicted labels from Kmeans')

outputtrainclasss = to_categorical(outputtrainclass)

labelDA=deep_learningclass(inputtrainclass, outputtrainclasss,filename2)
labelDA=np.argmax(labelDA, axis=-1) 
#labelDA=run_model(modelclass,inputtrainclass, outputtrainclass,inputtest,filename2)

X_train=inputtrain
X_test=inputtest
y_train=dd
y_test=dd


np.savetxt('y_train6.out',y_train, fmt = '%4.6f', newline = '\n')
np.savetxt('X_train6.out',X_train, fmt = '%4.6f', newline = '\n')
np.savetxt('y_traind6.out',ydamir, fmt = '%4.6f', newline = '\n')
np.savetxt('labelDA6.out',labelDA, fmt = '%d', newline = '\n')
np.savetxt('inputtest6.out',inputtest, fmt = '%4.6f', newline = '\n')

a = open('numrowstest6.out', 'w')
#a.write('%4.6f',numrowstest)
a.write("%i \n" % (numrowstest))
a.close()

#-------------------Regression----------------#
print('Learn regression of the clusters with different labels from k-means ' )

#clementanswer=np.zeros((numrowstest,1))

print('Start the regression')

oldfolder = os.getcwd()
os.chdir(oldfolder)

print('')
##Creating Folders and Copying Simulation Datafiles
#print(' Creating the folders and copying the simulation files for the forward problem ')
print('')
for j in range(nclusters):
    folder = 'CCRmodel6_%d'%(j)
    if os.path.isdir(folder): # value of os.path.isdir(directory) = True
        shutil.rmtree(folder)      
    os.mkdir(folder)
    shutil.copy2('y_train6.out',folder)
    shutil.copy2('X_train6.out',folder)
    shutil.copy2('y_traind6.out',folder)


    shutil.copy2('numrowstest6.out',folder)

print( ' Solving the prediction problem' )
print('')
os.chdir(oldfolder)
##
import multiprocessing

import numpy as np
#from sklearn.neural_network import MLPRegressor

import os
from joblib import Parallel, delayed

filename6='regressor6.asv'
#
def parad6(j):
    
    oldfolder = os.getcwd()
    folder = 'CCRmodel6_%d'%(j)
    os.chdir(folder)
    y_train=np.genfromtxt("y_train6.out", dtype='float')

    y_train = np.reshape(y_train,(-1,1), 'F') 
    
    
    X_train = np.genfromtxt("X_train6.out", dtype='float') #533051 by 28
   
    X_train = np.reshape(X_train,(-1,numruth), 'F') 
    
    y_traind = np.genfromtxt("y_traind6.out", dtype='float') #533051 by 28
   
    y_traind = np.reshape(y_traind,(-1,1), 'F') 

    label0=(np.asarray(np.where(y_train == j))).T    
#    model0 = RandomForestRegressor(n_estimators=2000)
    model0 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
    a0=X_train[label0,:]
    a0=np.reshape(a0,(-1,numruth),'F')

    b0=y_traind[label0,:]
    b0=np.reshape(b0,(-1,1),'F')
    
    model0.fit(a0, b0)
    pickle.dump(model0, open(filename6, 'wb'))
    os.chdir(oldfolder)
    print(" cluster %d has been processed"%(j))
    
print('')
print('cluster in parallel')
print('')
clementanswer6=np.zeros((numrowstest,1))    
num_cores = multiprocessing.cpu_count()
number_of_realisations = range(nclusters)
Parallel(n_jobs=nclusters, verbose=50)(delayed(
    parad6)(j)for j in number_of_realisations)


print('')
print('predict in series')
print('')
for i in range(nclusters):
    folder = 'CCRmodel6_%d'%(i)
    os.chdir(folder)    
    loaded_model = pickle.load(open(filename6, 'rb'))

    
    labelDA0=(np.asarray(np.where(labelDA == i))).T


##----------------------##------------------------##
    a00=inputtest[labelDA0,:]
    a00=np.reshape(a00,(-1,numruth),'F')
    if a00.shape[0]!=0:
        clementanswer6[np.ravel(labelDA0),:]=np.reshape(loaded_model .predict(a00),(-1,1))
    os.chdir(oldfolder)

    
clementanswer6=scaler6.inverse_transform(clementanswer6)
print('End of building the machine for output 6')
print('')
print('End of building the machines for the 6 outputs')
CCR=np.concatenate((clementanswer1,clementanswer2,clementanswer3,clementanswer4,clementanswer5,clementanswer6), axis=1)
np.savetxt('CCR.out',CCR, fmt = '%4.8f', newline = '\n')

Trued=np.concatenate((outputtest1,outputtest2,outputtest3,outputtest4,outputtest5,outputtest6), axis=1)
np.savetxt('True.out',Trued, fmt = '%4.8f', newline = '\n')
print('')
print(' Compute L2 and R2 for the 6 machine')
#
print('For output 1')
outputtest1 = np.reshape(outputtest1, (numrowstest, 1))
Lerrorsparse=(LA.norm(outputtest1-clementanswer1)/LA.norm(outputtest1))**0.5
L_21=1-(Lerrorsparse**2)
#Coefficient of determination
outputreq=np.zeros((numrowstest,1))
for i in range(numrowstest):
    outputreq[i,:]=outputtest1[i,:]-np.mean(outputtest1)

#outputreq=outputreq.T
CoDspa=1-(LA.norm(outputtest1-clementanswer1)/LA.norm(outputreq))
CoD1=1 - (1-CoDspa)**2 ;
print ('R2 of fit using the machine for output 1 is :', CoD1)
print ('L2 of fit using the machine for output 1 is :', L_21)
print('')

print('For output 2')
outputtest2 = np.reshape(outputtest2, (numrowstest, 1))
Lerrorsparse=(LA.norm(outputtest2-clementanswer2)/LA.norm(outputtest2))**0.5
L_22=1-(Lerrorsparse**2)
#Coefficient of determination
outputreq=np.zeros((numrowstest,1))
for i in range(numrowstest):
    outputreq[i,:]=outputtest2[i,:]-np.mean(outputtest2)

#outputreq=outputreq.T
CoDspa=1-(LA.norm(outputtest2-clementanswer2)/LA.norm(outputreq))
CoD2=1 - (1-CoDspa)**2 ;
print ('R2 of fit using the machine for output 2 is :', CoD2)
print ('L2 of fit using the machine for output 2 is :', L_22)
print('')


print('For output 3')
outputtest3 = np.reshape(outputtest3, (numrowstest, 1))
Lerrorsparse=(LA.norm(outputtest3-clementanswer3)/LA.norm(outputtest3))**0.5
L_23=1-(Lerrorsparse**2)
#Coefficient of determination
outputreq=np.zeros((numrowstest,1))
for i in range(numrowstest):
    outputreq[i,:]=outputtest3[i,:]-np.mean(outputtest3)

#outputreq=outputreq.T
CoDspa=1-(LA.norm(outputtest3-clementanswer3)/LA.norm(outputreq))
CoD3=1 - (1-CoDspa)**2 ;
print ('R2 of fit using the machine for output 3 is :', CoD3)
print ('L2 of fit using the machine for output 3 is :', L_23)
print('')

print('For output 4')
outputtest4 = np.reshape(outputtest4, (numrowstest, 1))
Lerrorsparse=(LA.norm(outputtest4-clementanswer4)/LA.norm(outputtest4))**0.5
L_24=1-(Lerrorsparse**2)
#Coefficient of determination
outputreq=np.zeros((numrowstest,1))
for i in range(numrowstest):
    outputreq[i,:]=outputtest4[i,:]-np.mean(outputtest4)

#outputreq=outputreq.T
CoDspa=1-(LA.norm(outputtest4-clementanswer4)/LA.norm(outputreq))
CoD4=1 - (1-CoDspa)**2 ;
print ('R2 of fit using the machine for output 4 is :', CoD4)
print ('L2 of fit using the machine for output 4 is :', L_24)
print('')

print('For output 5')
outputtest5 = np.reshape(outputtest5, (-1, 1))
Lerrorsparse=(LA.norm(outputtest5-clementanswer5)/LA.norm(outputtest5))**0.5
L_25=1-(Lerrorsparse**2)
#Coefficient of determination
outputreq=np.zeros((numrowstest,1))
for i in range(numrowstest):
    outputreq[i,:]=outputtest5[i,:]-np.mean(outputtest5)

#outputreq=outputreq.T
CoDspa=1-(LA.norm(outputtest5-clementanswer5)/LA.norm(outputreq))
CoD5=1 - (1-CoDspa)**2 ;
print ('R2 of fit using the machine for output 5 is :', CoD5)
print ('L2 of fit using the machine for output 5 is :', L_25)
print('')

print('For output 6')
outputtest6 = np.reshape(outputtest6, (-1, 1))
Lerrorsparse=(LA.norm(outputtest6-clementanswer6)/LA.norm(outputtest6))**0.5
L_26=1-(Lerrorsparse**2)
#Coefficient of determination
outputreq=np.zeros((numrowstest,1))
for i in range(numrowstest):
    outputreq[i,:]=outputtest6[i,:]-np.mean(outputtest6)

#outputreq=outputreq.T
CoDspa=1-(LA.norm(outputtest6-clementanswer6)/LA.norm(outputreq))
CoD6=1 - (1-CoDspa)**2 ;
print ('R2 of fit using the machine for output 6 is :', CoD6)
print ('L2 of fit using the machine for output 6 is :', L_26)
print('')

print('Plot figures now')
print('')
fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,1)
plt.plot(outputtest1[0:50000,:], color = 'red', label = 'Real data')
plt.plot(clementanswer1[0:50000,:], color = 'blue', label = 'Predicted data from Machine')

plt.title('OUT_tur_STRESS_TOR_i', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
#plt.axis([0,(100),0,(100)])
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
plt.legend()
plt.show()



fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,2)
plt.plot(outputtest2[0:50000,:], color = 'red', label = 'Real data')
plt.plot(clementanswer2[0:50000,:], color = 'blue', label = 'Predicted data from Machine')

plt.title('OUT_tur_ENERGY_FLUX_i', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
#plt.axis([0,(100),0,(100)])
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
plt.legend()
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,3)
plt.plot(outputtest3[0:50000,:], color = 'red', label = 'Real data')
plt.plot(clementanswer3[0:50000,:], color = 'blue', label = 'Predicted data from Machine')

plt.title('OUT_tur_PARTICLE_FLUX_2', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
#plt.axis([0,(100),0,(100)])
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
plt.legend()
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,4)
plt.plot(outputtest4[0:50000,:], color = 'red', label = 'Real data')
plt.plot(clementanswer4[0:50000,:], color = 'blue', label = 'Predicted data from Machine')

plt.title('OUT_tur_PARTICLE_FLUX_1', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
#plt.axis([0,(100),0,(100)])
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
plt.legend()
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,5)
plt.plot(outputtest5[0:50000,:], color = 'red', label = 'Real data')
plt.plot(clementanswer5[0:50000,:], color = 'blue', label = 'Predicted data from Machine')

plt.title('OUT_tur_ENERGY_FLUX_1', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
#plt.axis([0,(10000),0,(10000)])
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
plt.legend()
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,6)
plt.plot(outputtest6[0:50000,:], color = 'red', label = 'Real data')
plt.plot(clementanswer6[0:50000,:], color = 'blue', label = 'Predicted data from Machine')

plt.title('OUT_tur_PARTICLE_FLUX_3', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
#plt.axis([0,(100),0,(100)])
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
plt.legend()
plt.show()


print('')
#
print('Plot figures')
fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,1)
plt.scatter(clementanswer1[0:50000,:],outputtest1[0:50000,:], color ='c')
plt.title('OUT_tur_STRESS_TOR_i', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
#plt.axis([0,(100),0,(100)])
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,2)
plt.scatter(clementanswer2[0:50000,:],outputtest2[0:50000,:], color ='c')
plt.title('OUT_tur_ENERGY_FLUX_i', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
#plt.axis([0,(100),0,(100)])
plt.show()



fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,3)
plt.scatter(clementanswer3[0:50000,:],outputtest3[0:50000,:], color ='c')
plt.title('OUT_tur_PARTICLE_FLUX_2', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
#plt.axis([0,(100),0,(100)])
plt.show()




fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,4)
plt.scatter(clementanswer4[0:50000,:],outputtest4[0:50000,:], color ='c')
plt.title('OUT_tur_PARTICLE_FLUX_1', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
#plt.axis([0,(100),0,(100)])
plt.show()

fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,5)
plt.scatter(clementanswer5[0:50000,:],outputtest5[0:50000,:], color ='c')
plt.title('OUT_tur_ENERGY_FLUX_1', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
#plt.axis([0,(100),0,(100)])
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,6)
plt.scatter(clementanswer6[0:50000,:],outputtest6[0:50000,:], color ='c')
plt.title('OUT_tur_PARTICLE_FLUX_3', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
#plt.axis([0,(100),0,(100)])
plt.show()

CoDoverall=(CoD6+CoD5+CoD4+CoD3+CoD2+CoD1)/6
R2overall=(L_26+L_25+L_24+L_23+L_22+L_21)/6
print ('Overall R2 of fit using the CCR machine is :', CoDoverall)
print ('Overall L2 of fit using the CCR machine is :', R2overall)

##---------------------Predict on seen data now-------------------------------##

print('end of program')






