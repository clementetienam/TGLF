# -*- coding: utf-8 -*-
"""
Created on Tuesday June 06 12:05:47 2019

@author: Dr Clement Etienam
Supervisor: Professor Kody Law
Prediction of test data on the TGLF-CCR pre-trained model
"""
from __future__ import print_function
print(__doc__)
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import shutil
import multiprocessing
from scipy.stats import rankdata, norm
from numpy import linalg as LA

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

print( 'Standalone prediction for TGLF ')
oldfolder = os.getcwd()
cores = multiprocessing.cpu_count()
print(' ')
print(' This computer has %d cores, which will all be utilised in parallel '%cores)
#print(' The number of cores to be utilised can be changed in runeclipse.py and writefiles.py ')
print(' ')

start = datetime.datetime.now()
print(str(start))


#------------------Begin Code-----------------------------------------------------------------#
print('Load the input data you want to predict from')
#testdata=np.genfromtxt("testdata.out", dtype='float') # If you have your own test data simply slot in here
#testdata= np.reshape(tesdata,(-1,22), 'F') 
sgsim = open("orso.out") #533051 by 28
sgsim = np.fromiter(sgsim,float)
data = np.reshape(sgsim,(385747,28), 'F')

testdata=data[0:300000,0:22] # for now we will use this
print('Standardize and normalize (make gaussian) the test data')

numrows1=len(testdata) 
newbig=np.zeros((numrows1,22))
for i in range(22):
    input11=testdata[:,i]
    newX = norm.ppf(rankdata(input11)/(len(input11) + 1))
    newbig[:,i]=newX.T

input1=newbig
scaler = MinMaxScaler()
(scaler.fit(input1))
input1=(scaler.transform(input1))
numrows=len(input1)    # rows of input
numrowstest=numrows
numcols = len(input1[0])
numruth = numcols
inputtest=input1

output=data[:,22:29]
print(' Determine how many clusters were used for training CCR')
ruuth = len([i for i in os.listdir(oldfolder) if os.path.isdir(i)])
ruuth=ruuth/6 # We divide by 6 because CCR was done individually for the 6 outputs
nclusters=int(ruuth)

print('For the first output')
outputtrain=np.reshape((output[0:300000,0]),(-1,1)) #select the first 300000 for training
ydamir=outputtrain
outputtrain1=outputtrain
scaler1 = MinMaxScaler()
(scaler1.fit(ydamir))
ydamir=(scaler1.transform(ydamir))

#outputtrain=np.arcsinh(outputtrain)
outputtest1=np.reshape((output[300001:numrows,0]),(-1,1))
numcolstest = 1


filename1= 'Claffiermodel1.asv'
loaded_model = pickle.load(open(filename1, 'rb'))
labelDA = loaded_model.predict(inputtest)
labelDA=np.argmax(labelDA, axis=-1) 

#-------------------Regression prediction---------------------------------------------------#

oldfolder = os.getcwd()
os.chdir(oldfolder)
filename1='regressor1.asv'
clementanswer1=np.zeros((numrowstest,1))    
print('predict in series')
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
print('End of prediction for output 1')
print('')
##----------------------------Output 2--------------------------------------------------------##

print('Predict for output 2')
outputtrain=np.reshape((output[0:300000,1]),(-1,1)) #select the first 300000 for training
ydami=outputtrain

ydamir=outputtrain
outputtrain2=outputtrain
scaler2 = MinMaxScaler()
(scaler2.fit(ydamir))
ydamir=(scaler2.transform(ydamir))

filename2= 'Claffiermodel2.asv'
loaded_model = pickle.load(open(filename2, 'rb'))
labelDA = loaded_model.predict(inputtest)
labelDA=np.argmax(labelDA, axis=-1) 
#-------------------#---------------------------------#

filename2='regressor2.asv'

clementanswer2=np.zeros((numrowstest,1))    

print('predict in series')

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
print('End of prediction for output 2')
print('')
##----------------------------Output 3--------------------------------------------------------

print('Predict for for output 3')
outputtrain=np.reshape((output[0:300000,2]),(-1,1)) #select the first 300000 for training
ydami=outputtrain
outputtrain3=outputtrain
ydamir=outputtrain
scaler3 = MinMaxScaler()
(scaler3.fit(ydamir))
ydamir=(scaler3.transform(ydamir))

filename2= 'Claffiermodel3.asv'

loaded_model = pickle.load(open(filename2, 'rb'))
labelDA = loaded_model.predict(inputtest)
labelDA=np.argmax(labelDA, axis=-1) 
#-------------------#----------------------------------------------------------#

oldfolder = os.getcwd()
os.chdir(oldfolder)

filename3='regressor3.asv'

clementanswer3=np.zeros((numrowstest,1))    
print('predict in series')
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
print('End of prediction for output 3')
print('')

##----------------------------Output 4--------------------------------------------------------

print('Build the machine for output 4')
outputtrain=np.reshape((output[0:300000,3]),(-1,1)) #select the first 300000 for training
ydami=outputtrain
outputtrain4=outputtrain
ydamir=outputtrain
scaler4 = MinMaxScaler()
(scaler4.fit(ydamir))
ydamir=(scaler4.transform(ydamir))

filename2= 'Claffiermodel4.asv'
loaded_model = pickle.load(open(filename2, 'rb'))
labelDA = loaded_model.predict(inputtest)
labelDA=np.argmax(labelDA, axis=-1) 

#-------------------Regression---------------------------------------------------------------#
oldfolder = os.getcwd()
os.chdir(oldfolder)

filename4='regressor4.asv'
clementanswer4=np.zeros((numrowstest,1))    
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
print('End of prediction for output 4')
print('')
##----------------------------Output 5--------------------------------------------------------

print('Predict for output 5')
outputtrain=np.reshape((output[0:300000,4]),(-1,1)) #select the first 300000 for training
ydami=outputtrain
outputtrain5=outputtrain
ydamir=outputtrain
scaler5 = MinMaxScaler()
(scaler5.fit(ydamir))
ydamir=(scaler5.transform(ydamir))

filename2= 'Claffiermodel5.asv'
loaded_model = pickle.load(open(filename2, 'rb'))
labelDA = loaded_model.predict(inputtest)
labelDA=np.argmax(labelDA, axis=-1) 

#-------------------Regression----------------#
oldfolder = os.getcwd()
os.chdir(oldfolder)

filename5='regressor5.asv'

clementanswer5=np.zeros((numrowstest,1))    
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
print('Finished prediction for output 5')
print('')

##----------------------------Output 6--------------------------------------------------------

print('Prediction for output 6')
outputtrain=np.reshape((output[0:300000,5]),(-1,1)) #select the first 300000 for training
ydami=outputtrain
outputtrain6=outputtrain
ydamir=outputtrain
scaler6 = MinMaxScaler()
(scaler6.fit(ydamir))
ydamir=(scaler6.transform(ydamir))
#
filename2= 'Claffiermodel6.asv'
loaded_model = pickle.load(open(filename2, 'rb'))
labelDA = loaded_model.predict(inputtest)
labelDA=np.argmax(labelDA, axis=-1) 
#ddtest=np.reshape(ddtest,(-1,1))
#-------------------#---------------------------------#


oldfolder = os.getcwd()
os.chdir(oldfolder)
filename6='regressor6.asv'
#

clementanswer6=np.zeros((numrowstest,1))    

print('predict in series')

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
print('End of Prediction for output 6')
print('')
outputtest1=outputtrain1
outputtest2=outputtrain2
outputtest3=outputtrain3
outputtest4=outputtrain4
outputtest5=outputtrain5
outputtest6=outputtrain6

print('Save predictions to file')
valueCCR2=np.concatenate((clementanswer1,clementanswer2,clementanswer3,clementanswer4,clementanswer5,clementanswer6), axis=1)
np.savetxt('CCRprediction.out',valueCCR2, fmt = '%4.6f', newline = '\n')

Trued=np.concatenate((outputtest1,outputtest2,outputtest3,outputtest4,outputtest5,outputtest6), axis=1)
np.savetxt('Trueexamp.out',Trued, fmt = '%4.6f', newline = '\n')


def Performance_plot(CCR,Trued):
	print(' Compute L2 and R2 for the 6 machine')
	clementanswer1=np.reshape(CCR[0:50000,0],(-1,1))
	clementanswer2=np.reshape(CCR[0:50000,1],(-1,1))
	clementanswer3=np.reshape(CCR[0:50000,2],(-1,1))
	clementanswer4=np.reshape(CCR[0:50000,3],(-1,1))
	clementanswer5=np.reshape(CCR[0:50000,4],(-1,1))
	clementanswer6=np.reshape(CCR[0:50000,5],(-1,1))
	
	outputtest1=np.reshape(Trued[0:50000,0],(-1,1))
	outputtest2=np.reshape(Trued[0:50000,1],(-1,1))
	outputtest3=np.reshape(Trued[0:50000,2],(-1,1))
	outputtest4=np.reshape(Trued[0:50000,3],(-1,1))
	outputtest5=np.reshape(Trued[0:50000,4],(-1,1))
	outputtest6=np.reshape(Trued[0:50000,5],(-1,1))
	numrowstest=len(outputtest6)
	print('For output 1')
	outputtest1 = np.reshape(outputtest1, (-1, 1))
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
	outputtest2 = np.reshape(outputtest2, (-1, 1))
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
	outputtest3 = np.reshape(outputtest3, (-1, 1))
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
	outputtest4 = np.reshape(outputtest4, (-1, 1))
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
	
	
	CoDoverall=(CoD5+CoD4+CoD3+CoD2)/4
	R2overall=(L_25+L_24+L_23+L_22)/4
    
      
	CoDview=np.zeros((1,4))
	R2view=np.zeros((1,4))
    
	#CoDview[:,0]=CoD1
	CoDview[:,0]=CoD2
	CoDview[:,1]=CoD3
	CoDview[:,2]=CoD4
	CoDview[:,3]=CoD5
	#CoDview[:,5]=CoD6
	
	#R2view[:,0]=L_21
	R2view[:,0]=L_22
	R2view[:,1]=L_23
	R2view[:,2]=L_24
	R2view[:,3]=L_25
	#R2view[:,5]=L_26
	
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
	
	return CoDoverall,R2overall,CoDview,R2view
	
CoDoverall,L_2overall,CoDview,R2view=Performance_plot(valueCCR2,Trued)
print ('Overall R2 of fit using the CCR machine is :', CoDoverall)
print ('Overall L2 of fit using the CCR machine is :', L_2overall)
print('')

##---------------------End of Program-------------------------------##

print('end of program')






