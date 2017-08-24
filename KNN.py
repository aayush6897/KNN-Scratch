#importing necessary libraries
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import pandas as pd

#loading dataset using Pandas in pandas dataframe
data=pd.read_csv('/home/aayush/MLearning/datasets/iris.csv')
data.head()       #prints the top 5 values in dataframe

#converting pandas dataframe into numpy array
data=np.asarray(data)
#shuffling the data
np.random.shuffle(data)


#to split dataset into Training and Testing parts
split=int(0.8 * data.shape[0])     #da.shape returns the shape of da array

xtrain=data[:split,:-1]           #xtrain contains features of our training set
xtest=data[split:,:-1]            #xtest contains features of out testing set

ytrain=data[:split,-1]             #ytrain contain labels of our training set
ytest=data[split:,-1]              #ytest contain labels of our testing set


#printing the shape of array after splitting
print(xtrain.shape,xtest.shape)
print(ytrain.shape,ytest.shape)


#function to calculate distance
def dist(x1,x2):
    d=np.sqrt(((x1-x2)**2).sum())      #here we are calculatng mean square error
    return d


# function to predict the possible label
def knearn(xtrain,ytrain,xtest,k=5):        #function takes the training set features, labels  
                                            #and testing feature to predict its label, it also 
                                            #takes k as parameter if you wish to change.
    vals=[]                                    #empty list to store labels and corresponding distance
    for i in range(xtrain.shape[0]):
        d=dist(xtrain[i],xtest)                #calculate distance btw testing and training set
        vals.append([d,ytrain[i]])             #append distance and corresponding labels as a list to main list
        
        
    sorted_list=sorted(vals,key=lambda x:x[0])     #sort the list wrt. to distance
    sorted_list=np.asarray(sorted_list)                #to convert list into numpy array
    
    k_neighbours=sorted_list[:k,-1]             #choose top k values from list and append to new list neighbours
    
    frequency=np.unique(k_neighbours,return_counts=True)  #to get unique labels from neighbours and their count
    
    max_count=frequency[1].argmax()    #to get the index of the unique labels occuring maximun no. of time
    
    return(frequency[0][max_count])    #to return the label corresponding maximum count



#function to get the accuracy of our algorithm on testing set.
def gacc(k=5):
    pred=[]
    for iac in range(xtest.shape[0]):
        pred.append(knearn(xtrain,ytrain,xtest[iac],k)) #to predict labels of all the testing data
        
    pred=np.asarray(pred)     #converting them to numpy array
    
    for ig in range(pred.shape[0]):    #to print true if predicted labels match original labels otherwise false
        if(pred[ig]==ytest[ig]):
            print("True")
        else:
            print("False")
            
    return 100*float((ytest == pred).sum())/pred.shape[0]   #to print tht total accuracy of our algorithm



print(gacc(5))
