# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 20:26:58 2018

@author: Erman
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

class OilRegression():
    
# Object variables     
    def __init__(self):
        
        self.df=pd.read_csv("data.csv")
        self.X=self.df.drop(['target_00','target_01'], axis=1).values.tolist()
        self.y=None
        self.max_iter=1000000
        self.pca=PCA()
        self.explainedVariance=[]
        self.kFold=3
        self.modelList=[]
        self.validationErrors=[]
        self.degree=3
        self.models=[]
        self.label=1
        self.clf = linear_model.Lasso(max_iter=self.max_iter)
        self.model=None
        self.splitRatio=0.33
        self.trainX=[]
        self.trainY=[]
        self.testY=[]
        self.testY=[]
        self.validateX=[]
        self.validateY=[]
        self.results=None
        self.finalError=0
        self.modelType=1
        
#Let's see how features are correlated . It turns out that they are highly correlated. First two translated components pretty much explain everything.       
#This method plots explained variances of translated components. Reconnaissance.       
    def PCATransform(self):
        self.pca.fit(self.X)
        self.explainedVariance=self.pca.explained_variance_
        plt.plot(self.explainedVariance)
        plt.title("Variances in Decreasing Order", fontsize=15)
        plt.show()
        
#Get the info and put it in. 
#First enter the column to be predicted, either 1 or 2.
# Then enter the type of regression, 1 for Linear Regression, 2 for Non-Linear Regression.
#Lastly, the degree of the model for the non-linear regression.      
    def getInfo(self):
        self.label=int(input("Enter the number of the label to be predicted. 1 or 2 : "))
        self.modelType=int(input("Enter 1 for a Linear Model(Lasso) and 2 for a Non-Linear model: "))
        
        if self.modelType ==2:
            self.degree=int(input("Enter the degree of the NL model (2-3) : "))
            
            if self.degree ==2:
                self.max_iter=10000
            else:
                self.max_iter=7500
        self.clf = linear_model.Lasso(max_iter=self.max_iter)    

# Split training and test sets for a label, i.e either label 1 or label 2        
    def trainTestSplit(self):
        
        if self.label==1:
            self.y=self.df[['target_00']].values.tolist()
            
        elif self.label==2:
            self.y=self.df[['target_01']].values.tolist()
            
        self.trainX, self.testX,self.trainY, self.testY = train_test_split(self.X, self.y, test_size=self.splitRatio)

# For this dataset, I want to do a feature reduction due to high correlation. Also, there is one dummy feature column. Lasso Regression handles these issues well.
# I added a nonlinear model option just in case. For label 1, linear regression (Lasso) yields accurate results. 
 # For the second label, a nonlinear model yields a better result. Linear model fails to predict correctly.   
#Lasso for Label 1 gives a low error but gives a high error for Label 2 as mentioned.
# Nonlinear model with a degree 2 gives fairly good results and with a degree 3 gives acceptably accurate results(<10^-6) for label 2.
#Computational time increases with nonlinearity and with the degree of nonlinearity.
    def fixTheModel(self):
        if self.modelType==1:
            self.model=self.clf
        elif self.modelType==2:
            self.model=make_pipeline(PolynomialFeatures(self.degree), self.clf)

# Training and validation. Train on (k-1/k) portion of the data and validate on 1/k portion. We have k models at hand, trained on random 1/k portions of the data.
# Pick out the least biased model, that is the one having the least validation error.           
    def trainAndValidate(self):
        
        self.model.fit(self.trainX,self.trainY)
#        validationRatio=1/self.kFold
#        
#        for validation in range(self.kFold):
#            self.trainX, self.validateX,self.trainY, self.validateY = train_test_split(self.trainX, self.trainY, test_size=validationRatio)
#            self.model.fit(self.trainX,self.trainY)
#            outcome=self.model.predict(self.validateX)
#               
#            self.validationErrors.append(mean_squared_error(outcome,self.validateY))
#            self.models.append(self.model)
#    
## Choose the model that is the least biased of all validated models.        
#        self.model=self.models[self.validationErrors.index(min(self.validationErrors))]
#
## Release the memory
#        del self.models[:]


        
#Test our model. Get the results and mean squared error.       
    def test(self):
        self.results=self.model.predict( self.testX)
        self.finalError=mean_squared_error(self.results,self.testY)

        
# Plot the results against the real values.
    def plotTheResult(self):
        
        modelName=''
        if self.modelType==1:
            modelName='Linear Lasso Model for label  '+str(self.label)
        else:
            modelName='Nonlinear Polynomial Model of '+str(self.degree)+ " degree  for Label  "+str(self.label)
        
        sampledPredictedResults=[]
        sampledRealValues=[]
        samplePeriod=30
        
        for sample in range(0,len(self.results),samplePeriod):
            sampledPredictedResults.append(self.results[sample])
            sampledRealValues.append(self.testY[sample])
        
        plt.plot(sampledPredictedResults,'r--', label="Model Results ")
        plt.plot(sampledRealValues, 'bs', label="Real Values ")
        plt.legend(loc='best')
        plt.title(modelName)
        plt.show()

#Print results and values side by side for a naked eye comparison if you wish.            
    def printResults(self):
       
       for ii in range(len(self.results)):
           print(self.testY[ii],self.results[ii])

#Report the results. Validation errors are important to see the accuracy of trained model.
# If the error is above 10^-5, the model is inaccurate.       
    def report(self):
        
        modelName=""
        ending=""
        if self.modelType==1:
            modelName="Linear"
        else:
            modelName="Non-linear"
            ending=" of "+ str(self.degree)+ " degree."
        
         
        print(modelName+" model "+ ending)    
        
#        print(str(self.kFold)+" fold validation errors: ")
#        print(self.validationErrors)
            
        print("Overall error is: ",self.finalError )


 # Wrap up a plethora of methods.       
    def trainTestWrapper(self):
            
            self.trainTestSplit()
            self.fixTheModel()
            self.trainAndValidate()
            self.test()
            
        
if __name__ == '__main__':
        
        myRegression=OilRegression()
        myRegression.PCATransform()
        myRegression.getInfo()
        myRegression.trainTestWrapper()
        myRegression.report()
        myRegression.plotTheResult()
        
        