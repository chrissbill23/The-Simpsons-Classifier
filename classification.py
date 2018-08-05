from sklearn.neural_network import MLPClassifier
from console import alert, stepM, ShowTimer
from sklearn import model_selection
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import numpy as np
import pandas as pd
from ploting import plot_confusion_matrix

class ImgClassification:
    def __init__(self):
        self.__X_train = []
        self.__Y_train = []
        self.__X_test = []
        self.__Y_test = []
        self.__y_pred = []
        self.__algorithm = None
        self.__preProcess = False
        
    def setTrainingImg(self, data, labels):
        self.__X_train = data
        self.__Y_train = labels
        return self
    def getTrainingImg(self):
        return self.__X_train, self.__Y_train
    def getPredicted(self):
        return self.__y_pred
    def setTestingImg(self, data, labels):
        self.__X_test = data
        self.__Y_test = labels
        return self
    def getTestingImg(self):
        return self.__X_test, self.__Y_test
    def getPreprocessor(self):
        return self.__preProcessor
    def setProcessor(self, preprocessor):
        self.__preProcessor = preprocessor
        return self
    def preProcess(self, before=True):
        self.__preProcess = before
        return self
    def setAlgorithm(self, algo):
        self.__algorithm = algo
        return self
    def getModel(self):
        return self.__algorithm
    def fit(self):
        if self.__preProcess == True:
            alert("Preprocessing before fitting")
            self.__preProcessor.setDataSet(self.__X_train, self.__Y_train)
            self.__preProcessor.preprocess()
            self.__X_train, self.__Y_train = self.__preProcessor.getPreprocessed()
        stepM("Fitting Data")       
        elapsed=ShowTimer()
        elapsed.start()
        self.__algorithm.fit(self.__X_train, self.__Y_train)
        elapsed.stop()
        print("Done fitting.")   
        return self   
    def predictX(self, labels=None):
        labels = np.sort(labels, kind='heapsort') #sort labels for confusion matrix   
        if self.__preProcess == True:
            alert("Preprocessing before testing")
            self.__preProcessor.setDataSet(self.__X_test, self.__Y_test)
            self.__preProcessor.preprocess()
            self.__X_test, self.__Y_test = self.__preProcessor.getPreprocessed()
        stepM("Testing...")
        elapsed=ShowTimer()
        elapsed.start()
        self.__y_pred = self.__algorithm.predict(self.__X_test)
        elapsed.stop()
        print("Done testing.")
        print(classification_report(self.__Y_test, self.__y_pred))
        cnf_matrix = confusion_matrix(self.__Y_test, self.__y_pred)
        print(pd.crosstab(self.__Y_test, self.__y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
        plot_confusion_matrix(cnf_matrix, classes=labels, title='Confusion matrix')
        return self
    def predictProba(self):
        if self.__preProcess == True:
            alert("Preprocessing before testing")
            self.__preProcessor.setDataSet(self.__X_test, self.__Y_test)
            self.__preProcessor.preprocess()
            self.__X_test, self.__Y_test = self.__preProcessor.getPreprocessed()
        stepM("Testing...")
        elapsed=ShowTimer()
        elapsed.start()
        self.__y_pred = self.__algorithm.predict_proba(self.__X_test)
        elapsed.stop()
        print("Done testing.")
        return self        
    def CVLearning(self, n_splits=3, test_size=0.25, random_state=None, labels=None):
        if len(self.__X_train) > 0 and len(self.__Y_train) > 0:  
            labels = np.sort(labels, kind='heapsort') #sort labels for confusion matrix    
            if self.__preProcess == True:
                alert("Preprocessing before fitting")
                self.__preProcessor.setDataSet(self.__X_train, self.__Y_train)
                self.__preProcessor.preprocess()
                self.__X_train, self.__Y_train = self.__preProcessor.getPreprocessed()     
            elapsed=ShowTimer()
            elapsed.start() 
            cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
            split_index = 1
            for train_index, test_index in cv.split(self.__X_train, self.__Y_train):
                X = [self.__X_train[j] for j in train_index]
                Y_t = [self.__Y_train[i] for i in train_index]
                X_test = [self.__X_train[j] for j in test_index]
                y_test = np.array([self.__Y_train[i] for i in test_index])
                self.__X_test.extend(X_test)
                self.__Y_test.extend(y_test)
                
                stepM("Fitting split #"+str(split_index)) 
                self.__algorithm.fit(X, Y_t)
                stepM("Testing split #"+str(split_index))  
                y_pred = self.__algorithm.predict(X_test)
                print(classification_report(y_test, y_pred))
                print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
                cnf_matrix = confusion_matrix(y_test, y_pred)
                plot_confusion_matrix(cnf_matrix, classes=labels, title='Confusion matrix')
                self.__y_pred.extend(y_pred)
                split_index = split_index +1
            elapsed.stop()
            print("Done fitting.")   
            print('Done testing')
        else:
            alert('Training set missing')
    