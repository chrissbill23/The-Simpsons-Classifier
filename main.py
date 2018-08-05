from fileloaders import ImgCSVLoader
from preprocessing import ImgPreprocessor
import numpy as np
import glob
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from console import alert
from ploting import pieY, histY, plotImage, plotProbThresh
from classification import ImgClassification
import pandas as pd
#from keras.models import Sequential
#from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
#from keras.utils import np_utils

filenames = []
Y = []
X_test = []
'''
f = [img for img in glob.glob("Data/Training/maggie_simpson/*.jpg")]
filenames.append(f[:int(0.75*len(f))])
X_test.append(f[int(0.75*len(f)):])
Y.append("maggie_simpson")
'''
f = [img for img in glob.glob("Data/Training/abraham_grampa_simpson/*.jpg")]
X_test.append([img for img in glob.glob("Data/Test/abraham_grampa_simpson/*.jpg")])
filenames.append(f)
Y.append("abraham_grampa_simpson")

f = [img for img in glob.glob("Data/Training/homer_simpson/*.jpg")]
X_test.append([img for img in glob.glob("Data/Test/homer_simpson/*.jpg")])
filenames.append(f)
Y.append("homer_simpson")

f = [img for img in glob.glob('Data/Training/bart_simpson/*.jpg')]
X_test.append([img for img in glob.glob("Data/Test/bart_simpson/*.jpg")])
filenames.append(f)
Y.append('bart_simpson')

f = [img for img in glob.glob('Data/Training/lisa_simpson/*.jpg')]
X_test.append([img for img in glob.glob("Data/Test/lisa_simpson/*.jpg")])
filenames.append(f)
Y.append('lisa_simpson')

f = [img for img in glob.glob('Data/Training/marge_simpson/*.jpg')]
X_test.append([img for img in glob.glob("Data/Test/marge_simpson/*.jpg")])
filenames.append(f)
Y.append('marge_simpson')

loader = ImgCSVLoader()

alert('Loading training set')
images, labels = loader.setImagesFromFolder(filenames, Y).getData()
alert('Loading test set')
X_test, Y_test = loader.setImagesFromFolder(X_test, Y).getData()

alert('First plots')
histY(labels, ylabel='Totale immagini')
pieY(labels, ylabel='Totale immagini')
plotImage(images[3])

preprocessor = ImgPreprocessor().convert2D().resize(288, 288).scale().ajustGamma(1).hogDescriptor(orientations=10, pixels_per_cell=(8, 8), cells_per_block=(1, 1), multichannel=True)
fitter = ImgClassification().preProcess().setProcessor(preprocessor)

risp = input("Proceed with SVM Classifier learning? [y/n]")
if risp == "y":
    alert('SVM classifier')
    svm = SVC(C=5, class_weight="balanced")    
    fitter.setAlgorithm(svm).setTrainingImg(images, labels).CVLearning(n_splits=3, random_state=0, labels=Y)
    fitter.preProcess(False).fit()
    fitter.setTestingImg(X_test, Y_test).preProcess().predictX(Y)
    modello = fitter.getModel()
    print('Massimo valore in modulo dei coefficienti dei SV =',np.amax(np.abs(modello.dual_coef_)))
    Y = np.sort(Y, kind='heapsort')
    histY(X=Y, Y=modello.n_support_)
    alphas = []
    for i in range(0, len(Y)-1):
        alphas.append(np.arange(1, len(modello.dual_coef_[i])+1, 1))
    histY(X=alphas, Y=modello.dual_coef_, xlabel='Coefficiente del suport vector')
    
risp = input("Proceed with MLP Classifier learning? [y/n]")
if risp == "y":
    alert('MLPClassifier fitting')
    mlp = MLPClassifier(hidden_layer_sizes=(350))
    fitter.setAlgorithm(mlp).setTrainingImg(images, labels).CVLearning(random_state=0, labels=Y)
    fitter.preProcess(False).fit()
    fitter.setTestingImg(X_test, Y_test).preProcess().predictX(Y)
    y_prob = fitter.preProcess(False).predictProba().getPredicted()
    plotProbThresh(Y_test, y_prob, Y, thresh = np.arange(0.005, 1.0, 0.005))
    
'''
risp = input("Proceed with CNN Classifier learning? [y/n]")
if risp == "y":
    alert('CNN fitting')
    size = 64
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(1, size, size)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    images, labels = loader.set3DImagesFromFolder(filenames, Y, size, size, True).getData()
    preprocessor = ImgPreprocessor().labelEncoding().setDataSet(images, labels).preprocess()
    X_train, Y_train = preprocessor.getPreprocessed()
    X_train = X_train/255
    X_train = X_train.reshape(X_train.shape[0], 3, size, size)
    print(X_train.shape)
    Y_train = np_utils.to_categorical(Y_train, num_classes=5)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)
'''