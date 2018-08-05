import pandas
import numpy as np
import cv2
from tqdm import tqdm
from skimage import io
from skimage.color import rgb2gray

class ImgCSVLoader:
    def setFile(self, file):
        self.__file = file
        return self
    def loadFile(self):
        self.__data = pandas.read_csv(self.__file).values
        self.__images = []
        tot = self.__data.shape[0]
        none = []
        for i in tqdm(range(0, tot)):
            image = io.imread(self.__data[i,0]).tolist()
            if image is not None:
                self.__images.append(image)
            else:
                none.append(i)
        if len(none) > 0:
            self.__data = np.delete(self.__data, none, 0)
        self.__labels = self.__data[:,len(self.__data[0])-1]
        return self
    def getData(self):
        return self.__images, self.__labels
    def getImages(self):
        return self.__images
    def setImagesFromFolder(self, filenames2Darr, labels, togray=False):
        tot = len(filenames2Darr)
        self.__labels = []
        self.__images = []
        if tot == len(labels):
            for i in tqdm(range(0, tot)):
                self.__images.extend([io.imread(img) for img in filenames2Darr[i]])
                self.__labels.extend(np.full((len(filenames2Darr[i]),), labels[i]))
            if togray == True:
                for i in tqdm(range(0, len(self.__images))):
                    self.__images[i] = rgb2gray(self.__images[i])
            self.__images = np.asarray(self.__images)
            self.__labels = np.asarray(self.__labels)
            self.__images = self.__images.tolist()
            self.__labels = self.__labels.tolist()
        return self
    def set3DImagesFromFolder(self, filenames2Darr, labels, width, height, togray=False):
        tot = len(filenames2Darr)
        self.__labels = []
        if tot == len(labels):
            l = []
            for i in tqdm(range(0, tot)):
                l.extend([io.imread(img) for img in filenames2Darr[i]])
                self.__labels.extend(np.full((len(filenames2Darr[i]),), labels[i]))
            for i in tqdm(range(0,len(l))):
                l[i] = np.int16(cv2.resize(l[i], (width, height)))       
            if togray == True:
                for i in tqdm(range(0, len(l))):
                    l[i] = rgb2gray(l[i])
            self.__images = np.array(l)
            self.__labels = np.asarray(self.__labels)
        return self
