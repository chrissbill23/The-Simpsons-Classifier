import numpy as np
import cv2
from tqdm import tqdm
from console import alert, stepM
from sklearn.preprocessing import OneHotEncoder, scale, LabelEncoder
from skimage.feature import hog

class ImgPreprocessor:
    def __init__(self):
        self.__gamma = None
        self.__labelOHE = False
        self.__labelE = False
        self.__hog = False
        self.__scale = False
        self.__rgbScale = False
        self.__2D = False
        self.__label_encoder = None
        self.__onehot_encoder = None
        self.__width = None
        self.__height = None
    def setDataSet(self, X, Y):
        self.__X = X
        self.__Y = Y
        return self
    def getPreprocessed(self):
        return self.__X, self.__Y
    def convert2D(self):
        self.__2D = True
        return self
    def rgbMinMaxScaling(self, apply=True):
        self.__rgbScale = apply
        return self
    def scale(self, apply=True):
        self.__scale = apply
        return self
    def resize(self, width, height):
        self.__width = width
        self.__height = height
        return self
    def ajustGamma(self, gamma):
        self.__gamma = gamma
        return self
    def labelOneHotEncoding(self, apply=True):
        self.__labelOHE = apply
        return self
    def hogDescriptor(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', transform_sqrt=False, feature_vector=True, multichannel=None):
        self.__hog = True
        self.__hog_orientations = orientations
        self.__hog_pixels_per_cell = pixels_per_cell
        self.__hog_cells_per_block = cells_per_block
        self.__hog_block_norm = block_norm
        self.__hog_transform_sqrt = transform_sqrt
        self.__hog_feature_vector = feature_vector
        self.__hog_multichannel = multichannel
        return self
    def labelEncoding(self, apply=True):
        self.__labelE = apply
        return self
    def InverselabelEncoding(self, Y):
        if self.__label_encoder is not None:
            return self.__label_encoder.inverse_transform(Y)
    def InverseOHEncoding(self, Y, labels):
        ohe = self.oneHotEncode(labels)
        temp = []
        for i in range(0, len(Y)):
            val = None
            for j in range(0, len(ohe)):
                z = 0
                while z < len(ohe[j]) and ohe[j][z] == Y[i][z]:
                    z = z+1
                if z == len(ohe[j]) and z == len(Y[i]):
                    temp.append(labels[j])
        return np.array(temp)
                
    def preprocess(self):
        if len(self.__X) > 0:  
            if self.__width is not None and self.__height is not None:
                stepM('Resizing images')
                for i in tqdm(range(0,len(self.__X))):
                    self.__X[i] = cv2.resize(self.__X[i], (288, 288))
            if self.__gamma is not None:
                invGamma = 1.0 / self.__gamma
                table = np.array([((i / 255.0) ** invGamma) * 255
                    for i in np.arange(0, 256)]).astype('uint8')    
                stepM('Applying gamma')
                for i in tqdm(range(0,len(self.__X))):
                    self.__X[i] = cv2.LUT(self.__X[i], table)
            if self.__hog == True:
                stepM('Applying hog feature descriptor')
                for i in tqdm(range(0,len(self.__X))):
                    self.__X[i] = hog(self.__X[i],orientations=self.__hog_orientations, 
                                      pixels_per_cell=self.__hog_pixels_per_cell, 
                                      cells_per_block=self.__hog_cells_per_block, block_norm=self.__hog_block_norm, 
                                      transform_sqrt=self.__hog_transform_sqrt, feature_vector=self.__hog_feature_vector, 
                                      multichannel=self.__hog_multichannel)
            if self.__2D == True:
                stepM('Converting independent variables to vectors')
                if self.__hog == False:
                    for i in tqdm(range(0,len(self.__X))):
                        self.__X[i] = self.__X[i].transpose(2,0,1).reshape(3,-1).flatten()
                else:
                    for i in tqdm(range(0,len(self.__X))):
                        self.__X[i] = self.__X[i].ravel()
                    
            if self.__rgbScale == True:
                stepM('rgb min max scaling')
                for i in tqdm(range(0,len(self.__X))):
                    self.__X[i] = np.float16(self.__X[i])/255
            
            if self.__scale == True:
                stepM('center scaling')
                for i in tqdm(range(0,len(self.__X))):
                    self.__X[i] = scale(self.__X[i])
            
            self.__X = np.array(self.__X, np.float16)
        if len(self.__Y) > 0 :
            if self.__labelOHE == True:
                stepM('Label One hot encoding')
                self.__Y = self.oneHotEncode(self.__Y)
                print('\nDone encoding label.\n')
            else:
                if self.__labelE == True:
                    stepM('Label integer encoding')
                    self.__Y = self.__labelEncode(self.__Y)
                    print('\nDone encoding label.\n')
                self.__Y = np.asarray(self.__Y)
        return self
    def oneHotEncode(self, Y):
        # binary encode
        if self.__onehot_encoder is None:
            self.__onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = self.__labelEncode(Y)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        return np.array(self.__onehot_encoder.fit_transform(integer_encoded), dtype=np.int8)
        
    def __labelEncode(self, Y):
        # integer encode
        if self.__label_encoder is None:
            self.__label_encoder = LabelEncoder()
        integer_encoded = self.__label_encoder.fit_transform(Y)
        return integer_encoded