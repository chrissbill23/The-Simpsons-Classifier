import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import recall_score, precision_score, f1_score

def __countValues(y):
    x = {}
    Y = []
    for i in range(0, len(y)):
        try:
            x[y[i]] = x[y[i]] + 1
        except KeyError:
            x[y[i]] = 1
            Y.append(y[i])
    return x, Y
def histY(data=None, X=None, Y=None, xlabel="", ylabel="", title='', showLegend=False):
    y, x = [], []
    if data is not None:
        y, x = __countValues(data)
        for i in x:
            plt.bar(i,y[i], label=i)
    else:
        y, x = Y, X    
        for i in range(0,len(x)):
            plt.bar(x[i],y[i], label=i)
    if xlabel != "":
        plt.xlabel(xlabel)
    if ylabel != "":
        plt.ylabel(ylabel)
    if showLegend == True:
        plt.legend()
    plt.title(title)
    plt.show()
def pieY(Y, xlabel="", ylabel="", title='', showLegend=False):
    x, y = __countValues(Y)
    plt.pie(x=[x[i] for i in y], labels=y, autopct='%1.1f%%', shadow=True)
    plt.show()
def plotImage(im):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(im)
    plt.axis('off')
    histo = plt.subplot(1,2,2)
    histo.set_ylabel('Totale')
    histo.set_xlabel('Pixel Intensity')
    n_bins = 30
    plt.hist(im[:,:,0].flatten(), bins= n_bins, lw = 0, color='r', alpha=0.5);
    plt.hist(im[:,:,1].flatten(), bins= n_bins, lw = 0, color='g', alpha=0.5);
    plt.hist(im[:,:,2].flatten(), bins= n_bins, lw = 0, color='b', alpha=0.5);
    plt.show()
def plot_confusion_matrix(cm, classes=None,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
def plotProbThresh(y_test, y_prob, classnames, thresh = np.arange(0, 1.1, 0.01)):
    plt.figure(1)
    classnames = np.sort(classnames, kind='heapsort')
    for c in range(0, len(classnames)):
        precision_scores = []
        recall_scores = []
        f1_scores = []
        for j in thresh:
            y_pred = []
            y_test2 = []
            for i in range(0,len(y_test)):
                if(y_test[i] != classnames[c]):
                    y_test2.append(0)
                else:
                    y_test2.append(1)
                if y_prob[i][c] >= j:
                    y_pred.append(1)
                else: 
                    y_pred.append(0)
            y_test2, y_pred = np.array(y_test2), np.array(y_pred)
            recall_scores.append(recall_score(y_test2, y_pred))
            precision_scores.append(precision_score(y_test2, y_pred))
            f1_scores.append(f1_score(y_test2, y_pred))
        conta = 0
        for i in y_test:
            if i == classnames[c]:
                conta = conta + 1
        plt.plot(thresh, recall_scores, label='Recall')
        plt.plot(thresh, precision_scores, label='Precision')
        plt.plot(thresh, f1_scores, label='F1 score')
        plt.title('Classe: '+str(classnames[c])+'\n Test size: '+str(conta))
        plt.legend(loc='best', shadow=True, fancybox=True)
        plt.xlabel("Threshold")
        plt.grid(False)
        plt.show()
'''
def plotProbThreshAll(y_test, y_prob, classnames, thresh = np.arange(0, 1.1, 0.01)):
    plt.figure(1)
    classnames = np.sort(classnames, kind='heapsort')
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for j in thresh:
        y_pred = []
        for i in range(0, len(y_prob)):
            max = j
            index = -1
            max2 = -1
            index2 = -1
            for k in range(0, len(y_prob[i])):
                if y_prob[i][k] > max:
                    max = y_prob[i][k]
                    index = k
                if y_prob[i][k] > max2:
                    max2 = y_prob[i][k]
                    index2 = k
            if index != -1:
                y_pred.append(classnames[index])
                print(j, max, max2, y_pred)
            else:
                y_pred.append(classnames[index2])
                print(j, max, max2, y_pred)
        y_pred = np.array(y_pred)
        recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
        precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))        
    plt.plot(thresh, recall_scores, label='Recall')
    plt.plot(thresh, precision_scores, label='Precision')
    plt.plot(thresh, f1_scores, label='F1 score')
    plt.title('Tutte le classi')
    plt.legend(loc='best', shadow=True, fancybox=True)
    plt.xlabel("Threshold")
    plt.ylabel("Media pesata")
    plt.grid(False)
    plt.show()
'''