import joblib
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import random
from PIL import Image, ImageTk

dir = 'D:/MMU/TPR/project/gui' # change before run

train_data = joblib.load(os.path.join(dir, 'traindata.h5'))
label = np.concatenate((np.ones(35), np.full(48,2), np.full(32,3), np.full(53,4)),axis=0)
label.reshape(168,1)
X_train, X_test, y_train, y_test = train_test_split(train_data, label, test_size=0.4, random_state=123)
display_image = X_test[:]
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
pca = PCA(50, svd_solver='randomized', whiten=True)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


''' SVM PART '''

clf = svm.SVC(gamma=0.001, C=0.1, decision_function_shape='ovr', kernel='linear', verbose=False, random_state=123)
clf.fit(X_train_pca, y_train)
prediction = clf.predict(X_test_pca)


''' GENERATE IMAGE FROM TESTING SET '''

def gen_image():
    rint = random.randrange(0,68)
    img = display_image[rint].reshape(224,224)
    # display = Image.fromarray((img * 255).astype(np.uint8))
    display = Image.fromarray(img)
    display = display.resize((224,224), Image.ANTIALIAS)
    fdis = ImageTk.PhotoImage(display)
    
    return fdis, rint # ImageTk object, random int

''' PREDICTION '''

def pred(img_num):
    npred = clf.predict([X_test_pca[img_num]])
    return npred[0] # 1,2,3,4


''' REPORT '''

''' CLASSIFICATION REPORT '''
def class_report():
    cr = classification_report(y_test,prediction, output_dict=True)
    df = pd.DataFrame(cr).transpose()
    return df

''' CONFUSION MATRIX '''
def conf_mat():
    cmat = confusion_matrix(y_test, prediction)
    score = accuracy_score(y_test, prediction)
    ax = sns.heatmap(cmat, annot=True, fmt='.2f', linewidths=.5, square=True, cmap='Reds_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Accuracy Score: {}'.format(round(score,3)), size=10)
    plt.show()