from audioop import mul
from tkinter import Y
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# import cv2 -----> This is the library with which we are going to use our computer's camera.
# import numpy as np - This is so that we can perform complex mathematical/list operations
# import pandas as pd - This is so that we can treat our data as DataFrames. We already know how helpful they are.
# import seaborn as sns - This is a python module to prettify the charts that we draw with matplotlib. We have used it a couple of times.
# import matplotlib.pyplot as plt - This library is used to draw the charts.
# from sklearn.datasets import fetch_openml - This function allows us to retrieve a data set by name from OpenML, a public repository for machine learning data and experiments
# from sklearn.model_selection import train_test_split - This is to split our data into training and testing.
# from sklearn.linear_model import LogisticRegression - This is for creating a LogiticRegression Classifier
# from sklearn.metrics import accuracy_score - This is to measure the accuracy score of the model.


X = np.load('image.npz')['arr_0']
Y = pd.read_csv('labels.csv')["labels"]

print(pd.Series(Y).value_counts())

classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',]
nclasses = len(classes)

samplePerClass = 5

fig = plt.figure(figsize = (nclasses*2 , (1+samplePerClass*2)) )

idx_cls = 0

for cls in classes:
    idxs = np.flatnonzero(Y==cls)
    idxs = np.random.choice(idxs , samplePerClass , replace= False)
    i = 0

    for idx in idxs :
        plt_idx = i * nclasses + idx_cls + 1

        p = plt.subplot(samplePerClass , nclasses , plt_idx)

        p = sns.heatmap(np.array(X.loc[idx]).reshape(22,30) , cmap=plt.cm.gray ,  xticklabels=False, yticklabels=False, cbar=False)
        
        p = plt.axis('off')
        i = i + 1

    idx_cls += 1



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=2500,random_state=9 , train_size = 7500)

X_train_S = X_train/255
X_test_S = X_test/255


lr = LogisticRegression( solver = 'saga' , multi_class='multinomial').fit(X_train_S , Y_train)

Y_pred = lr.predict(X_test_S)

print(accuracy_score(Y_test,Y_pred))


cm = pd.crosstab(Y_test,Y_pred , rownames=['Actual'] , colnames = ['Predicted'])

plt.figure(figsize=(10,10))

p = sns.heatmap(cm, annot=True, fmt="d", cbar=False)

