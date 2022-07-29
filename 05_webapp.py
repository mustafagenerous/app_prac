# import libraries for webapp new
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# heading
st.write("""
# Explore differnet ML models
Lets check which are best for your data""")

# side bar and dataset optons
dataset_name= st.sidebar.selectbox("Select Dataset",
("Iris","Breast Cancer","Wine")) #"Wine Quality"

# Classifier options kon sa classifier select krna ha
classifier_name= st.sidebar.selectbox("Select Classifier",
("KNN","SVM","Random Forest"))

# import data function define kren gay dataset ko dakhne k lia
def get_dataset(dataset_name):
    data=None
    if dataset_name=="Iris":
        data=datasets.load_iris()
    elif dataset_name=="Wine":
        data=datasets.load_wine()
    else:
        data=datasets.load_breast_cancer()
    X=data.data
    y=data.target
    return X,y

# ab gunction ko call kren
X,y=get_dataset(dataset_name)

# ab dataset ki shape print krwaen gay
st.write("Shape of dataset: ",X.shape)
st.write("Number of classes: ",len(np.unique(y)))

# classifier kay parameters user in put say
def add_parameter_ui(classifier_name):
    params= dict()
    if classifier_name=="SVM":
        C= st.sidebar.slider("C",0.01,10.0)
        params["C"]=C # it is degree of classificatio
    elif classifier_name=="KNN":
            K= st.sidebar.slider("K",1,15)
            params["K"]=K # it is number of nearest neighbours
    else:
        max_depth= st.sidebar.slider("Max Depth",2,15)
        params["max_depth"]=max_depth # it is depth of tree of every tree that grow in random forest
        n_estimators= st.sidebar.slider("n_estimators",1,100)
        params["n_estimators"]=n_estimators # it is number of trees in random forest
    return params

# ab is function ko call kren
params=add_parameter_ui(classifier_name)

# classifier ko call kren
def get_classifier(classifier_name,params):
    if classifier_name=="SVM":
        clf=SVC(C=params["C"])
    elif classifier_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["K"])
    else:
        clf=RandomForestClassifier(max_depth=params["max_depth"],
        n_estimators=params["n_estimators"],random_state=1234)
    return clf

# checkbox to show code
if st.checkbox("Show Code"):
    with st.echo():
        clf=get_classifier(classifier_name,params)

        # ab data ko test train ma split kren gay
        X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=1234)

        #ab hum apnay classifier ko train kren gay
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)

        # accuracy score check r sth ab isy print krwaen gay
        acc= accuracy_score(y_test,y_pred)
        

#ab is function ko call kren
clf=get_classifier(classifier_name,params)

# ab data ko test train ma split kren gay
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=1234)

#ab hum apnay classifier ko train kren gay
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# accuracy score check r sth ab isy print krwaen gay
acc= accuracy_score(y_test,y_pred)
st.write(f'Classifier={classifier_name}')
st.write(f'Accurcy= ', acc)

# plot add kren gay sth scatter plot, PCA component annalyzery say add kren gay
# PCA ziada feature ko 2 dimensio ma plot kwa deta ha
pca= PCA(2)
X_projected= pca.fit_transform(X)

# ab data ko  0 r 1 dimension say slice kren gay
x1= X_projected[:,0]
x2=X_projected[:,1]

fig= plt.figure()
plt.scatter(x1,x2,c=y, alpha=0.7,cmap='viridis') # alpha is for tranparant point, camp is color map jo graph ha wo coloful hu ga

plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)

