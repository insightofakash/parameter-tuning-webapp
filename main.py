import streamlit as st
from sklearn import datasets
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt


st.title("Parameter Tuning of Classifier Models")

st.write("..made by Akash Dey")
st.write("### We will have 3 demo datasets and you will\
            be able to tune the parameters and check\
            how it effects the accuracy of the classifier\
            models.")


datasetname = st.sidebar.selectbox("Select Dataset", ("Breast Cancer", "Wine Dataset", "IRIS"))
classifiername = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))


def read_dataset(datasetname):
    if datasetname == "IRIS":
        df = datasets.load_iris()
    elif datasetname == "Breast Cancer":
        df = datasets.load_breast_cancer()
    elif datasetname == "Wine Dataset":
        df = datasets.load_wine()
    X = df.data
    y = df.target
    return X, y
    
X, y = read_dataset(datasetname)
st.write("The shape of the feature matrix of " ,datasetname, " dataset is: ", X.shape)
st.write("Total number of classes in dependent vector y is: ", len(np.unique(y)))

def add_parameter(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.00)
        params["C"] = C
    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("Max Depth", 2, 15)
        n_estimator = st.sidebar.slider("Number of Estimator", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimator"] = n_estimator
    return params

params = add_parameter(clf_name = classifiername)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C = params["C"])
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(max_depth=params["max_depth"], n_estimators=params["n_estimator"], random_state=42)
    return clf   

clf = get_classifier(clf_name=classifiername, params = params)


#model building

random_state = st.sidebar.write("---")
random_state = st.sidebar.slider("Random State")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)

clf.fit(X_train, y_train)
predict = clf.predict(X_test)
accuracy = accuracy_score(y_test, predict)

st.write("#### The accuracy score using ", classifiername, " classifier for ", datasetname , "database is: ", accuracy)


# Plotting using PCA

pca = PCA(3)
X_projected = pca.fit_transform(X)

fig = px.scatter_3d(X, x=X[:,0], y=X[:,1], z=X[:,2], color=X[:,0])
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20)
)
st.plotly_chart(fig)