# Assignment 5- Machine Learning in Python
### Explores basic machine learning using [scikit-learn/caret/etc.], including data preparation, model training, and evaluation. 
 
## Project Description

This project was used to practice:
    - How to download and install python Scipy (we actually will add it to our environment5py.yml)
    - Load a dataset and understand it's structure using statistical summaries and data visualization.
    - Create machine learning models, choose the most ideal model and build confidence that the accuracy is reliable.

This project is adapted from the Machine Learning Mastery group, the tutorial is titled "Your First Machine Learning Project in Python Step-By-Step" by Jason Brownlee

 All utilized the iris.csv dataset

**Table of Contents**

- Installation Instructions
- Usage
- Project Structure
- License
- Citations and Acknowledgements

 
## Installation Instructions

1. Create and clone your repository
- Begin by creating a Git Repository on GitHub
- Clone the repository to your local machine via the following command line "git clone https://github.com/your-username/your-repot-title.git

2. Create Files and Add the files to your repository

- Created a environment5py.yml file for conda environments only (as this was the only type in this project) with the following channels and dependencies named "stats-env"
name: env-5py
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - python=3.10
  - jupyterlab
  - ipykernel
  - matplotlib
  - numpy
  - statsmodels
  - seaborn
  - scipy
  - pandas

3) Set up your environment with the following command lines to activate your .yml
cd Assignment4-Stats-Scripts (or whichever working directory you are in)
conda env create -f environment.yml
conda activate stats-env
jupyter lab

4) Create a folder labelled notebook and within create an assignment5_ml_python.ipynb file

5) Create a detailed README.md (this file)

## Usage
How to use the code. Please see assignment5_ml_python.ipynb notebook for detailed examples, inputs/outputs and example data visualization.

### How to check the versions of libraries
 
#Python version
import sys
print('Python: {}'.format(sys.version))
#scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
#numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
#matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
#pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
#scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

These will give a read out of the versions below the run cell.
See notebook for example.


#### How to load and summarize the data

#Import read_csv function for pandas
from pandas import read_csv
#Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
#shape
print(dataset.shape)
#head
print(dataset.head(20))
#descriptions
print(dataset.describe())
#class distribution
print(dataset.groupby('class').size())

#### How to visualize the dataset

#visualize the data
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
#Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
#box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
#histograms
dataset.hist()
pyplot.show()
#scatter plot matrix
scatter_matrix(dataset)
pyplot.show()


#### Methods generating and evaluating our algorithms 

#compare algorithms
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
#Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)
#Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
#evaluate each model in turn
results = []
names = []
for name, model in models:
kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
results.append(cv_results)
names.append(name)
print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
#Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

#### Validate the predictions of the best model

#make predictions
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
#Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
#Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
#Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
#Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

## Project Structure

Assignment4-Stats-Scripts
    |_______ assignment5_ml_python.ipynb
    |_______ iris.csv
    |_______ environment5py.yml   
    |_______ README.md
    

    
## License 

This repository is intended for educational use only
 

## Acknowledgments and Citations
This project is adapted from the Machine Learning Mastery group, the tutorial is titled "Your First Machine Learning Project in Python Step-By-Step" by Jason Brownlee
