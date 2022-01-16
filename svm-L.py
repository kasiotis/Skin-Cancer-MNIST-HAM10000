import pandas as pd
import numpy as np
import keras
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Parameters for MNIST dataset
img_rows, img_cols = 28, 28
num_classes = 7

# import the data of images and the data from HAM_metadata.csv
dataset_images = pd.read_csv("cancer-data/hmnist_28_28_L.csv")
dataset_images = dataset_images.sample(n=2000, random_state=1)
# removing the 'label' column from the data frame so I only keep the image data
images = dataset_images.drop(['label'], axis=1)
print(images.shape)
# keeping only the label column
labels = dataset_images['label']

# Parsing the images into an array and resizing them from 4d arrays to 3d arrays
images = np.array(images)
images = images.reshape(images.shape[0], -1)
# Parsing the labels into an array
labels = np.array(labels)

# Normalizing the images.
images = (images - np.mean(images)) / np.std(images)

# Splitting my predictive and response data into training and testing sets with an 80:20 ratio
# while the state is set to a constant so that the splitting can be done reproducibly
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, random_state=1, test_size=0.20)

# Encoding my labels
lab = LabelEncoder()
y_train = lab.fit_transform(y_train)
y_test = lab.fit_transform(y_test)

# Finding the best parameters by cross-validation
parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 0.01, 0.1, 0.5],
               'C': [1, 10, 100, 1000]}]
print("# Tuning hyper-parameters")
clf = GridSearchCV(SVC(), parameters, cv=num_classes)
clf.fit(x_train, y_train)
print('best parameters:')
print(clf.best_params_)
print('-------------------------------------')
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

# setting the optimal parameters that were found
optimal_C = 100
optimal_gamma = 0.0001

# Fitting the model
svc = SVC(kernel="rbf", gamma=optimal_gamma, C=optimal_C)
svc.fit(x_train, y_train)

pred = svc.predict(x_test)

# printing the accuracy of the SVM model
print("The accuracy score is: ", accuracy_score(y_test, pred))
