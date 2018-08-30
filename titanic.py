# Artificial Neural Network

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 2:12]
X.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
X.iloc[:,6].fillna("C",inplace=True)
X = X.values
z = X[:,0]
z[z == 3 ] = 0
z[z == 1 ] = 3
z[z == 0 ] = 1
X[:, 0] = z
y = dataset.iloc[:, 1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 2:6])
X[:, 2:6] = imputer.transform(X[:, 2:6])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 6] = labelencoder_X_2.fit_transform(X[:, 6])
onehotencoder_1 = OneHotEncoder(categorical_features = [6])
X = onehotencoder_1.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer w/dropout
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
classifier.add(Dropout(rate=0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 25, epochs = 250)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer, hidden_nodes):
    classifier = Sequential()
    classifier.add(Dense(units = hidden_nodes, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units = hidden_nodes, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))        
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25,32], 'epochs':[250,500], 'optimizer':['adam','rmsprop'], 'hidden_nodes':[4,10]}
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

#optimal parameters: batch size 32, epochs 250, hidden nodes: 10, optomizer adam 82.6%
final_set = pd.read_csv('test.csv')
final_set = final_set.iloc[:,0]
test_set = pd.read_csv('test.csv')
test_set.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_set.iloc[:,6].fillna("C",inplace=True)
test_set = test_set.values
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(test_set[:, 2:6])
test_set[:, 2:6] = imputer.transform(test_set[:, 2:6])
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
test_set[:, 0] = labelencoder_X_1.fit_transform(test_set[:, 0])
labelencoder_X_2 = LabelEncoder()
test_set[:, 1] = labelencoder_X_2.fit_transform(test_set[:, 1])
labelencoder_X_3 = LabelEncoder()
test_set[:, 6] = labelencoder_X_3.fit_transform(test_set[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [0])
test_set = onehotencoder.fit_transform(test_set).toarray()
test_set = test_set[:, 1:]
onehotencoder_1 = OneHotEncoder(categorical_features = [7])
test_set = onehotencoder_1.fit_transform(test_set).toarray()
test_set = test_set[:, 1:]
test_set = sc.fit_transform(test_set)

#Add results
num = 0
final_set= pd.concat([final_set, pd.DataFrame(data=[], columns=['Survived'])], axis=1)
for x in test_set:
    new_prediction = classifier.predict(sc.transform(np.array([x])))
    val = 1 if new_prediction >0.5 else 0
    final_set['Survived'][num] =  val
    num += 1

# Save Final Set
final_set.to_csv('result.csv', index=False)