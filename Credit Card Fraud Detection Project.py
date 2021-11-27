#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection Project

# ### Imports

# In[3]:


import numpy as np 
import pandas as pd 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model, datasets, metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import BaseLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns


# ## **Logistic Regression and Random Forest Models**

# In[37]:


# Reading dataset
input_data = pd.read_csv("creditcard.csv")

input_data = input_data.drop(['Time', 'Amount'], axis=1) #Testing shows that with the `Time` column the accuracy of the mosdels decreases


# In[38]:


# Setting labels and features
y = input_data['Class']
X = input_data.drop(['Class'], axis=1)


# In[39]:


# Shuffle and split the data into training and testing subsets
X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size = 0.4, random_state = 0)


# In[43]:


# Classifying with Logistic Regression 
lm_classifier = LogisticRegression()
lm_classifier.fit(X,y)
y_pred = lm_classifier.predict(X_training)
lm_classifier.score(X_testing, y_testing) # Accuracy
precision = precision_score(y_training, y_pred, average='binary')
recall = recall_score(y_training, y_pred, average='binary')

cm = confusion_matrix(y_training, y_pred)

df_cm = pd.DataFrame(cm2, index = ['TP', 'TN'])
df_cm.columns = ['Predicted Pos', 'Predicted Neg']

print("\n Precision: ", precision, "\n Recall: ", recall)
sns.heatmap(df_cm, annot=True, fmt="d")


# In[45]:


# Classifying with Random Forest
rf_classifier = RandomForestClassifier()
rf_classifier = rf_classifier.fit(X_training, y_training)

y_pred = rf_classifier.predict(X_training)
rf_classifier.score(X_testing, y_testing) # Accuracy
precision = precision_score(y_training, y_pred, average='binary')
recall = recall_score(y_training, y_pred, average='binary')

cm3 = confusion_matrix(y_training, y_pred)

df_cm3 = pd.DataFrame(cm3, index = ['TP', 'TN'])
df_cm3.columns = ['Predicted Pos', 'Predicted Neg']

print("\n Precision: ", precision, "\n Recall: ", recall)
sns.heatmap(df_cm3, annot=True, fmt="d")


# ## **Neural Network approach**

# In[46]:


# Re-Importing and normalizing data with StandardScaler
input_data = pd.read_csv('creditcard.csv')
input_data.iloc[:, 1:29] = StandardScaler().fit_transform(input_data.iloc[:, 1:29])
data_matrix = input_data.as_matrix()
X = data_matrix[:, 1:29]
Y = data_matrix[:, 30]
class_weights = dict(zip([0, 1], compute_class_weight('balanced', [0, 1], Y)))


# In[47]:


# k-fold cross-validation
seed = 3
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
cvscores = []
predictions = np.empty(len(Y))
predictions[:] = np.NAN
proba = np.empty([len(Y), kfold.n_splits])
proba[:] = np.NAN
k = 0 


# In[48]:


for train, test in kfold.split(X, Y):
    model = Sequential()
    model.add(Dense(28, input_dim=28))
    model.add(Dropout(0.2))
    model.add(Dense(22))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    bl = BaseLogger()
    cp = ModelCheckpoint(filepath="checkpoint.hdf5", verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=8, verbose=0, mode='auto')
    rlop = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=0.001)
    tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    model.fit(X[train], Y[train], batch_size=1000, nb_epoch=100, verbose=0, shuffle=True, validation_data=(X[test], Y[test]), class_weight=class_weights, callbacks=[bl, cp, es, rlop, tb])

    # Current iteration score
    scores = model.evaluate(X[test], Y[test], verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    # Storing predicted probs
    proba[train, k] = model.predict_proba(X[train]).flatten()
    k += 1


# In[54]:


pred = np.nanmean(proba, 1) > 0.5
pred = pred.astype(int)
print(classification_report(Y, pred))
# Print 
pd.crosstab(Y, pred, rownames=['Actual values'], colnames=['Predictions'])

