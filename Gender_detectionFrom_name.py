import pandas as pd
import numpy as np
#ML package
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


#load  data
df=pd.read_csv('dataset.csv')
df.size
#Data Cleaning
#checking for column name consistency
df.columns
#Data Cleaning
#checking for column name consistency
df.columns
df.isnull().isnull().sum()

#Number of Female Names
df[df.sex=='F'].size


#Number of Male Names
df[df.sex=='M'].size





df_names=df


#Replace F with o && M(male) with  1
df_names.sex.replace({'F':0,'M':1},inplace=True)
df_names.sex.unique()
df_names.dtypes
Xfeatures=df_names['name']

#Feature Extraction
countervect=CountVectorizer()
X=countervect.fit_transform(Xfeatures)



#Features
X
#labels
y=df_names.sex

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.33, random_state=42)

# Naive Bayes Classifier

classifier = MultinomialNB()
classifier.fit(X_train,y_train)
classifier.score(X_test,y_test)

AccuracyNBMd=classifier.score(X_test,y_test)
# Accuracy of our Model on train
print("Accuracy of Model of naive_bayes on train",classifier.score(X_train,y_train))
# Accuracy of our Model
print("Accuracy of Model of naive_bayes on test",classifier.score(X_test,y_test))


# A function to do it
def genderpredictionNB(a):
    test_name = [a]
    vector = countervect.transform(test_name).toarray()
    if classifier.predict(vector) == 0:
        print("Female")
    else:
        print("Male")
genderpredictionNB("Tanisha")# sample test



# By Analogy most female names ends in 'A' or 'E' or has the sound of 'A'
def features(name):
    name = name.lower()
    return {
        'first-letter': name[0], # First letter
        'first2-letters': name[0:2], # First 2 letters
        'first3-letters': name[0:3], # First 3 letters
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }

# Vectorize the features function
features = np.vectorize(features)


# Extract the features for the dataset
df_X = features(df_names['name'])
df_y = df_names['sex']



# Train Test Split
dfX_train, dfX_test, dfy_train, dfy_test = train_test_split(df_X, df_y, test_size=0.33, random_state=42)



# Train Test Split
dfX_train, dfX_test, dfy_train, dfy_test = train_test_split(df_X, df_y, test_size=0.33, random_state=42)

#print(dfX_train)


#Dictvectorizer
dv = DictVectorizer()
dv.fit_transform(dfX_train)

# Model building Using DecisionTree


 
dclf = DecisionTreeClassifier()
my_xfeatures =dv.transform(dfX_train)
dclf.fit(my_xfeatures, dfy_train)

# Build Features and Transform them

# Predicting Gender of Name
# Male is 1,female = 0


# Build Features and Transform them
def genderpredictionDecTree(a):
    sample_name_eg = [a]
    transform_dv =dv.transform(features(sample_name_eg))
    vect3 = transform_dv.toarray()
    if dclf.predict(vect3) == 0:
         print("Female")
    else:
         print("Male")

## Accuracy of Models Decision Tree Classifier Works better than Naive Bayes
# Accuracy on training set
print("Accuracy on training set of Decision Tree Classifier: ",dclf.score(dv.transform(dfX_train), dfy_train))

# Accuracy on test set
print("Accuracy on test set of Decision Tree Classifier: ",dclf.score(dv.transform(dfX_test), dfy_test))

AccuracyDecTreeMd=dclf.score(dv.transform(dfX_test), dfy_test)



if AccuracyNBMd > AccuracyDecTreeMd :

    print("Detect Gender by NB: ")
    genderpredictionNB("Rahat")
else:
	print("Detect Gender by DecisionTree: ")
	genderpredictionDecTree("Rahat")
    