# Natural Language Processing

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #quoting is to ignore the double
#quotes in the reviews

#cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) #here ^ is used for negation, sub = 'substitute'
    review = review.lower()
    review = review.split()
    ps = PorterStemmer() #stemming i.e. keeping the roots of the word  
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #removing some irrelevant from reviews  
    review = ' '.join(review)
    corpus.append(review)
    
# creating the bag of words model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)  
X = cv.fit_transform(corpus).toarray()   
y = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy')
classifier.fit(X_train,y_train)

# Predicting the Test set resultss
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

    


 

                                                                                  