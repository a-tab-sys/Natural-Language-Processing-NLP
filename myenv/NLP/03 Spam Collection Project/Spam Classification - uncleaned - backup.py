

import pandas as pd
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# 0. Import Dataset
# sep is asking what is the seperating feature? its a tab. we will see in our dataset that we have a huge list of:
# output (spam or ham), then a tab, then the input (message)
# output (spam or ham), then a tab, then the input (message)
# .....
# we are setting column names for the dataset. the output column is label, input is message

# messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=["label", "message"])
messages = pd.read_csv(r'c:\Users\Hasaan\Documents\NLP\myenv\03 Preprocessing (Word2Vec, AvgWord2Vec)\Temp name\smsspamcollection\SMSSpamCollection', sep='\t', names=["label", "message"])
print(messages)

# tells you number of rows, columns
print(messages.shape)   

# to pick up a specific index
print(messages['message'].loc[100])
print(messages['label'].loc[100])

# Data cleaning and preprocessing
# 1. Tokenization
# 2. Stopwords
# 3. Stemming - stemming creates alot of incorrect words, but because we are creating a spam classification- its ok. For a chatbox or something then lemmatization would be much better.
# 4. Lemmatization (meaningful term)
# 5. Text => Vectors(Bow, TFIDF, Word2Vec, AvgWord2Vec)

corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-z0-9]', ' ', messages['message'][i])   #remove any special characters other than a-z or A-Z or 0-9
    review = review.lower()
    review = review.split()     #
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]   #apply stopwords and stemming
    review = ' '.join(review)       #join then
    corpus.append(review)           #add to corpus
# print(corpus)





# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500, binary=True, ngram_range=(2,2))
X = cv.fit_transform(corpus).toarray()
# print(X) #bow array wrt our features
print(X[0]) #bow array for features in sentance 1

print(X.shape)      # (5572, 2500) 2500 features bc you set max features

# doing something with y values? label encoding...look into this
y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
print(X_train, y_train)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

#prediction
y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report
score=accuracy_score(y_test,y_pred)
print(score)

from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))





# Creating the TFIDF model
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features=2500, ngram_range=(1,2))
X = tv.fit_transform(corpus).toarray()

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

#prediction
y_pred=spam_detect_model.predict(X_test)

score=accuracy_score(y_test,y_pred)
print(score)

from sklearn.metrics import classification_report

print(classification_report(y_pred,y_test))






#Another way - ............................................
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)
accuracy_score(y_pred, y_test)
print(classification_report(y_pred, y_test))





# Word2vec Implementation
# 2 types: skipgram and cbow
# gensim library lets us train word2vec model from scratch, the google news used before was a pretrained model

# should we train from scratch or just use a pretrained model? 
# generally if the pretrained model captures >= 75% of the vocablary, then go ahead with pretrained
# but if you see that your dataset has text that is not present in the pretrained model - you can just train it from scratch

#word2vec basicallyconverts words to a vector of a FIXED size
#ex the google news model converted words to a vector of 300 dimensions
# suppose i had a word: king -> [....300 dimensions]
# our input here is some message, and our output spam or ham
# I/P                         O/P
# I want to eat pizza         Spam/ham

#word to vec will for each word in the message, convert it to 300 dimensions but we need out whold input (the 5 words) to be 300 dimensions.
#to fix this problem: we can use avg word 2 vec



# AvgWord2Vec
# if i had a sentance: "Please subscribe my channel"
# word to vec will convert each word to a vector of 300 dimensions

#again i want the entire sentance to be represented by 300 dimensions
#avg word2vec will do this by taking the average of all the seperate vectors of the 4 words in the sentance. the avg (or sum(double check)) of the first, second, third...300th diemension of all the words will be calculated, and this will be the vector representing the entire sentance. 




# import os

# # Get the path to the current directory (where 2.py is located). __file__ is a special variable in Python that refers to the current script's file path
# current_dir = os.path.dirname(__file__)
# print(current_dir)

# # Create the relative path to your specific file:SMSSpamCollection
# file_path = os.path.join(current_dir, 'smsspamcollection', 'SMSSpamCollection')
# print(file_path)

#CONTINUE AT 43 MINUTES
