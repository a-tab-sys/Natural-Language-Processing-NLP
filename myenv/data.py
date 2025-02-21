# pip install nltk
import nltk
# stemming is done using PorterStemmer library
from nltk.stem import PorterStemmer
# used for stop words   
from nltk.corpus import stopwords
# used for lemmatization
from nltk.stem import WordNetLemmatizer
# regular expression library can be used to clear some text of any special character
import re

from sklearn.feature_extraction.text import CountVectorizer

# triple quotation marks (""" """ or ''' ''') are used in Python to define multiline strings. They allow a string to span multiple lines without requiring explicit newline characters (\n).

# NLTK's download() function checks if the resource is already available on your system. If the resource has already been downloaded and is present in the correct location, it won't download it again.

paragraph="""Narendra Damodardas Modi[a] (born 17 September 1950)[b] is an Indian politician who has served as the prime minister of India since 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the member of parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindu nationalist paramilitary volunteer organisation. He is the longest-serving prime minister outside the Indian National Congress.[4]. Modi was born and raised in Vadnagar in northeastern Gujarat, where he completed his secondary education. He was introduced to the RSS at the age of eight. 
At the age of 18, he was married to Jashodaben Modi, whom he abandoned soon after, only publicly acknowledging her four decades later when legally required to do so. Modi became a full-time worker for the RSS in Gujarat in 1971. The RSS assigned him to the BJP in 1985 and he rose through the party hierarchy, becoming general secretary in 1998.[c] In 2001, Modi was appointed chief minister of Gujarat and elected to the legislative assembly soon after. His administration is considered complicit in the 2002 Gujarat riots,[d] and has been criticised for its management of the crisis. According to official records, a little over 1,000 people were killed, three-quarters of whom were Muslim; independent sources estimated 2,000 deaths, mostly Muslim.[13] A Special Investigation Team appointed by the Supreme Court of India in 2012 found no evidence to initiate prosecution proceedings against him.[e] While his policies as chief minister were credited for encouraging economic growth, his administration was criticised for failing to significantly improve health, poverty and education indices in the state.[f]"""
# print(paragraph)

# ------------------------------------------------------------------------------
# tokenization -- convert paragraph to sentances, then focus on the words
# it returns a sentance-tokenized copy of your text, using NLTK's recommended sentance.

# required by NLTK's sent_tokenize function:
nltk.download('punkt_tab')

sentences = nltk.sent_tokenize(paragraph, language='english')
# print(type(sentences))      #its a list
# print(sentences)

# ------------------------------------------------------------------------------
# stemming -- reduces a word to find the base root word 

# creating an object of the PorterStemmer class
stemmer=PorterStemmer()

# give it a word and it will convert to base root
# print(stemmer.stem("going"))
# print(stemmer.stem("facial"))
# print(stemmer.stem("thinking"))
# print(stemmer.stem("drinking"))
# print(stemmer.stem("historical"))
# print(stemmer.stem("goes"))

# ------------------------------------------------------------------------------
# lemmatiztion -- reduces a word to find a MEANINGFUL base root word 
# required by WordNetLemmatizer:
nltk.download('wordnet')

# creating an object of the WordNetLemmatizer class
lemmatizer=WordNetLemmatizer()

# give it a word and it will convert to meaningful base root
# print(lemmatizer.lemmatize("going"))
# print(lemmatizer.lemmatize("facial"))
# print(lemmatizer.lemmatize("thinking"))
# print(lemmatizer.lemmatize("drinking"))
# print(lemmatizer.lemmatize("historical"))
# print(lemmatizer.lemmatize("goes"))

# ------------------------------------------------------------------------------
# clean up out data from special characters

# store our new corpus after we clean
corpus=[]

# print(len(sentences))

for i in range (len(sentences)):
    # sub returns a string by replacing the leftmost character
    # we want to replace all special characters
    # ^ : this means other than
    # other that small a to small z and big A to big Z, replace with a blank character on our sentances[i]. then convert everyhting to lower case.
    textreview = re.sub('[^a-zA-Z]', ' ', sentences[i])
    textreview=textreview.lower()
    corpus.append(textreview)

# print(corpus)

# ------------------------------------------------------------------------------
# to view english stopwords
# print(stopwords.words('english'))

# ------------------------------------------------------------------------------
# apply preprocessing to our data

nltk.download('stopwords')

## tokenization, stopwords, stemming/lemmatization
# here, i directly refers to an element of the corpus list, which contains the cleaned sentences. So, each i is still a sentence (the processed version), and the loop will print each sentence with a 1-second delay between each one.
for i in corpus:
    words=nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):   #filters out stopwords and lemmatizes/stems the remaining words  
            # print(stemmer.stem(word))
            print(lemmatizer.lemmatize(word))

# ------------------------------------------------------------------------------
# bag of words
# CountVectorizer is a class in the sklearn.feature_extraction.text module. It converts a collection of text documents into a matrix of token counts, commonly used in text processing tasks like feature extraction for machine learning models

# creating an object of the CountVectorizer class
cv=CountVectorizer()
# cv=CountVectorizer(binary=True) #if you want binary bag of words

X=cv.fit_transform(corpus)

# shows the vacabulary and the index (feature number), not frequency
print(cv.vocabulary_)

#bag of words for sentance 2. if binary youll only have 1's and 0's. if not binary you can have 2,3,4,5,...
print(X[1].toarray())