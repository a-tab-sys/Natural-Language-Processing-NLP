# we will use gensim library
# skipgram and cbow do basically the same thing - but if the dataseat is very big, you can select skipgram to use

import gensim
from gensim.models import Word2Vec, KeyedVectors
# references: https://stackoverflow.com/questions/46433778/import-googlenews-vectors-negative300-bin

import gensim.downloader as api

# we will use a pretrained model. google has created a pretrained model called word2vec-google-news. google has trained with its google news text data which had over 3 billion words and they have creeted a word2vec that gives you an output with 300 dimensions
# wv is what we named our model. variable for our model
wv = api.load('word2vec-google-news-300')

# when i print the vector of king - youll see a 300 diemension word
vec_king = wv['king']
print(vec_king)

# similar, i print the vector of man - youll see a 300 diemension word
vec_man = wv['man']
print(vec_man)


print(vec_king.shape)

# most similar function. it will take the word (cricket), convert that to vectors and show ou words that are similar based on the =ir cosime simmilarity
print(wv.most_similar('cricket'))
print(wv.most_similar('happy'))     
print(wv.most_similar('king'))

# similarity function tells you the cosine similarity between 2 words. 
print(wv.similarity("hockey","sports"))
print(wv.similarity("html","programmer"))
print(wv.similarity("cat","banana"))

# you can add and subtract vectors
vec=wv['king']-wv['man']+wv['woman']
print(vec)

# variables need to be given in form of list
print(wv.most_similar([vec]))




