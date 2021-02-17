import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import string
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import sklearn.preprocessing


df = pd.read_csv('Tweets.csv')


english_stops = stopwords.words('english')
english_stops += list(string.punctuation)
english_stops += ['``', "''", "'s"]


docs = []
new_docs = []

#Tweets
tweets = df['text']


#Airline sentiment which we use as labels
labels = df['airline_sentiment']

Y = np.array(labels)




for tweet in tweets:
	docs.append(tweet)



stemmer = SnowballStemmer("english", ignore_stopwords=True)
for doc in docs:
    #removing urls from tweets
	doc = re.sub(r"http\S+", "", doc)
	words = word_tokenize(doc)
	element = []
	for word in words:
		if word not in english_stops:
			stemmed_word = stemmer.stem(word)
			lowcased_word = stemmed_word.lower()
            #appending after applying stemming, removing stop words, and converting to lowcase.
			element.append(lowcased_word)
		
	element = ' '.join(element)
	new_docs.append(element)
#print(new_docs)





#new_docs contains all tweets after pre-processing
vv = TfidfVectorizer()
vv.fit(new_docs)

transformed = vv.transform(new_docs)







X_train,X_test,Y_train,Y_test = train_test_split(transformed,Y,test_size=0.2)

clf = MultinomialNB()
clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)

score = f1_score(Y_test, y_pred, average='micro') 

print(score)



# neigh = KNeighborsClassifier()

# neigh.fit(X_train, Y_train)

# y_pred = neigh.predict(X_test)

# score = f1_score(Y_test, y_pred, average='micro') 

# print(score)	




# clf = RandomForestClassifier(random_state=0)

# clf.fit(X_train, Y_train)

# y_pred = clf.predict(X_test)

# score = f1_score(Y_test, y_pred, average='micro') 
# print(score)