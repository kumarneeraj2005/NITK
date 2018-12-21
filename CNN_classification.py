import pandas as pd
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
#from keras.layers import Dense, Dropout, Conv1D
#from keras.layers import Flatten
#from keras.layers import Embedding
from keras.layers import Dense, Embedding, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.utils import to_categorical
from sklearn import preprocessing

#%% Load cleaner function
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string

#initiate stemmer to stem words
stemmer = SnowballStemmer(language='english')
stemmer.stem('Wood preserving')
stop = set(stopwords.words('english'))

#initiate lemmatizer to reduce words to their lemma
from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()

#initiate list of punct to remove
exclude = set(string.punctuation)
#exclude.discard(set(["'"]))


doc_test = 'This is a test line'
def clean(doc):
    punc_free = ''.join([i for i in doc.lower() if i not in exclude])
    stop_free = " ".join([i for i in punc_free.split() if i not in stop])
    num_free = ''.join([ch for ch in stop_free if ch not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']])
    normalized = " ".join(stemmer.stem(word) for word in num_free.split())
    return normalized


#%%
#load the data set
train1 = pd.read_csv('/Users/tiwarn1/Desktop/NITS/Documnet Classification/stack_tags_3_classes_train_data.csv', encoding='latin1')

#train1 = pd.read_csv('file:///C:/Users/mayank_kapoor1/Documents/stack_tags_3_classes_train_data.csv', encoding='latin1')
#drop rows with null variables
train1 = train1.dropna()
#shuffle the rows of the data set randomly
train1 = train1.sample(frac=1).reset_index(drop=True)
#to lower
train1['post'] = train1.post.str.lower()
#apply the cleaner function
train1['post_cleaned'] = [clean(x) for x in train1.post]


#convert the features and labels to list
train_x = train1.post_cleaned.tolist()

#convert to cats
cats = preprocessing.LabelEncoder()
cats.fit(train1.tags)
train_y = cats.transform(train1.tags)

#train_y = train1.tags.astype('category').cat.codes.values
##train_y = pd.get_dummies(train1['tags'])
num_class = len(np.unique(train1.tags.values))
#%%
# prepare tokenizer to convert posts to sequences of text
#max words in dictionary = 1000
t = Tokenizer(num_words=2000)
t.fit_on_texts(train_x)
vocab_size = t.num_words

# integer encode the documents/encode new set of docs for prediction
encoded_docs = t.texts_to_sequences(train_x)
#print(encoded_docs)

#%%
# pad posts to a max length of 100 words
#this will be a uniform input to the model
max_length = 100
padded_docs = pad_sequences(encoded_docs,
                            maxlen=max_length, padding='post')
print(padded_docs)

#%%

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#split the dataset into training and validation datasets
train_x, valid_x, train_y, valid_y = train_test_split(padded_docs, train_y, test_size=0.4, random_state = 123)

#get one hot encoding for the target feature
#train_y = pd.get_dummies(train_y)
#valid_y = pd.get_dummies(valid_y)


#%%
#define keras sequential model
model = Sequential()
#embedding layer size 20
emb = Embedding(vocab_size, 20, input_length=max_length, trainable=True)
model.add(emb)
#add dropout
model.add(Dropout(0.3))
#adda convolutional layer with filter size 3
model.add(Conv1D(64, 3, activation='relu'))
#create a max poling filter with size 8
model.add(MaxPooling1D(pool_size=8))
#flatten the matrices to 1D
model.add(Flatten())
#final output layer with total number of classes as 20
#It is a one hot encoded out
model.add(Dense(num_class, activation='softmax'))

# compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])
# summarize the model
print(model.summary())
#fit the model
#print val - test accuracy after every epoch
model.fit(train_x, to_categorical(train_y),
          epochs=1, verbose=1, shuffle=True,
          validation_split=0.2)

#evaluate the model on separate validation
loss, accuracy = model.evaluate(valid_x, to_categorical(valid_y), verbose=1)
print('Accuracy: %f' % (accuracy*100), loss)
############################################################
############################################################
############################################################
############################################################
############################################################
#print the precision and recall numbers
predicted = model.predict(valid_x)
predicted = np.argmax(predicted, axis=1)
#get the actual labels
predicted_cats = cats.inverse_transform(predicted)
valid_y_cats = cats.inverse_transform(valid_y)
print(predicted_cats)
print(classification_report(valid_y_cats, predicted_cats))
pred_df = pd.DataFrame()
pred_df['valid_x__post_cleaned'] = t.sequences_to_texts(valid_x)
pred_df['actual_label'] = valid_y_cats
pred_df['predictions'] = predicted_cats
pred_df.to_csv('predictions.csv')
#print(pred_df)
#if want to see the classes wrt to doc cleaned text
input()