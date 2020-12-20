# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 15:19:04 2019

@author: hp
"""
from keras.datasets import imdb
from keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, Flatten,Dropout
from keras.models import Sequential
from keras.preprocessing import sequence

(X_train,y_train), (X_test,y_test) = imdb.load_data(num_words=5000)
word2id = imdb.get_word_index()
id2word = {i:word for word,i in word2id.items()}
# print([id2word.get(i) for i in X_train[6]])

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

model= Sequential()
model.add(Embedding(5000,64,input_length=max_words))
model.add(Conv1D(32,(3), activation = 'relu'))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.2))
model.add(Conv1D(16,(3), activation = 'relu'))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128,activation = 'relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer = 'adam', loss='binary_crossentropy',metrics=['accuracy'])
print(model.summary())
model.fit(X_train,y_train,batch_size = 32,epochs = 3)

scores= model.evaluate(X_test,y_test)
print(scores[1])

import numpy as np
from keras.preprocessing.text import Tokenizer
tk = Tokenizer()
test_sample = 'Not a good Movie'
test_sample = np.array(id2word.get(i) for i in test_sample)
test_sample=sequence.pad_sequences(test_sample, maxlen=max_words)

print(model.predict(test_sample))



model.save('nlp_embedding.h5')
from keras.models import load_model
model = load_model('nlp_embedding.h5')




