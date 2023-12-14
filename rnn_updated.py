import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

import pickle

acc_map = {}

with open("./train_test_data/X_test.pkl", 'rb') as file:
    X_test = pickle.load(file)
with open("./train_test_data/X_train.pkl", 'rb') as file:
    X_train = pickle.load(file)
with open("./train_test_data/y_test.pkl", 'rb') as file:
    y_test = pickle.load(file)
with open("./train_test_data/y_train.pkl", 'rb') as file:
    y_train = pickle.load(file)

y_test['new_value'] = np.where(y_test['label'] == -1, 'NOT_' + y_test['value'], y_test['value'])
y_train['new_value'] = np.where(y_train['label'] == -1, 'NOT_' + y_train['value'], y_train['value'])
y_test['new_value'] = np.where(y_test['label'] == 0, 'Unlinked_' + y_test['value'], y_test['value'])
y_train['new_value'] = np.where(y_train['label'] == 0, 'Unlinked_' + y_train['value'], y_train['value'])


values = ["ACHIEVEMENT", "BENEVOLENCE", "CONFORMITY", "HEDONISM", "POWER", "SECURITY",
          "SELF-DIRECTION","STIMULATION", "TRADITION", "UNIVERSALISM"]

y_train = y_train["new_value"]
y_test = y_test["new_value"]

X_train = np.array([np.array(sublist) for sublist in X_train])
y_train = np.array([np.array(sublist) for sublist in y_train])
X_test = np.array([np.array(sublist) for sublist in X_test])
y_test = np.array([np.array(sublist) for sublist in y_test])


print(X_train)
print(y_train)
print("x_test")
print(X_test)
print("y_test")
print(y_test)

encoded_dict = {
    'POWER': 1,
    'Unliked_POWER': 0,
    'NOT_POWER': 1,
    'ACHIEVEMENT': 1,
    'Unlinked_ACHIEVEMENT':0,
    'NOT_ACHIEVEMENT':1,
    'BENEVOLENCE':1,
    'Unlinked_BENEVOLENCE':0,
    'NOT_BENEVOLENCE':1,
    'CONFORMITY':1,
    'Unlinked_CONFORMITY':0,
    'NOT_COMFORMITY':1,
    'HEDONISM':1,
    'Unlinked_HEDONISM':0,
    'NOT_HEDONISM':1,
    'SECURITY':1,
    'Unlinked_SECURITY':0,
    'NOT_SECURITY':1,
    'SELF-DIRECTION':1,
    'Unlinked_SELF-DIRECTION':0,
    'NOT_SELF-DIRECTION':1,
    'STIMULATION':1,
    'Unlinked_STIMULATION':0,
    'NOT_STIMULATION':1,
    'TRADITION':1,
    'Unlinked_TRADITION':0,
    'NOT_TRADITION':1,
    'UNIVERSALISM':1,
    'Unlinked_UNIVERSALISM':0,
    'NOT_UNIVERSALISM':1
}

mapping_dict = {
    0: 'POWER',
    1: 'Unliked_POWER',
    2: 'NOT_POWER',
    3: 'ACHIEVEMENT',
    4: 'Unlinked_ACHIEVEMENT',
    5: 'NOT_ACHIEVEMENT',
    6: 'BENEVOLENCE',
    7: 'Unlinked_BENEVOLENCE',
    8: 'NOT_BENEVOLENCE',
    9: 'CONFORMITY',
    10: 'Unlinked_CONFORMITY',
    11: 'NOT_COMFORMITY',
    12: 'HEDONISM',
    13: 'Unlinked_HEDONISM',
    14: 'NOT_HEDONISM',
    15: 'SECURITY',
    16: 'Unlinked_SECURITY',
    17: 'NOT_SECURITY',
    18: 'SELF-DIRECTION',
    19: 'Unlinked_SELF-DIRECTION',
    20: 'NOT_SELF-DIRECTION',
    21: 'STIMULATION',
    22: 'Unlinked_STIMULATION',
    23: 'NOT_STIMULATION',
    24: 'TRADITION',
    25: 'Unlinked_TRADITION',
    26: 'NOT_TRADITION',
    27: 'UNIVERSALISM',
    28: 'Unlinked_UNIVERSALISM',
    29: 'NOT_UNIVERSALISM'
}

column_y = ["label"]
column_x = ["scenario"]

y_train = pd.DataFrame(y_train, columns=column_y)
X_train = pd.DataFrame(X_train, columns=column_x)
X_test = pd.DataFrame(X_test, columns=column_x)
y_test = pd.DataFrame(y_test, columns=column_y)


for i in range(0,30,3):




    y_pos_train_i, x_pos_train_i = y_train.loc[y_train["label"] == mapping_dict[i]], X_train.loc[y_train["label"] == mapping_dict[i]]
    y_neutral_train_i, x_neutral_train_i = y_train.loc[y_train["label"] == mapping_dict[i+1]], X_train.loc[y_train["label"] == mapping_dict[i+1]]
    y_neg_train_i, x_neg_train_i = y_train.loc[y_train["label"] == mapping_dict[i+2]], X_train.loc[y_train["label"] == mapping_dict[i+2]]

    ytrain_pos = pd.concat([y_pos_train_i, y_neutral_train_i])
    xtrain_pos = pd.concat([x_pos_train_i, x_neutral_train_i])

    ytrain_neg = pd.concat([y_neg_train_i, y_neutral_train_i])
    xtrain_neg = pd.concat([x_neg_train_i, x_neutral_train_i])



    print(xtrain_pos)
    print(ytrain_pos)

    ytrain_pos = ytrain_pos.replace(encoded_dict)

    # Tokenize the sentences
    xtrain_pos = xtrain_pos['scenario'].values
    X_test = X_test['scenario'].values

    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer2 = Tokenizer(num_words=max_words)

    tokenizer.fit_on_texts(xtrain_pos)
    tokenizer2.fit_on_texts(X_test)

    X_seq = tokenizer.texts_to_sequences(xtrain_pos)
    X_seq2 = tokenizer2.texts_to_sequences(X_test)

    # Pad sequences to have consistent length
    max_len = 100

    X_pad = pad_sequences(X_seq, maxlen=max_len)
    X_pad2 = pad_sequences(X_seq2, maxlen=max_len)

    # Manually split the data into training and testing sets
    split_index = int(len(xtrain_pos))
    xtrain_pos, _ = X_pad[:split_index], X_pad[split_index:]
    ytrain_pos, _ = ytrain_pos[:split_index], ytrain_pos[split_index:]

    split_index = int(len(xtrain_pos))
    X_test, _ = X_pad2[:split_index], X_pad[split_index:]


    print(xtrain_pos)

    embedding_dim = 100
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
    model.add(LSTM(units=100))
    model.add(Dense(units=1, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model on the training set
    model.fit(xtrain_pos, ytrain_pos, epochs=5, batch_size=64)

    scores = model.predict(X_test)
    print(f'Softmax Scores for vals:')
    print(scores)
