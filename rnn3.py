import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

acc_map = {}


values = ["ACHIEVEMENT", "BENEVOLENCE", "CONFORMITY", "HEDONISM", "POWER", "SECURITY",
          "SELF-DIRECTION","STIMULATION", "TRADITION", "UNIVERSALISM"]
for vals in values:

# Load the dataset
    file_path = "data/" + vals + ".csv"
    df = pd.read_csv(file_path)

# Preprocess the data
    X = df['scenario'].values
    y = df['label'].values

# Encode labels
    y = np.array([1 if label == 1 else 0 for label in y])

# Tokenize the sentences
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)

# Pad sequences to have consistent length
    max_len = 100
    X_pad = pad_sequences(X_seq, maxlen=max_len)

# Manually split the data into training and testing sets
    split_index = int(0.8 * len(X))
    X_train, X_test = X_pad[:split_index], X_pad[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

# Build the RNN model
    embedding_dim = 100
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
    model.add(LSTM(units=100))
    model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training set
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    acc_map[vals] = accuracy

for key in acc_map:
    print("accuracy for " + key + " is")
    print(acc_map[key])


