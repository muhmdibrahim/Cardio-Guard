import random
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

lemmatizer = WordNetLemmatizer()
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")

words = []
classes = []
documents = []
ignore_words = ["?", "!"]
data_file = open("intents.json").read()
intents = json.loads(data_file)

# words
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent["tag"])) 

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# lemmatizer
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents", '\n', documents, '\n')
print(len(classes), "classes", '\n', classes, '\n')
print(len(words), "unique lemmatized words", '\n', words, '\n')

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0] 
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)

x = []
y = []
for features, label in training:
    x.append(features)
    y.append(label)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=20, random_state=42)
val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=0.5, random_state=42)

print("Training data created")
print(len(train_x))
print("Training data created")
print(len(train_y))

# Sequential
model = Sequential()
model.add(Dense(512, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))
model.summary()

adam = Adam(learning_rate= 0.001) 
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

# fitting and saving the model, acc = 89%
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=10, validation_data=(val_x, val_y), verbose=2)
loss, accuracy = model.evaluate(test_x, test_y)
print(f'Sequential Test Loss: {loss}, Sequential Test Accuracy: {accuracy}')
model.save("cardioguardbot.h5", hist)
#model.save("chatbot_model.h5", hist)

# Plot training and validation accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

print("Model created")

# RNN: LSTM 
'''
from keras.layers import LSTM, Conv1D, MaxPooling1D

from keras.layers import Embedding

top_words = 5000
max_review_length = len(words)
epochs = 100
embedding_vecor_length = 16

model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(16)) 
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
model.summary()

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
adam= Adam(learning_rate = 0.0001)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

# fitting and saving the model, acc = 56%
hist= model.fit(np.array(train_x), np.array(train_y), epochs=epochs, batch_size=10,validation_data=(val_x, val_y), verbose=1)
print("model created")

loss, accuracy = model.evaluate(test_x, test_y)
print(f'RNN: LSTM Test Loss: {loss}, RNN: LSTM Test Accuracy:{accuracy}')

# Plot training and validation accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

print("model created")
'''

# CNN with LSTM
'''
from keras.layers import LSTM, Conv1D, MaxPooling1D
from keras.layers import Embedding

top_words = 5000
max_review_length = len(words)
epochs = 200
embedding_vecor_length = 16

model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(16)) # dropout=0.2, recurrent_dropout=0.2
model.add(Dense(len(train_y[0]), activation='softmax'))
model.summary()

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
adam = Adam(learning_rate = 0.001)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

# fitting and saving the model, acc = 54%
hist = model.fit(np.array(train_x), np.array(train_y), epochs=epochs, batch_size=5, validation_data=(val_x, val_y), verbose=1)

loss, accuracy = model.evaluate(test_x, test_y)
print(f'CNN: LSTM Test Loss: {loss}, CNN: LSTM Test Accuracy:{accuracy}')

# Plot training and validation accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
print("model created")
'''

# Bert 84.45%
'''
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Load and preprocess the data
data_file = open("intents.json").read()
intents = json.loads(data_file)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentences = []
labels = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

encoded_inputs = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='tf')
input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']

label_set = list(set(labels))
label_map = {label: i for i, label in enumerate(label_set)}
label_ids = np.array([label_map[label] for label in labels])

input_ids = input_ids.numpy()  # Convert input_ids to NumPy array
label_ids = label_ids.reshape(-1)  # Reshape label_ids to match input_ids shape

train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, label_ids, test_size=0.2, random_state=42)
val_inputs, test_inputs, val_labels, test_labels = train_test_split(test_inputs, test_labels, test_size=0.5, random_state=42)

# Convert the data back to TensorFlow tensors
train_inputs = tf.convert_to_tensor(train_inputs)
test_inputs = tf.convert_to_tensor(test_inputs)
val_inputs = tf.convert_to_tensor(val_inputs)
train_labels = tf.convert_to_tensor(train_labels)
test_labels = tf.convert_to_tensor(test_labels)
val_labels = tf.convert_to_tensor(val_labels)

# Load the BERT model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Define the model architecture
class BertClassifier(tf.keras.Model):
    def __init__(self, bert_model, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        outputs = self.bert(inputs)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

num_classes = len(label_set)
model = BertClassifier(bert_model, num_classes)

# Train the model
adam = Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=adam, loss=loss_fn, metrics=['accuracy'])
model.fit(train_inputs, train_labels, validation_data=(val_inputs, val_labels), batch_size=30, epochs=300)

# Evaluate the model
loss, accuracy = model.evaluate(test_inputs, test_labels)
print(f'Bert Test Loss: {loss}, Bert Test Accuracy: {accuracy}')

# Plot training and validation accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
'''