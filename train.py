import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import nltk
from model import LanguageModel
import os
from nltk.stem import WordNetLemmatizer
import sys

# Download nltk models
downloaded = True
if not downloaded:
    nltk.download('punkt')
    nltk.download('wordnet')


# Function to save DL Model
def save_checkpoint(epoch, model, optimiser, loss, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimiser_state': optimiser.state_dict(),
        'loss': loss,
    }

    torch.save(checkpoint, save_path)

# Init Training Data
words, classes, documents = [], [], []
ignore_symbols = ['?', '!', ',']

intents = json.loads(open('intents.json').read())

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_symbols]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Saving the init words into pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# HYPERPARAMETERS
learning_rate = 0.001
num_epoch = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LanguageModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.train()

# Training Data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    #output_row = classes.index(doc[1])
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype=object)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])

print("Training data created", len(train_y[0]))

for epoch in range(num_epoch):
    num_classes = len(classes)
    pred_list = []
    total_correct = 0
    total_sample = 0
    correct_per_class = list(0 for i in range(num_classes))
    total_per_class = list(0 for i in range(num_classes))
    total_per_pred = list(0 for i in range(num_classes))
    recall_list = []
    precision_list = []
    f1_list = []
    loss_list = []
    print(epoch)
    assert len(train_x) == len(train_y)
    for i in range(len(train_x)):
        x = train_x[i]
        y = train_y[i]
        x, y = torch.Tensor(x).to(device), torch.Tensor(y).to(device)
        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Testing Accuracy
        pred_list.append(outputs.data.tolist())
        c = (torch.argmax(outputs) == torch.argmax(y))
        num_correct = int(c.sum())
        total_correct += num_correct
        total_sample += len(y)
        print("Loss:", loss.item(), c)

save_path = os.path.join(os.getcwd(), "saved_model", f"model_final_epoch{epoch}.pth.tar")
save_checkpoint(epoch, model, optimizer, loss_list, save_path)







