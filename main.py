from model import LanguageModel
import torch
import os
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random

model = LanguageModel()
checkpoint = torch.load(os.path.join(os.getcwd(), 'saved_model', 'model_final_epoch99.pth.tar'))
model.load_state_dict(checkpoint['model_state'])
model.eval()

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    print(p)
    res = model(torch.Tensor(p))
    print(res)
    #ERROR_THRESHOLD = 0.25
    results = [[i,r.item()] for i,r in enumerate(res)] #if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list

def getResponse(ints, intents_json):
    highest_prob = ints[0]['probability']
    intent_pred = ints[0]['intent']
    for i in range(len(ints)):
        tag = ints[i]['intent']
        prob = ints[i]['probability']

        if prob > highest_prob:
            #print("HEre", prob, tag)
            highest_prob = prob
            intent_pred = tag
    print("tag", intent_pred)
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== intent_pred):
            result = random.choice(i['responses'])
            return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    #print(ints)
    res = getResponse(ints, intents)
    print(msg, "RES", res)
    return res

#chatbot_response("Hey")