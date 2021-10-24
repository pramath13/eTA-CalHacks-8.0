import json
import pickle
import numpy as np
import torch
import random
import nltk


words, classes, documents = [], [], []
ignore_symbols = ['?', '!', ',']

with open('xx.json', 'r').read() as json_file:
    intents = json.loads(json_file)


