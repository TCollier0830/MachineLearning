
'''#Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated textstarts sounding coherent.
It is recommended to run this script on GPU, as recurrentnetworks are quite computationally intensive.
If you try this script on new data, make sure your corpushas at least ~100k characters. ~1M is better.'''
from __future__ import print_function
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
import numpy as np
import random
import sys
import io

with io.open('Genesis.txt', encoding='utf-8') as f:
    text = f.read().lower()
print('text length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 58
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
# build the model: one LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(500, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.01))
model.add(Dense(500, activation='selu', kernel_regularizer=l2(0.001)))
model.add(Dense(len(chars), activation='softmax'))
optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.clip(preds, a_max = preds.max(), a_min = .0000000001)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, _):

    # Function invoked at end of each epoch. Prints generated text.
    #print()
    #print('----- Generating text after Epoch: %d' % epoch)

    if epoch%10 == 0:
        output.write('begin epoch: ' + str(epoch) + '\n')
        start_index = 0
        for diversity in [0.2, 0.5]:
            output.write('begin diversity: ' + str(diversity) + '\n')
            print('----- diversity:', diversity)
            generated = ''        
            sentence = text[start_index: start_index + 58]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)
            output.write(generated)
            for i in range(200):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                sentence = sentence[1:] + next_char
                output.write(str(next_char))
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
            output.write('\n')
    else:
        pass

    return

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

output = open('wisdom.txt', 'a+')

model.fit(x, y,
          batch_size=40,
          epochs=301,
          callbacks=[print_callback])