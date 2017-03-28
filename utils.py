import pandas as pd
import numpy as np
from tqdm import tqdm

from keras.preprocessing import sequence, text
from keras.utils import np_utils

def get_data_and_embeddings(fname):
    train = pd.read_csv("data/train.csv.gz")
    data = pd.read_csv(fname)
    try:
        y = data.is_duplicate.values
    except:
        print("No target values -- assuming this is test data")
        y = None
    
    tk = text.Tokenizer(num_words=200000)
    
    max_len = 40
    tk.fit_on_texts(list(train.question1.values) + list(train.question2.values.astype(str)))
    del train
    x1 = tk.texts_to_sequences(data.question1.values.astype(str))
    x1 = sequence.pad_sequences(x1, maxlen=max_len)
    
    x2 = tk.texts_to_sequences(data.question2.values.astype(str))
    x2 = sequence.pad_sequences(x2, maxlen=max_len)

    word_index = tk.word_index
    
    
    embeddings_index = {}
    f = open('data/glove.840B.300d.txt')
    for i, line in enumerate(tqdm(f)):
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            continue
        embeddings_index[word] = coefs
    f.close()
        
    print('Found %s word vectors.' % len(embeddings_index))
    
    embedding_matrix = np.zeros((len(word_index) + 1, 300)).astype(np.float32)
    for word, i in tqdm(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
                
    return x1, x2, y, embedding_matrix, word_index
