import numpy as np
import pickle as pkl

with open('ARABIC_LETTERS_LIST.pickle', 'rb') as file:
    ARABIC_LETTERS_LIST = pkl.load(file)
with open('DIACRITICS_LIST.pickle', 'rb') as file:
    DIACRITICS_LIST = pkl.load(file)
with open('RNN_SMALL_CHARACTERS_MAPPING.pickle', 'rb') as file:
        CHARACTERS_MAPPING = pkl.load(file)
with open('RNN_CLASSES_MAPPING.pickle', 'rb') as file:
    CLASSES_MAPPING = pkl.load(file)
with open('RNN_REV_CLASSES_MAPPING.pickle', 'rb') as file:
    REV_CLASSES_MAPPING = pkl.load(file)


def to_one_hot(data, size):
    one_hot = list()
    for elem in data:
        cur = [0] * size
        cur[elem] = 1
        one_hot.append(cur)
    return one_hot

def map_data(data_raw):
    X = list()
    Y = list()
    
    for line in data_raw:        
        x = [CHARACTERS_MAPPING['<SOS>']]
        y = [CLASSES_MAPPING['<SOS>']]
        
        for idx, char in enumerate(line):
            if char in DIACRITICS_LIST:
                continue
            
            x.append(CHARACTERS_MAPPING[char])
            
            if char not in ARABIC_LETTERS_LIST:
                y.append(CLASSES_MAPPING[''])
            else:
                char_diac = ''
                if idx + 1 < len(line) and line[idx + 1] in DIACRITICS_LIST:
                    char_diac = line[idx + 1]
                    if idx + 2 < len(line) and line[idx + 2] in DIACRITICS_LIST and char_diac + line[idx + 2] in CLASSES_MAPPING:
                        char_diac += line[idx + 2]
                    elif idx + 2 < len(line) and line[idx + 2] in DIACRITICS_LIST and line[idx + 2] + char_diac in CLASSES_MAPPING:
                        char_diac = line[idx + 2] + char_diac
                y.append(CLASSES_MAPPING[char_diac])
        
        assert(len(x) == len(y))
        
        x.append(CHARACTERS_MAPPING['<EOS>'])
        y.append(CLASSES_MAPPING['<EOS>'])
        
        y = to_one_hot(y, len(CLASSES_MAPPING))
        
        X.append(x)
        Y.append(y)
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    return X, Y