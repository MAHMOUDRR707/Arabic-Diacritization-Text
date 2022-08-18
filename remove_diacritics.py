import pickle as pkl

with open('DIACRITICS_LIST.pickle', 'rb') as file:
    DIACRITICS_LIST = pkl.load(file)

def remove_diacritics(data_raw):
    return data_raw.translate(str.maketrans('', '', ''.join(DIACRITICS_LIST)))