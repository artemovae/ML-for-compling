import pickle
import time
from nltk.tokenize import RegexpTokenizer
from re import sub
from pandas import DataFrame
from pymystem3 import Mystem
import texterra
from pymorphy2 import MorphAnalyzer

API_KEY = '9988cfb979b80264baeba1386cc7e455f99f943c'

morph = MorphAnalyzer()
m = Mystem()
t = texterra.API(API_KEY)
alpha_tokenizer = RegexpTokenizer('\w+')
df_events = DataFrame.from_csv('events.csv')
df_news = DataFrame.from_csv('raw_news.csv')
texts = list(df_news.text.values)


def remove_url(text):
    return sub(r'http\S+', '', text)


def normalize_with_pymorphy(tokens):
    return [morph.parse(word)[0].normal_form for word in tokens]


def normalize_with_mystem(tokens):
    return ''.join(m.lemmatize(' '.join(tokens))).split()


def normalize_with_texterra(tokens):
    time.sleep(10)
    text = ' '.join(tokens)
    return [token[3] for token in list(t.lemmatization(text))[0]]


all_type_of_tokens = {}
tokenizers = {'pymorphy': normalize_with_pymorphy, 'mystem': normalize_with_mystem, 'texterra': normalize_with_texterra}

for name, tokenizer in tokenizers.items():
    all_texts = []
    for text in texts:
        print(texts.index(text))
        text = remove_url(text)
        tokens = alpha_tokenizer.tokenize(text)
        tokens = tokenizer(tokens)
        all_texts.append(tokens)
    with open('tokens_from_' + name + '.pickle', 'wb') as f:
        pickle.dump(all_texts, f)
