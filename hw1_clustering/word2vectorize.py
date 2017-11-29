# -*- coding: utf-8 -*-
from os import path
import glob
import pickle
import gensim
import numpy as np

num_of_features = 200

def create_model(name_of_alg, list_of_docs):
    model = gensim.models.Word2Vec(list_of_docs, size=num_of_features, min_count=30, window=30)
    model.save(name_of_alg.replace('lemmas','w2v_model2')+'.mdl')
    return model

def get_matrix(model, texts, alg_name):
    vectors_list = []
    for text_id in range(len(texts)):
        vector_for_each_text = []
        for word in texts[text_id]:
            try:
                featureVec = np.zeros(shape=(1, num_of_features), dtype='float32')
                featureVec = np.add(featureVec, model[word])
                vector_for_each_text.append(featureVec)
            except KeyError:
                pass

        first = np.array(vector_for_each_text[0])

        for i in range(1, len(vector_for_each_text)):
            first = np.add(first, vector_for_each_text[i])
        resultVec = np.divide(first, len(vector_for_each_text))
        vectors_list.append(resultVec)

    vectors_array = np.array(vectors_list[0])

    for i in range(1, len(vectors_list)):
        vectors_array = np.vstack((vectors_array,vectors_list[i]))

    with open(alg_name.replace('lemmas','vectors2')+'.pickle', 'wb') as f:
            pickle.dump(np.matrix(vectors_array), f)


data = {}
for filename in glob.iglob(path.join(path.dirname(__file__),'lemmas_from_*.pickle'), recursive=True):
    with open(filename, 'rb') as f:
        data_lemmas = pickle.load(f)
        data[path.splitext(path.basename(filename))[0]] = data_lemmas

for name_of_alg, list_of_docs in data.items():
    model = create_model(name_of_alg, list_of_docs)
    get_matrix(model,list_of_docs,name_of_alg)