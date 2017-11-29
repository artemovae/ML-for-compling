# -*- coding: utf-8 -*-
from os import path
import glob
import pickle
import gensim
import numpy as np

num_of_features = 200


def create_model(name_of_alg, list_of_docs):
    sentences = [gensim.models.doc2vec.TaggedDocument(words=list_of_docs[doc], tags=[u'SENT_']) for doc
                 in range(len(list_of_docs))]
    model = gensim.models.doc2vec.Doc2Vec(sentences, size=num_of_features, min_count=70, window=70)
    model.save(name_of_alg.replace('lemmas', 'd2v_model') + '.mdl')
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    return model


def get_matrix(model, list_of_docs, alg_name):
    vectors_list = []
    for text_id in range(len(list_of_docs)):
        vector_for_each_text = []
        for word in list_of_docs[text_id]:
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
        vectors_array = np.vstack((vectors_array, vectors_list[i]))
    if vectors_array.shape == (1930,200):
        print('fuck yeah')
        with open(alg_name.replace('lemmas', 'vectors_d4v_') + '.pickle', 'wb') as f:
            pickle.dump(np.matrix(vectors_array), f)
    else:
        print(vectors_array.shape)


data = {}
for filename in glob.iglob(path.join(path.dirname(__file__), 'lemmas_from_*.pickle'), recursive=True):
    with open(filename, 'rb') as f:
        data_lemmas = pickle.load(f)
        data[path.splitext(path.basename(filename))[0]] = data_lemmas

models = {}
for filename in glob.iglob('d2v_model_from_*.mdl', recursive=True):
    model = gensim.models.Doc2Vec.load(filename)
    models[path.splitext(path.basename(filename))[0]] = model

for name_of_alg, list_of_docs in data.items():
    print(name_of_alg)
    model = create_model(name_of_alg, list_of_docs)
    get_matrix(model, list_of_docs, name_of_alg)
