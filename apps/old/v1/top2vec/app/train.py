import numpy as np 
import pandas as pd 
import json
import os
from top2vec import Top2Vec

df = pd.read_csv('health_tech.csv')
docs_article = df.loc[df.num_words_per_article>200].article_clean.tolist()
docs_title = df.title_clean.tolist()
docs_title = [ x for x in docs_title if type(x) == str ]

print('Articles:', len(docs_article))
print('Titles:', len(docs_title))

for model_size, model_name in dict({ 'sm': 'universal-sentence-encoder', 'lg': 'universal-sentence-encoder-large' }).items():
    for doc_type, docs in dict({ 'titles': docs_title, 'articles': docs_article }).items():
        for phrases in [ False, True ]:
            model_file = f'{doc_type}-{model_size}' + ('-phrases' if phrases else '') + '.t2v'
            print(model_file)
            model = Top2Vec(
                documents=docs,
                ngram_vocab=phrases,
                embedding_model=model_name,
                speed='deep-learn',
                workers=8
            )
            model.save(model_file)
