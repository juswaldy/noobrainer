"""
@author: noobrainers
@contact: Bryan T. Kim bryantaekim at gmail dot com
          Daniel Lee dslee47 at gmail dot com
          Juswaldy Jusman juswaldy at gmail dot com
@description: Streamlit Hierarchical Clustering app file.
"""
import streamlit as st
import pandas as pd
import sys
sys.path.append('../')
from models.schemas import *
from utils.clustr import *

class setVars:
    # Setting seed, label, feature, year, time period
    def __init__(self, **kwargs):
        self.rand_state = 42
        self.label = 'section_clean'
        self.feature = 'title_clean'
        self.year = '2020'
        self.time_period = 'month'
        self.times = 1
        self.filepath = 'C:/Users/gemin/4B_Cap_data'

def app():
    """ Hierarchical Clustering app """

    # Set variables.
    setvar = setVars()

    # Page settings.
    header = 'Hierarchical Clustering'

    # Page elements.
    with st.sidebar:
        st.subheader(header)
        d = st.date_input('Select date')

    # Load corpus.
    corpus = pd.read_csv('data/health_tech_time.csv', low_memory=False).reset_index(drop=True)

    # Get keywords from the user.
    keywords = st.text_input('Enter keywords separated by commas:')
    subset_cat = keywords.replace(' ', '').split(',')
    cols = ['id', 'date', 'year', 'month', 'day', 'section_clean', 'title_clean', 'article_clean']
    
    st.write(subset_cat, cols, setvar.label, setvar.feature, setvar.time_period, setvar.year)
    sub_corpus = prepCorpus(corpus, subset_cat, cols, setvar.label, setvar.feature, setvar.time_period, d.year)

    st.dataframe(sub_corpus)
    
    _raw_text = sub_corpus[sub_corpus[setvar.time_period].isin([setvar.times])][setvar.feature]
    _labels = sub_corpus[sub_corpus[setvar.time_period].isin([setvar.times])][setvar.label].values
    
    # Calculate similiarity and transform the corpus into a tf-idf matrix
    cos_sim, tfidf_matrix = tfidf_vectorizer(_raw_text, max_df=.5, min_df=.025)
    
    ### Plot dendrograms and distances among clusters
    #Z = _plot_dendrogram(cos_sim, _labels, zoom_in=False)
    #_ = _plot_dendrogram(cos_sim, _labels, zoom_xlim=4500)
    #_plot_distance(Z, 25)
    #_ = _plot_dendrogram(cos_sim, _labels, threshold=90, zoom_xlim=4500)
    
    ### Visualize the select clusters
    #_n_clusters = 4
    #clusters = _clustering(tfidf_matrix, _n_clusters)
    #xs, ys = _reduce_dim(cos_sim, setvar.rand_state)

    #df = pd.DataFrame(dict(x=xs, y=ys, segment=clusters, label=_labels, text=_raw_text.values)) 
    
    #_plot_clusters(df)
