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
from datetime import datetime
from glob import glob

class setVars:
    # Setting seed, label, feature, year, time period
    def __init__(self, **kwargs):
        self.rand_state = 42
        self.label = 'section_clean'
        self.feature = 'title_clean'
        self.year = '2020'
        self.time_period = 'month'
        self.times = 1
        self.day = 1
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
        
        st.session_state.clustr_model = st.selectbox('Choose a Model', glob('models/clustr-*'))

        # Get range and start date.
        st.slider('Select starting date')

        # Get user specified threshold for number of ??.
        threshold = st.slider('Select number of ??', 40, 200, 5)

    start_date, end_date = st.slider(
        "Select date range:",
        min_value=datetime(2016, 1, 1),
        max_value=datetime(2020, 12, 31),
        value=(datetime(2016, 1, 1), datetime(2016, 1, 31)))
    start_date, end_date = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    st.write("You're scheduled for:", start_date, end_date)

    # Load corpus.
    import time
    start = time.time()
    corpus = pd.read_csv('data/health_tech_time.csv', low_memory=False).reset_index(drop=True).fillna('none')
    end = time.time()
    st.write((end - start))

    # Get keywords from the user.
    keywords = st.text_input('Enter keywords separated by commas:')
    subset_cat = keywords.replace(' ', '').split(',')
    cols = ['id', 'date', 'year', 'month', 'day', 'section_clean', 'title_clean', 'article_clean']

    start = time.time()
    cos_sim, tfidf_matrix, _labels, _raw_text = get_tfidf(corpus, subset_cat, cols, setvar.label, setvar.feature, start_date, end_date)
    end = time.time()
    st.write((end - start))

    ### Plot dendrograms and distances among clusters
    fig, Z = plot_dendrogram(cos_sim, _labels, zoom_in=False)
    st.pyplot(fig)

    fig, _ = plot_dendrogram(cos_sim, _labels, zoom_xlim=4500)
    st.pyplot(fig)

    fig = plot_distance(Z, 25)
    st.pyplot(fig)
    
    fig, _ = plot_dendrogram(cos_sim, _labels, threshold=threshold, zoom_xlim=4500)
    st.pyplot(fig)

    ### Visualize the select clusters
    _n_clusters = 4
    clusters = clustering(tfidf_matrix, _n_clusters)
    xs, ys = reduce_dim(cos_sim, setvar.rand_state)

    df = pd.DataFrame(dict(x=xs, y=ys, segment=clusters, label=_labels, text=_raw_text.values))
    
    fig = plot_clusters(df)
    st.pyplot(fig)
