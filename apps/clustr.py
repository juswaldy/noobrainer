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
from datetime import datetime, timedelta
from glob import glob

class setVars:
    # Setting seed, label, feature, year, time period
    def __init__(self, **kwargs):
        self.rand_state = 42
        self.label = 'section_clean'
        self.feature = 'title_clean'

def app():
    """ Hierarchical Clustering app """

    # Set variables.
    setvar = setVars()

    # Page settings.
    header = 'Hierarchical Clustering'

    # Load corpus.
    corpus = pd.read_csv('data/health_tech_time.csv', low_memory=False).reset_index(drop=True).fillna('none')

    # Show samples from corpus.
    st.write('Samples from the corpus:')
    cols = [ 'section_clean', 'title_clean', 'article_clean', 'date', 'num_words_per_article' ]
    st.dataframe(corpus[cols].sample(3).reset_index(drop=True), height=500)

    # Page elements.
    with st.sidebar:
        st.subheader(header)
        
        st.session_state.clustr_model = st.selectbox('Choose a Corpus', glob('data/clustr-*'))

        with st.form(key='clustr_plot1_form'):
            # Get range and start date.
            start_date = st.date_input('Select starting date', datetime(2016, 1, 1))
            num_days = st.slider('Number of days', 1, 31, 31)
            end_date = start_date + timedelta(days=num_days)
            plot1_submitted = st.form_submit_button(label='Submit')

        with st.form(key='clustr_plot2_form'):
            # Get distance threshold from user.
            distance_threshold = st.slider('Select distance threshold', 40, 200, 80)
            plot2_submitted = st.form_submit_button(label='Submit')

        with st.form(key='clustr_plot3_form'):
            _n_clusters = int(st.text_input('Number of clusters (min 2)', '2'))
            plot3_submitted = st.form_submit_button(label='Submit')

    # Get keywords from the user.
    keywords = st.text_input('Enter keywords separated by commas:')
    subset_cat = keywords.replace(' ', '').split(',')
    cols = ['id', 'date', 'year', 'month', 'day', 'section_clean', 'title_clean', 'article_clean']

    # PIL.Image.frombytes('RGB', 
    #fig.canvas.get_width_height(),fig.canvas.tostring_rgb())

    # Draw the selected plots.
    if plot1_submitted:
        cos_sim, tfidf_matrix, _labels, _raw_text = get_tfidf(corpus, subset_cat, cols, setvar.label, setvar.feature, str(start_date), str(end_date))

        # Plot dendrograms and distances among clusters.
        fig, Z = plot_dendrogram(cos_sim, _labels, zoom_in=False)
        st.pyplot(fig)

        fig, _ = plot_dendrogram(cos_sim, _labels, zoom_xlim=4500)
        st.pyplot(fig)

        with st.expander('See details', expanded=False):
            fig = plot_distance(Z, 25)
            st.pyplot(fig)

        # Visualize distance threshold.
        fig, _ = plot_dendrogram(cos_sim, _labels, threshold=distance_threshold, zoom_xlim=4500)
        st.pyplot(fig)

        clusters = clustering(tfidf_matrix, _n_clusters)
        xs, ys = reduce_dim(cos_sim, setvar.rand_state)

        df = pd.DataFrame(dict(x=xs, y=ys, segment=clusters, label=_labels, text=_raw_text.values), index=None)

        fig = plot_clusters(df)
        st.pyplot(fig)

        with st.expander('See details', expanded=False):
            st.dataframe(df)
            segments = pd.DataFrame(df.segment.value_counts(), index=None)
            st.dataframe(segments)
