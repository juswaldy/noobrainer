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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io

class setVars:
    # Setting seed, label, feature
    def __init__(self, **kwargs):
        self.rand_state = 42
        self.label = 'label'
        self.feature = 'title_clean'

def fig2img(figure):
    img_buf = io.BytesIO()
    figure.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    return im

def app():
    """ Hierarchical Clustering app """

    # Set variables.
    setvar = setVars()

    st.header(st.session_state.clustr_header)

    # Page elements.
    with st.sidebar:
        st.subheader(st.session_state.clustr_header)
        
        # If debugging, show option to choose corpus.
        if st.session_state.debug:
            corpus_filepath = st.selectbox('Choose a Corpus', sorted(glob('data/*')))
        else:
            corpus_filepath = ''

        with st.form(key='clustr_plot1_form'):
            if st.session_state.debug:
                st.write('**Step 1. Select a date range**')

            # Get range and start date.
            start_date = st.date_input('Starting date', datetime(2018, 10, 31))
            if st.session_state.debug:
                num_days = st.slider('Number of days', 1, 31, 31)
            else:
                # If not debug, set number of days to 3.
                num_days = 3
            end_date = start_date + timedelta(days=num_days)
            st.write(f'Range: {start_date} to {end_date}')
            plot1_submitted = st.form_submit_button(label='Submit')

        if st.session_state.debug:
            with st.form(key='clustr_plot2_form'):
                st.write('**Step 2. Separate the clusters**')
                # Get distance threshold from user.
                distance_threshold = st.text_input('Distance threshold (0-1000)', '')
                distance_threshold = int(distance_threshold) if distance_threshold.isdigit() else 0
                plot2_submitted = st.form_submit_button(label='Submit')

            with st.form(key='clustr_plot3_form'):
                st.write('**Step 3. Clusters from Step 2**')
                _n_clusters = int(st.text_input('Number of clusters (min 2)', '2'))
                plot3_submitted = st.form_submit_button(label='Submit')

    # Load corpus.
    if st.session_state.corpus_filepath == corpus_filepath and st.session_state.clustr_corpus:
        corpus = st.session_state.clustr_corpus
    else:
        corpus_filepath = st.session_state.corpus_filepath if corpus_filepath == '' else corpus_filepath
        corpus = pd.read_csv(corpus_filepath, low_memory=False).reset_index(drop=True).fillna('none')
        st.session_state.clustr_corpus = corpus
        st.session_state.corpus_filepath = corpus_filepath

    # Show samples from corpus.
    st.write('Samples from the corpus:')
    cols = [ 'title_clean', 'label', 'date' ]
    samples = corpus[cols].sample(4, random_state=52).reset_index(drop=True)
    samples.label = [ st.session_state.ner_labels[i] for i in samples.label ]
    samples.date = samples.date.apply(lambda x: x[:10])
    st.dataframe(samples, height=600)

    subset_cat = []

    # Draw the selected plots.
    if plot1_submitted:
        cos_sim, tfidf_matrix, _labels, _raw_text = get_tfidf(corpus, subset_cat, cols, setvar.label, setvar.feature, str(start_date), str(end_date))

        if st.session_state.debug:
            fig, Z = plot_dendrogram(cos_sim, _labels, zoom_in=False)
            st.pyplot(fig)
            image = fig2img(fig)
            st.session_state.clustr_plot1 = image

        fig, _ = plot_dendrogram(cos_sim, _labels, zoom_xlim=150*num_days)
        st.pyplot(fig)
        image = fig2img(fig)
        st.session_state.clustr_plot2 = image

        if st.session_state.debug:
            with st.expander('See details', expanded=False):
                fig = plot_distance(Z, 25)
                st.pyplot(fig)
                image = fig2img(fig)
                st.session_state.clustr_plot3 = image
        
        st.session_state.cos_sim = cos_sim
        st.session_state.tfidf_matrix = tfidf_matrix
        st.session_state.labels = [ st.session_state.ner_labels[x] for x in _labels ]
        st.session_state.raw_text = _raw_text
    else:
        # If plots exist, display them.
        if st.session_state.clustr_plot1 is not None:
            if st.session_state.debug:
                st.image(st.session_state.clustr_plot1)
            st.image(st.session_state.clustr_plot2)
            if st.session_state.debug:
                with st.expander('See details', expanded=False):
                    st.image(st.session_state.clustr_plot3)

    if st.session_state.debug:
        if plot2_submitted:
            # Visualize distance threshold.
            fig, _ = plot_dendrogram(st.session_state.cos_sim, st.session_state.labels, threshold=distance_threshold, zoom_xlim=1500)
            st.pyplot(fig)
            image = fig2img(fig)
            st.session_state.clustr_plot4 = image
        else:
            # If plots exist, display them.
            if st.session_state.clustr_plot4 is not None:
                st.image(st.session_state.clustr_plot4)

    segment_detail_cols = ['segment', 'label', 'text']
    if st.session_state.debug:
        if plot3_submitted:
            clusters = clustering(st.session_state.tfidf_matrix, _n_clusters)
            xs, ys = reduce_dim(st.session_state.cos_sim, setvar.rand_state)

            df = pd.DataFrame(dict(x=xs, y=ys, segment=clusters, label=st.session_state.labels, text=st.session_state.raw_text.values), index=None)
            st.session_state.clustr_df = df.to_dict()

            fig = plot_clusters(df)
            st.pyplot(fig)
            image = fig2img(fig)
            st.session_state.clustr_plot5 = image

            with st.expander('See details', expanded=False):
                st.dataframe(df[segment_detail_cols], width=5000)
                segments = pd.DataFrame(df.segment.value_counts(), index=None)
                st.dataframe(segments)
        else:
            # If plots exist, display them.
            if st.session_state.clustr_plot5 is not None:
                st.image(st.session_state.clustr_plot5)
                with st.expander('See details', expanded=False):
                    df = pd.DataFrame(st.session_state.clustr_df, index=None)
                    st.dataframe(df[segment_detail_cols], width=5000)
                    segments = pd.DataFrame(df.segment.value_counts(), index=None)
                    st.dataframe(segments)
