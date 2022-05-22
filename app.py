"""
@author: noobrainers
@contact: Bryan T. Kim bryantaekim at gmail dot com
          Daniel Lee dslee47 at gmail dot com
          Juswaldy Jusman juswaldy at gmail dot com
@description: Streamlit main app file.
@credits: The MultiApp structure is taken from https://github.com/harmkenn/python-stat-tools
"""

import streamlit as st
from multiapp import MultiApp
from apps import ner, clustr, tomo
from PIL import Image
import pandas as pd

################################################################################
# Page settings.

page_icon = Image.open('favicon.ico')
st.set_page_config(
    page_title='noobrainer',
    page_icon=page_icon,
    layout="wide"
)

################################################################################
# Set up session state.

session_objects = {
    'query': '',
    'num_topics': -1,
    'numwords_per_topic': -1,
    'topics_reduced': False,
    'last_request': '',
    'tomo_wordclouds': [],
    'tomo_model': 'models/tomo-all-87k-articles-single-21.pkl',
    'clustr_model': 'models/clustr-titles-single-17.pkl',
    'ner_model': 'models/ner-healthtechother-articles-21.pkl',
    'clustr_plot1': None,
    'clustr_plot2': None,
    'clustr_plot3': None,
    'clustr_plot4': None,
    'clustr_plot5': None,
    'clustr_df': {}
}
for obj, val in session_objects.items():
    if obj not in st.session_state:
        st.session_state[obj] = val

################################################################################
# Main app.

app = MultiApp()
app.add_app("Topic Modeling", tomo.app)
app.add_app("Hierarchical Clustering", clustr.app)
app.run()
