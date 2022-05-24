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

instance_url = 'localhost'
page_icon = Image.open('favicon.ico')
st.set_page_config(
    page_title='noobrainer',
    page_icon=page_icon,
    layout='wide',
    menu_items={ 'About': 'noobrainer v0.1\n\n---\n\nAPI/backend: http://{}:8000/docs\n\n---\n\nBryan T. Kim bryantaekim at gmail dot com\n\nDaniel Lee dslee47 at gmail dot com\n\nJuswaldy Jusman juswaldy at gmail dot com\n\n---\n\n'.format(instance_url) }
)

################################################################################
# Set up session state.

session_objects = {
    'api_url': 'http://localhost:8000',
    'tomo_header': 'Topics Most Common',
    'clustr_header': 'Historical Analysis',
    'ner_header': 'Test/Update the Model',
    'query': '',
    'num_topics': -1,
    'numwords_per_topic': -1,
    'topics_reduced': False,
    'last_request': '',
    'tomo_wordclouds': [],
    'tomo_model': 0,
    'clustr_model': 'models/clustr-titles-single-17.pkl',
    'ner_wordclouds': [],
    'ner_model': 0,
    'clustr_plot1': None,
    'clustr_plot2': None,
    'clustr_plot3': None,
    'clustr_plot4': None,
    'clustr_plot5': None,
    'clustr_df': {},
    'clustr_filepath': '',
    'clustr_corpus': None,
    'tomo_top10': []
}
for obj, val in session_objects.items():
    if obj not in st.session_state:
        st.session_state[obj] = val

################################################################################
# Main app.

app = MultiApp()
app.add_app(st.session_state.tomo_header, tomo.app)
app.add_app(st.session_state.clustr_header, clustr.app)
app.add_app(st.session_state.ner_header, ner.app)
app.run()
