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
    'topics_reduced': False,
    'last_request': '',
    'tomo_wordclouds': []
}
for obj, val in session_objects.items():
    if obj not in st.session_state:
        st.session_state[obj] = val

################################################################################
# The main query form.

with st.form(key='main_form'):
    query = st.text_area(
        label='',
        height=200,
        max_chars=500,
        help='Tell us what you are thinking',
        placeholder='Enter your request/report here...')
    submitted = st.form_submit_button(label='Submit')

if submitted:
    st.session_state['query'] = query

################################################################################
# Main app.

app = MultiApp()
app.add_app("Named Entity Recognition", ner.app)
app.add_app("Hierarchical Clustering", clustr.app)
app.add_app("Topic Modeling", tomo.app)
app.run()
