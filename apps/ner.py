"""
@author: noobrainers
@contact: Bryan T. Kim bryantaekim at gmail dot com
          Daniel Lee dslee47 at gmail dot com
          Juswaldy Jusman juswaldy at gmail dot com
@description: Streamlit Named Entity Recognition app file.
"""
import streamlit as st
import pandas as pd
import sys
sys.path.append('../')
from models.schemas import *
from utils.ner import *

def app():
    """ Named Entity Recognition app """

    # Page settings.
    header = 'Named Entity Recognition'

    # Page elements.
    with st.form(key='ner_form'):
        query = st.text_area(
            label='',
            height=200,
            max_chars=1000,
            help='Tell us what you are thinking',
            placeholder='Enter your request/report here...')
        submitted = st.form_submit_button(label='Submit')

    if submitted:
        st.session_state.query = query

    with st.sidebar:
        st.subheader(header)
    
    # Load corpus.
    df = pd.read_csv('data/small-articles-single.csv', low_memory=False).reset_index(drop=True)

    # Get keywords.
    keywords = set()
    for words in df.topic_words.values:
        keywords.update(eval(words.replace('\n', '').replace(' ', ',')))

    # Get positive and negative keywords.
    col1, col2, _ = st.columns([1, 1, 2])
    keywords_pos = col1.multiselect('Select keywords', list(keywords))
    keywords_neg = col2.multiselect('Select negative keywords', list(keywords.difference(set(keywords_pos))))
