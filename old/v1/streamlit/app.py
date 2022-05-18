# Based on https://github.com/streamlit/example-app-bert-keyword-extractor
# Also this is very useful https://github.com/vivien000/st-click-detector

import streamlit as st
from st_click_detector import click_detector
import numpy as np
from pandas import DataFrame
import seaborn as sns
from PIL import Image
import os
import json

page_icon = Image.open('favicon.ico')
fourthbrain_logo = Image.open('fourthbrain-logo.png')

st.set_page_config(
    page_title='noobrainer',
    page_icon=page_icon,
    layout='wide'
)

c30, c31, c32 = st.columns([0.4, 2, 3])
with c30:
    st.image('n_6_lg.gif', width=100)
with c31:
    st.title('noobrainer')

with st.expander('About this app', expanded=False):
    st.write(
        """
        This is our capstone project for [fourthbrain.ai](fourthbrain.ai) MLE Cohort 6, Automated Metadata Tagging and Topic Modeling.
	    """
    )
    st.markdown('')

st.markdown('')
with st.form(key='my_form'):
    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        ModelType = st.radio(
            'Models',
            ['NER', 'Hierarchical Clustering', 'LDA Topic Modeling'],
            help='Currently we only have 3 options (NER, Clustering, or LDA) to explore the patterns in your text. More to come!',
        )

        kw_model = None

        top_N = st.slider(
            '# of results',
            min_value=1,
            max_value=30,
            value=10,
            help='You can choose the number of keywords/keyphrases to display. Between 1 and 30. Default is 10.',
        )
        min_Ngrams = st.number_input(
            'Minimum Ngram',
            min_value=1,
            max_value=4,
            help='The minimum value for the ngram range.'
        )

        max_Ngrams = st.number_input(
            'Maximum Ngram',
            value=2,
            min_value=1,
            max_value=4,
            help='The maximum value for the keyphrase_ngram_range.'
        )

        StopWordsCheckbox = st.checkbox(
            'Remove stop words',
            value=True,
            help='Tick this box to remove stop words from the document (currently English only)',
        )

        use_Phrases = st.checkbox(
            'Use Keyphrases',
            value=False,
            help='Tick this box to use keyphrases instead of just keywords (default False = keywords only)',
        )

        Diversity = st.slider(
            'Keyword diversity (MMR only)',
            value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            help="""The higher the setting, the more diverse the keywords.
            
Note that the *Keyword diversity* slider only works if the *MMR* checkbox is ticked.

""",
        )

    with c2:
        st.write('#### Step 1. Enter text')
        doc = st.text_area(
            'Type or paste your text here: (max 500 words)',
            height=100,
        )

        MAX_WORDS = 500
        import re
        res = len(re.findall(r'\w+', doc))
        if res > MAX_WORDS:
            st.warning(
                '⚠️ Your text contains '
                + str(res)
                + ' words.'
                + ' Only the first 500 words will be reviewed. Stay tuned as increased allowance is coming! 😊'
            )

            doc = doc[:MAX_WORDS]

        submit_button = st.form_submit_button(label='Submit')

    if use_Phrases:
        phrases = True
    else:
        phrases = False

    StopWords = 'english'

if not submit_button:
    st.stop()

if min_Ngrams > max_Ngrams:
    st.warning('min_Ngrams can\'t be greater than max_Ngrams')
    st.stop()

keywords = kw_model.extract_keywords(
    doc,
    keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
    use_mmr=mmr,
    stop_words=StopWords,
    top_n=top_N,
    diversity=Diversity,
)

st.header('')

cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

