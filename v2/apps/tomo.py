"""
@author: noobrainers
@contact: Bryan T. Kim bryantaekim at gmail dot com
          Daniel Lee dslee47 at gmail dot com
          Juswaldy Jusman juswaldy at gmail dot com
@description: Streamlit Topic Modeling app file.
"""

import streamlit as st
import json
import requests
import sys
sys.path.append('../')
from models.schemas import *
from wordcloud import WordCloud
from scipy.special import softmax
import random

wordcloud_backgrounds = [
    'ivory',
    'whitesmoke',
    'snow',
    'floralwhite',
    'honeydew',
    'azure',
    'beige',
    'mintcream',
    'oldlace',
    'ghostwhite'
]

def display_topic_wordcloud(img, cols, numcols, n, topic_score):
    """ Display topic wordcloud """
    with cols[n].container():
        cols[n].write('`{:.1%}`'.format(topic_score))
        cols[n].image(img, use_column_width=True)
    n += 1
    if n >= numcols:
        n = 0
    return n
    

def app():
    """ Topic Modeling app """

    # Page settings.
    header = 'Topic Modeling'

    # Page elements.
    with st.sidebar:
        st.subheader(header)
        st.session_state.num_topics = st.slider('Number of topics', min_value=1, max_value=40)
        st.session_state.topics_reduced = st.checkbox('Topics reduced')

    # Prepare topic query request from the current input fields.
    request = json.dumps(PredictionRequest(
        query_string=st.session_state.query,
        num_topics=st.session_state.num_topics,
        topics_reduced=st.session_state.topics_reduced).__dict__)

    # If query is empty, reset the session state.
    if st.session_state.query == '':
        del st.session_state.tomo_wordclouds
        st.session_state.tomo_wordclouds = []
        st.session_state.last_request = ''
    else:
        # Show header before wordclouds.
        st.subheader('Similar topics: (% cosine similarity)')

        # Send topic query and show response as wordclouds.
        numcols = 5
        cols = st.columns(numcols)
        if st.session_state.last_request == request:

            # If parameters haven't changed, show the previous wordclouds.
            n = 0
            for (topic_score, img) in st.session_state.tomo_wordclouds:
                n = display_topic_wordcloud(img, cols, numcols, n, topic_score)

        else:
            del st.session_state.tomo_wordclouds
            st.session_state.tomo_wordclouds = []

            # Send request to the server and get the response.
            response = requests.post('http://localhost:8000/tomo/topics/query', data=request)

            # Show response as wordclouds.
            n = 0
            bgcolor = random.choice(wordcloud_backgrounds)
            for r in response.json():
                # Generate wordcloud from words/scores.
                topic_word_scores = dict(zip(r['topic_words'], softmax(r['word_scores'])))
                img = WordCloud(width=320, height=240, background_color=bgcolor).generate_from_frequencies(topic_word_scores).to_image()
                
                # Display wordcloud.
                n = display_topic_wordcloud(img, cols, numcols, n, r['topic_score'])

                # Save wordcloud to session state.
                st.session_state.tomo_wordclouds.append([r['topic_score'], img])

            # Save request to session state.
            st.session_state.last_request = request
    
