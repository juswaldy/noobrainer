"""
@author: noobrainers
@contact: Bryan T. Kim bryantaekim at gmail dot com
          Daniel Lee dslee47 at gmail dot com
          Juswaldy Jusman juswaldy at gmail dot com
@description: Streamlit Topic Modeling app file.
"""
import streamlit as st
from st_click_detector import click_detector
import json
import requests
from glob import glob
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

def display_topic_wordcloud_clickable(img, cols, numcols, n, topic_score, topic_num):
    """ Display clickable topic wordcloud """
    with cols[n].container():
        cols[n].write('`{:.1%}`'.format(topic_score))
        cols[n].image(img, use_column_width=True)
        cols[n].button()
    n += 1
    if n >= numcols:
        n = 0
    return n
    

def app():
    """ Topic Modeling app """

    # Page settings.
    header = 'Topic Modeling'

    # Page elements.
    with st.form(key='main_form'):
        query = st.text_area(
            label='',
            height=200,
            max_chars=1000,
            help='Tell us what you are thinking',
            placeholder='Enter your request/report here...')
        submitted = st.form_submit_button(label='Submit')
    
    if submitted:
        st.session_state['query'] = query

    with st.sidebar:
        st.subheader(header)
        st.session_state.tomo_model_path = st.selectbox('Choose a Model', glob('models/tomo-*'))
        st.session_state.num_topics = st.slider('Number of topics', min_value=1, max_value=40, value=10)
        st.session_state.topics_reduced = st.checkbox('Topics reduced', value=False)
        numwords_per_topic = st.slider('Number of words per topic', min_value=5, max_value=50, value=20)

    # Prepare topic query request from the current input fields.
    request = json.dumps(PredictionRequest(
        query_string=st.session_state.query,
        num_topics=st.session_state.num_topics,
        topics_reduced=st.session_state.topics_reduced
    ).__dict__)

    # If query is empty, reset the session state.
    if st.session_state.query == '':
        del st.session_state.tomo_wordclouds
        st.session_state.tomo_wordclouds = []
        st.session_state.last_request = ''
    else:
        # Show header before wordclouds.
        st.subheader('Similar topics: (`% cosine similarity`)')

        # Send topic query and show response as wordclouds.
        numcols = 5
        cols = st.columns(numcols)
        if (st.session_state.last_request == request) and (st.session_state.numwords_per_topic == numwords_per_topic):

            # If parameters haven't changed, show the previous wordclouds.
            n = 0
            for (topic_num, topic_score, img) in st.session_state.tomo_wordclouds:
                n = display_topic_wordcloud(img, cols, numcols, n, topic_score, topic_num)

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
                topic_word_scores = dict(zip(r['topic_words'][:numwords_per_topic], softmax(r['word_scores'][:numwords_per_topic])))
                img = WordCloud(width=320, height=240, background_color=bgcolor).generate_from_frequencies(topic_word_scores).to_image()
                
                # Display wordcloud.
                n = display_topic_wordcloud(img, cols, numcols, n, r['topic_score'], r['topic_num'])

                # Save wordcloud to session state.
                st.session_state.tomo_wordclouds.append([r['topic_num'], r['topic_score'], img])

            # Save session state.
            st.session_state.last_request = request
            st.session_state.numwords_per_topic = numwords_per_topic
    
        # Top 10 Section, Title, Article and Time
