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

def show_topic_detail(topic_num, container):
    """ Show topic details on the container """
    applystyle_button(topic_num, 40)
    with container:
        st.write('# Topic #{}'.format(topic_num))


def display_topic_wordcloud_clickable(img, cols, numcols, n, topic, container):
    """ Display clickable topic wordcloud """
    with cols[n].container():
        cols[n].button('Topic #{} ({:.1%})'.format(topic['topic_num'], topic['topic_score']),
            key=topic['topic_num'],
            on_click=show_topic_detail,
            kwargs={'topic_num': topic['topic_num'], 'container': container})
        cols[n].image(img, use_column_width=True)
    n += 1
    if n >= numcols:
        n = 0
    return n

def applystyle_button(topic_num, n_buttons):
    def get_button_indices(button_ix):
        return {
            'nth_child': button_ix,
            'nth_last_child': n_buttons - button_ix + 1
        }

    clicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        border-color: rgb(255, 75, 75);
        color: rgb(255, 75, 75);
        box-shadow: rgba(255, 75, 75, 0.5) 0px 0px 0px 0.2rem;
        outline: currentcolor none medium;
    }
    """
    unclicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        pointer-events: none;
        cursor: not-allowed;
        opacity: 0.65;
        filter: alpha(opacity=65);
        -webkit-box-shadow: none;
        box-shadow: none;
    }
    """
    style = ""
    for ix in range(n_buttons):
        ix += 1
        if ix == topic_num:
            style += clicked_style % get_button_indices(ix)
        else:
            style += unclicked_style % get_button_indices(ix)
    st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


def app():
    """ Topic Modeling app """

    # Page settings.
    header = 'Topic Modeling'

    # Page main area.
    with st.form(key='tomo_form'):
        query = st.text_area(
            label='',
            height=200,
            max_chars=1000,
            help='Tell us what you are thinking',
            placeholder='Enter your request/report here...')
        submitted = st.form_submit_button(label='Submit')

    if submitted:
        st.session_state.query = query

    # Page side bar.
    with st.sidebar:
        st.subheader(header)
        st.session_state.tomo_model_path = st.selectbox('Choose a Model', glob('models/tomo-*'))
        num_topics = st.session_state.num_topics = st.slider('Number of topics', min_value=1, max_value=40, value=10)
        topics_reduced = st.session_state.topics_reduced = st.checkbox('Topics reduced', value=False)
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
        st.subheader('Similar topics: (% cosine similarity)')

        result_container = st.expander('Results', expanded=True)

        # Send topic query and show response as wordclouds.
        # If query/num_topics/reduced hasn't changed, use the stored response.
        numcols = 5
        cols = st.columns(numcols)
        if not (submitted or ((st.session_state.query == query) and (st.session_state.num_topics == num_topics) and (st.session_state.topics_reduced == topics_reduced))):

            # If parameters haven't changed, show the previous wordclouds.
            n = 0
            #for (topic, img) in st.session_state.tomo_wordclouds:
            #    n = display_topic_wordcloud_clickable(img, cols, numcols, n, topic, result_container)

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
                n = display_topic_wordcloud_clickable(img, cols, numcols, n, r, result_container)

                # Save wordcloud to session state.
                st.session_state.tomo_wordclouds.append([r, img])

            # Save session state.
            st.session_state.last_request = request
            st.session_state.numwords_per_topic = numwords_per_topic
    
        # Top 10 Section, Title, Article and Time
