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
import pandas as pd
import uuid
from apps.configs import *
import matplotlib.pyplot as plt

api_url = 'http://localhost:8000'

def display_topic_wordcloud(img, cols, numcols, n, topic_score):
    """ Display topic wordcloud """
    with cols[n].container():
        cols[n].write('`{:.1%}`'.format(topic_score))
        cols[n].image(img, use_column_width=True)
    n += 1
    if n >= numcols:
        n = 0
    return n

def get_topic_header(topic: object) -> str:
    """ Show topic header """
    # If topic_score is a float, then display percentage.
    # Otherwise it's just an index for top 10, so display the index+1 (starting at 1).
    return 'Topic #{} ({:.1%})'.format(topic['topic_num'], topic['topic_score']) if type(topic['topic_score']) == float else f'Topic #{topic["topic_num"]+1}'

def show_topic_detail(topic: object, container: object) -> None:
    """ Show topic details in the provided container """
    num, score, words, word_scores = topic['topic_num'], topic['topic_score'], topic['topic_words'], topic['word_scores']
    applystyle_button(topic['topic_num'], 40)
    with container:
        st.header(get_topic_header(topic))
        df = pd.DataFrame({'word': words, 'score': word_scores})
        df = df.groupby('word')[['score']].sum()
        df = df.sort_values(by='word')
        st.bar_chart(df)

def display_topic_wordcloud_clickable(img, cols, numcols, n, topic, container) -> int:
    """ Display clickable topic wordcloud """
    with cols[n].container():
        topic_header = get_topic_header(topic)

        # Make the button.
        cols[n].button(topic_header,
            key=topic['topic_num'],
            on_click=show_topic_detail,
            kwargs={'topic': topic, 'container': container})
        
        # Display the wordcloud.
        cols[n].image(img, use_column_width=True)

    # Increment the item index.
    n += 1
    if n >= numcols:
        n = 0

    # Return item index.
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
    header = 'Highest Ranking Topics'

    # Page side bar.
    with st.sidebar:
        st.subheader(header)
        tomo_model = st.selectbox('Choose a Model', sorted(glob('models/tomo-*')))
        num_topics = st.session_state.num_topics = st.slider('Number of topics', min_value=1, max_value=40, value=10)
        topics_reduced = st.session_state.topics_reduced = st.checkbox('Topics reduced', value=False)
        numwords_per_topic = st.slider('Number of words per topic', min_value=5, max_value=50, value=20)

    # If a different model is selected, request a model refresh.
    model_num_topics = model_about_values[tomo_model][7]
    if st.session_state.tomo_model != tomo_model:
        request = json.dumps(ModelRefresh(
            client_id=str(uuid.uuid4()),
            model_path=tomo_model
        ).__dict__)
        response = requests.post(f'{api_url}/tomo/model/refresh', data=request)
        st.session_state.tomo_model = tomo_model
        response = requests.get(f'{api_url}/tomo/topics/number')
        model_num_topics = str(response.json()['num_topics'])

        # Reset query and top10.
        st.session_state.query = ''
        st.session_state.tomo_top10 = []

    # Show summary of the selected model.
    with st.expander(f'About this model ({st.session_state.tomo_model})', expanded=False):
        model_about = pd.DataFrame(model_about_values[st.session_state.tomo_model], columns=['Value'], index=model_about_fields)
        model_about.at['Number of topics discovered', 'Value'] = model_num_topics
        st.dataframe(model_about)

    # Page header.
    st.title(header)
    st.header(f'Number of user submitted reports: {model_about_values[st.session_state.tomo_model][5]}')
    top10_container = st.expander('Top 10', expanded=True)

    # Get the model's top 10 topics.
    with top10_container:
        numcols = 5
        cols = st.columns(numcols)
        n = 0
        if len(st.session_state.tomo_top10) == 0:
            # Request top 10 topics.
            response = requests.get(f'{api_url}/tomo/topics/get-topics', params={'num_topics': 10})

            # Show response as wordclouds.
            bgcolor = random.choice(wordcloud_backgrounds)
            for r in response.json():
                # Generate wordcloud from words/scores.
                topic_word_scores = dict(zip(r['topic_words'][:numwords_per_topic], softmax(r['word_scores'][:numwords_per_topic])))
                img = WordCloud(width=320, height=240, background_color=bgcolor).generate_from_frequencies(topic_word_scores).to_image()
                
                # Ensure a topic_score: either an actual float score from the server, or the index+1.
                if 'topic_score' not in r:
                    r['topic_score'] = n + 1

                # Display wordcloud.
                n = display_topic_wordcloud_clickable(img, cols, numcols, n, r, top10_container)

                # Save wordcloud to session state.
                st.session_state.tomo_top10.append([r, img])

        else:
            for (topic, img) in st.session_state.tomo_top10:
                n = display_topic_wordcloud_clickable(img, cols, numcols, n, topic, top10_container)

    # Query area.
    with st.expander('Query', expanded=False):

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

        # Prepare ner classification request from the query.
        request = json.dumps(PredictionRequest(
            query_string=st.session_state.query,
            num_topics=st.session_state.num_topics,
            topics_reduced=st.session_state.topics_reduced
        ).__dict__)

        # Prepare topic query request from the query.
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

            result_container = st.container()

            # Send topic query and show response as wordclouds.
            # If query/num_topics/reduced hasn't changed, use the stored response.
            numcols = 5
            cols = st.columns(numcols)
            if not (submitted or ((st.session_state.query == query) and (st.session_state.num_topics == num_topics) and (st.session_state.topics_reduced == topics_reduced))):

                # If parameters haven't changed, show the previous wordclouds.
                n = 0
                for (topic, img) in st.session_state.tomo_wordclouds:
                    n = display_topic_wordcloud_clickable(img, cols, numcols, n, topic, result_container)

            else:
                del st.session_state.tomo_wordclouds
                st.session_state.tomo_wordclouds = []

                # Send request to the server and get the response.
                response = requests.post(f'{api_url}/tomo/topics/query', data=request)

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
