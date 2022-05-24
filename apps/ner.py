"""
@author: noobrainers
@contact: Bryan T. Kim bryantaekim at gmail dot com
          Daniel Lee dslee47 at gmail dot com
          Juswaldy Jusman juswaldy at gmail dot com
@description: Streamlit Named Entity Recognition app file.
"""
import streamlit as st
import pandas as pd
from wordcloud import WordCloud
from scipy.special import softmax
import random
import sys
sys.path.append('../')
import json
import uuid
import requests
import random
from models.schemas import *
from utils.ner import *
from apps.configs import *
from apps.helpers import *

def app():
    """ Named Entity Recognition app """

    # Site settings.
    api_url = st.session_state.api_url

    # Page settings.
    header = st.session_state.ner_header

    # Read in topic specialists.
    sme = pd.read_csv(st.session_state.sme_filepath)

    # Page side bar.
    with st.sidebar:
        st.subheader(header)
        available_models = {k: v for k, v in enumerate(model_about_values.keys())}
        
        # If debugging, show all models.
        if st.session_state.debug:
            ner_model = st.selectbox('Choose a Model', options=available_models.keys(), format_func=lambda x: available_models[x], index=st.session_state.ner_model)
        else:
            ner_model = 0
        num_topics = st.session_state.num_topics = st.slider('Number of topics', min_value=1, max_value=40, value=10)
        topics_reduced = st.session_state.topics_reduced = st.checkbox('Topics reduced', value=False)
        numwords_per_topic = st.slider('Number of words per topic', min_value=5, max_value=50, value=20)

    # If a different model is selected, request a model refresh.
    model_num_topics = model_about_values[available_models[ner_model]][7]
    if st.session_state.ner_model != ner_model:
        request = json.dumps(ModelRefresh(
            client_id=str(uuid.uuid4()),
            model_path=available_models[ner_model]
        ).__dict__)
        response = requests.post(f'{api_url}/ner/model/refresh', data=request)
        response = requests.get(f'{api_url}/ner/topics/number')
        model_num_topics = str(response.json()['num_topics'])
        st.session_state.ner_model = ner_model

        # Reset query and ner_model.
        st.session_state.query = ''
        st.session_state.ner_model = 0

    # Page header.
    st.header('Test Model')

    # Query area.
    with st.form(key='ner_form'):
        query = st.text_area(
            label='Test User Report',
            height=200,
            max_chars=1000,
            help='Tell us what you are thinking',
            placeholder='Enter your request/report here...')
        submitted = st.form_submit_button(label='Submit')

    if submitted:
        st.session_state.query = query

        # Prepare ner classification request from the query.
        request = json.dumps(Classification(
            query_string=st.session_state.query,
            class_num=-1,
            class_str=''
        ).__dict__)

        # Send request to the server and get the response.
        response = requests.post(f'{api_url}/ner/classify', data=request).json()

        ner_label = response['class_str'][0]
        sme['label'] = sme['jobtitle'].apply(lambda x: x.lower()[0])
        top3_sme = sme[sme.label==ner_label][:3]

        st.subheader(top3_sme.jobtitle.values[0])

        cols = st.columns(3)
        for i in range(3):
            x = top3_sme.iloc[i]
            c1, c2 = st.columns([1, 4])
            with c1:
                cols[i].write(f'# {x.emoji}')
            with c2:
                cols[i].write(f'{x.firstname} {x.lastname}')
                cols[i].write(f'{x.email}')
                cols[i].write(f'{x.phone}')

    result_container = st.expander('Results', expanded=True)

    st.header('Update Model')
    with st.form(key='model_update_form'):
        c1, c2 = st.columns([1, 4])
        c1.text_input('New Label')
        c2.file_uploader("Choose a CSV file", accept_multiple_files=True)
        st.text_area('Keywords for New Label', height=200, max_chars=1000, placeholder='Enter additional keywords here, separated by commas')
        st.form_submit_button(label='Update Model')

    ## Prepare topic query request from the query.
    #request = json.dumps(PredictionRequest(
    #    query_string=st.session_state.query,
    #    num_topics=st.session_state.num_topics,
    #    topics_reduced=st.session_state.topics_reduced
    #).__dict__)

    ## If query is empty, reset the session state.
    #if st.session_state.query == '':
    #    del st.session_state.ner_wordclouds
    #    st.session_state.ner_wordclouds = []
    #    st.session_state.last_request = ''
    #else:
    #    # Show header before wordclouds.
    #    st.subheader('Similar topics: (% cosine similarity)')

    #    result_container = st.container()

    #    # Send topic query and show response as wordclouds.
    #    # If query/num_topics/reduced hasn't changed, use the stored response.
    #    with result_container:
    #        numcols = 5
    #        cols = st.columns(numcols)
    #        n = 0
    #        if (submitted or ((st.session_state.query == query) and (st.session_state.num_topics == num_topics) and (st.session_state.topics_reduced == topics_reduced))):
    #            del st.session_state.ner_wordclouds
    #            st.session_state.ner_wordclouds = []

    #            # Send request to the server and get the response.
    #            response = requests.post(f'{api_url}/tomo/topics/query', data=request)

    #            # Show response as wordclouds.
    #            n = 0
    #            bgcolor = random.choice(wordcloud_backgrounds)
    #            for r in response.json():
    #                # Generate wordcloud from words/scores.
    #                topic_word_scores = dict(zip(r['topic_words'][:numwords_per_topic], softmax(r['word_scores'][:numwords_per_topic])))
    #                img = WordCloud(width=320, height=240, background_color=bgcolor).generate_from_frequencies(topic_word_scores).to_image()
    #                
    #                # Display wordcloud.
    #                n = display_topic_wordcloud_clickable(img, cols, numcols, n, r, result_container)

    #                # Save wordcloud to session state.
    #                st.session_state.ner_wordclouds.append([r, img])

    #            # Save session state.
    #            st.session_state.last_request = request
    #            st.session_state.numwords_per_topic = numwords_per_topic
    #        else:
    #            # If parameters haven't changed, show the previous wordclouds.
    #            for (topic, img) in st.session_state.ner_wordclouds:
    #                n = display_topic_wordcloud_clickable(img, cols, numcols, n, topic, result_container)