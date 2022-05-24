"""
@author: noobrainers
@contact: Bryan T. Kim bryantaekim at gmail dot com
          Daniel Lee dslee47 at gmail dot com
          Juswaldy Jusman juswaldy at gmail dot com
@description: Streamlit Topic Modeling app file.
"""
from tkinter import W
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

        # Read in topic specialists.
        sme = pd.read_csv(st.session_state.sme_filepath)

        # Get top 3 specialists.
        sme['label'] = sme['jobtitle'].apply(lambda x: x.lower()[0])
        top3_sme = sme[sme.label==topic_labels[num]].sample(3)

        # Show topic specialists.
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

    # Site settings.
    api_url = st.session_state.api_url

    # Page settings.
    header = st.session_state.tomo_header

    # Page side bar.
    with st.sidebar:
        st.subheader(header)
        available_models = {k: v for k, v in enumerate(model_about_values.keys())}
        # If debugging, show all models.
        if st.session_state.debug:
            tomo_model = st.selectbox('Choose a Model', options=available_models.keys(), format_func=lambda x: available_models[x], index=st.session_state.tomo_model)
        else:
            tomo_model = st.session_state.tomo_model
        num_topics = st.slider('Number of topics', min_value=1, max_value=40, value=10)
        # topics_reduced = st.session_state.topics_reduced = st.checkbox('Topics reduced', value=False)
        numwords_per_topic = st.slider('Number of words per topic', min_value=5, max_value=50, value=20)

    # If a different model is selected, request a model refresh.
    model_num_topics = model_about_values[available_models[tomo_model]][7]
    if st.session_state.debug and st.session_state.tomo_model != tomo_model:
        request = json.dumps(ModelRefresh(
            client_id=str(uuid.uuid4()),
            model_path=available_models[tomo_model]
        ).__dict__)
        response = requests.post(f'{api_url}/tomo/model/refresh', data=request)
        response = requests.get(f'{api_url}/tomo/topics/number')
        model_num_topics = str(response.json()['num_topics'])
        st.session_state.tomo_model = tomo_model

        # Reset query and top10.
        st.session_state.query = ''
        del st.session_state.tomo_top10
        st.session_state.tomo_top10 = []

    # Page header.
    st.header(header)
    st.subheader(f'# of reports: {model_about_values[available_models[st.session_state.tomo_model]][5]}')
    top10_container = st.container()
    result_container = st.expander('Results', expanded=True)

    # Show summary of the selected model.
    with st.expander(f'About this model ({available_models[st.session_state.tomo_model]})', expanded=False):
        model_about = pd.DataFrame(model_about_values[available_models[st.session_state.tomo_model]], columns=['Value'], index=model_about_fields)
        model_about.at['Number of topics discovered', 'Value'] = model_num_topics
        st.dataframe(model_about)

    # Top 10 area.
    with top10_container:
        numcols = 5
        cols = st.columns(numcols)
        n = 0
        if len(st.session_state.tomo_top10) == 0 or st.session_state.num_topics != num_topics or st.session_state.numwords_per_topic != numwords_per_topic:

            del st.session_state.tomo_top10
            st.session_state.tomo_top10 = []

            response = requests.get(f'{api_url}/tomo/topics/get-topics', params={'num_topics': num_topics})

            # Show response as wordclouds.
            bgcolor = random.choice(wordcloud_backgrounds)
            for r in response.json():
                # Generate wordcloud from words/scores.
                topic_word_scores = dict(zip(r['topic_words'][:numwords_per_topic], softmax(r['word_scores'][:numwords_per_topic])))
                img = WordCloud(width=320, height=240, background_color=bgcolor).generate_from_frequencies(topic_word_scores).to_image()
                
                if st.session_state.debug:
                    st.write(f"{r['topic_num']} - {' '.join(r['topic_words'][:20])}")

                # Ensure a topic_score: either an actual float score from the server, or the index+1.
                if 'topic_score' not in r:
                    r['topic_score'] = n + 1

                # Display wordcloud.
                n = display_topic_wordcloud_clickable(img, cols, numcols, n, r, result_container)

                # Save settings and wordclouds to session state.
                st.session_state.tomo_top10.append([r, img])
                st.session_state.num_topics = num_topics
                st.session_state.numwords_per_topic = numwords_per_topic

        else:
            for (topic, img) in st.session_state.tomo_top10:
                n = display_topic_wordcloud_clickable(img, cols, numcols, n, topic, result_container)
