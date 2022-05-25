import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import requests
import json
import sys
sys.path.append('../')
from models.schemas import *
from typing import Dict

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
        st.write('')
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write('# :boy:')
            st.write('ehlo')
        c2.write('# :girl:')
        c3.write('# :male-farmer:')


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

def get_topics_nerlabels() -> Dict[int, str]:
    """ Get topic nerlabel """

    api_url = 'http://localhost:8000'
    num_topics = 40
    results = {}

    # Get the top 40 topics.
    response = requests.get(f'{api_url}/tomo/topics/get-topics', params={'num_topics': num_topics})

    # Go through each topic and request a class label from the NER backend.
    for topic in response.json():
        request = json.dumps(Classification(
            query_string=' '.join(topic['topic_words']),
            class_num=-1,
            class_str=''
        ).__dict__)
        response = requests.post(f'{api_url}/ner/classify', data=request).json()
        results[topic['topic_num']] = response['class_str']

    return results
