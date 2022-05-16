"""
@author: noobrainers
@contact: Bryan T. Kim bryantaekim at gmail dot com
          Daniel Lee dslee47 at gmail dot com
          Juswaldy Jusman juswaldy at gmail dot com
@description: Streamlit Hierarchical Clustering app file.
"""
import streamlit as st

def app():
    """ Hierarchical Clustering app """

    # Page settings.
    header = 'Hierarchical Clustering'

    # Page elements.
    with st.sidebar:
        st.subheader(header)
