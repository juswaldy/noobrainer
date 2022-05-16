"""
@author: noobrainers
@contact: Bryan T. Kim bryantaekim at gmail dot com
          Daniel Lee dslee47 at gmail dot com
          Juswaldy Jusman juswaldy at gmail dot com
@description: Streamlit Named Entity Recognition app file.
"""
import streamlit as st

def app():
    """ Named Entity Recognition app """

    # Page settings.
    header = 'Named Entity Recognition'

    # Page elements.
    with st.sidebar:
        st.subheader(header)
