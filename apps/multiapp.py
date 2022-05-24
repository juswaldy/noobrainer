# Based on https://github.com/harmkenn/python-stat-tools

"""Framework for running multiple Streamlit applications as a single app.
"""
import streamlit as st

class MultiApp:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """
    def __init__(self, logo):
        self.apps = []
        self.logo = logo

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):

        with st.sidebar:
            c1, c2 = st.columns([9, 19])
            with c1:
                st.image(self.logo, width=100)
            with c2:
                st.write('')
                st.write('<span style="font-size:40px; font-weight:bold">noobrainer</span>', unsafe_allow_html=True)
            with st.expander('About this app', expanded=False):
                st.write(
                    """
                    This is our GLG capstone project for [fourthbrain.ai](https://fourthbrain.ai) MLE Cohort 6, Automated Metadata Tagging and Topic Modeling.
                    Some'n some'n or other, watchamacallit.

                    Enjoy!

                    {bryantaekim, dslee47, juswaldy} at gmail dot com
                    """
                )
            app = st.radio(
                '',
                self.apps,
                format_func=lambda app: app['title'])

        app['function']()
