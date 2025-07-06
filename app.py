import os
import sys
import streamlit as st

# Add the model directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "model")))

# Import route pages
from pages._pages import home
from pages._pages import about
from pages._pages import github
from pages._pages import try_it

# Define page routes
routes = {
    "Home": home.main,
    "Try it out": try_it.main,
    "About": about.main,
    "GitHub": github.main,
}

# Set up Streamlit configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon=":brain:",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/sangeeth-1243/brain-tumor-detection",
        "Report a bug": "https://github.com/sangeeth-1243/brain-tumor-detection/issues",
        "About": "Detecting brain tumors using *deep Convolutional Neural Networks*",
    },
    initial_sidebar_state="collapsed",
)

# Hide sidebar toggle and customize select box font
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none;
    }
    [data-testid="stSelectbox"] .st-emotion-cache-13bfgw8 p {
        font-size: 24px;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Dropdown menu for page selection
def format_func(page):
    return page[0]

page = st.selectbox(
    "Menu",
    list(routes.items()),
    index=0,
    format_func=format_func,
)

# Load selected page
page[1]()
