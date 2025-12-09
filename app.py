# DEBUG LAUNCHER
import streamlit as st
from debug_wrapper import show_debug

st.session_state["debug"] = True
show_debug()
