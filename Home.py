import streamlit as st

st.set_page_config(layout="wide")
st.title('JobAId')
st.text("")
st.text("")
st.page_link(page='http://localhost:8501/Job_Search', label='Job Search', icon='👀')
st.text("")
st.page_link(page='http://localhost:8501/Interview_Bot', label='Interview Bot', icon='💬')
st.text("")
st.page_link(page='http://localhost:8501/Cover_Letter', label='Cover Letter Generator', icon='📝')