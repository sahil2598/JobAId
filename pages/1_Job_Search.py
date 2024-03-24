from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pyresparser import ResumeParser
import warnings
from serpapi import GoogleSearch
import os
import json
import streamlit as st
import pandas as pd
from io import StringIO
from langchain_openai import ChatOpenAI

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(layout='centered')

# get the match rating between the resume and the job posting
def get_rating(x, a=0.2, b=0.6, c=10, d=0):
    x = 0.2 if x < 0.2 else x
    x = 0.5 if x > 0.5 else x
    return round(((x - a) / (b - a)) * (d - c) + c, 2)

# create link to generate the cover letter
def get_cover_letter_link(applicant_info, file_name, job_description):
    cover_letter_link = f'http://localhost:8501/Cover_Letter?applicant_info={applicant_info}&resume_file={file_name}&job_description={job_description}'
    return cover_letter_link

if 'google_jobs_link' not in st.session_state:
    st.session_state.google_jobs_link = ''

# get job posting relevant to the query
@st.cache_resource
def get_suggestions(query, file_name):
    # Use pyresparser (uses Spacy NLP library)
    data = ResumeParser('docs/' + file_name).get_extracted_data()
    text_file_name = file_name[:-4] + '.txt'
    # write parsed details to a text file
    file = open(text_file_name, 'w+')
    applicant_info = ''
    for key in data.keys():
        # ignore personal details while writing to txt file
        if key not in ['name', 'email', 'mobile_number', 'no_of_pages']:
            file.write(key + ' : ')
            if data[key] is not None:
                if type(data[key]) is list:
                    for d in data[key]:
                        file.write(str(d) + '; ')
                else:
                    file.write(str(data[key]) + '; ')
            file.write('\n')
        elif key != 'no_of_pages':
            # include personal details in personal information
            applicant_info += key + ': ' + data[key] + '\n'

    file.close()
    # load the text file
    loader = TextLoader(text_file_name)
    resume = loader.load()
    # extract embeddings from the text file
    docembeddings = FAISS.from_documents(resume, OpenAIEmbeddings())
    docembeddings.save_local("llm_faiss_index")
    docembeddings = FAISS.load_local("llm_faiss_index",OpenAIEmbeddings())

    # parameters for the Google Search API
    params = {
        'api_key': os.environ['SERPAPI_API_KEY'],
        'uule': 'w+CAIQICINVW5pdGVkIFN0YXRlcw',		# encoded location (USA)
        'q': query,              		            # search query
        'hl': 'en',                         		# language of the search
        'gl': 'us',                         		# country of the search
        'engine': 'google_jobs',					# SerpApi search engine
        'start': 0									# pagination
    }

    # search for jobs according to query
    search = GoogleSearch(params)
    result_dict = search.get_dict()
    if 'error' in result_dict:
        st.write('Could not generate jobs')
    else:
        google_jobs_results = []
        for result in result_dict['jobs_results']:
            google_jobs_results.append(result)

        recommendations = []
        for job in google_jobs_results:
            job_description = ""
            # extract important aspects of the job listing
            for attr in job['job_highlights']:
                for item in attr['items']:
                    job_description += item + '\n'

            # get similarity score between resume and job posting
            relevant_text_with_score = docembeddings.similarity_search_with_score(job_description, k=1)

            # extract company and google jobs links
            link = job['related_links'][0]['link']
            st.session_state.google_jobs_link = result_dict['search_metadata']['google_jobs_url']
            
            cover_job_description = ''
            # create job description for cover letter generation
            for key in ['title', 'company_name', 'location', 'description']:
                cover_job_description += key + ': ' + job[key] + '\n'
            
            cover_job_description += job_description
            cover_link = get_cover_letter_link(applicant_info, 'docs/' + file_name, cover_job_description)

            job_details = {
                'Match Score': get_rating(relevant_text_with_score[0][1]),
                'Company Name': job['company_name'],
                'Role': job['title'],
                'Location': job['location'],
                'Company Website': link,
                'Cover Letter': cover_link
            }
            recommendations.append(job_details)

    return recommendations


st.title('Job Search')
uploaded_file = st.file_uploader('Upload Your Resume')
query = st.text_input('Enter Job Title')
if st.button('Search'):
    if uploaded_file is not None and query is not None:
        # read uploaded file into string
        bytes_data = uploaded_file.getvalue()
        stringio = StringIO(uploaded_file.getvalue().decode("latin-1"))
        string_data = stringio.read()
        with open(os.path.join('./docs', uploaded_file.name), 'wb') as f:
            # write resume to local disk
            f.write(uploaded_file.getbuffer())
            recommendations = get_suggestions(query, uploaded_file.name)
            df = pd.DataFrame(recommendations)
            st.dataframe(
                df,
                column_config={
                    # for columns with links
                    'Company Website': st.column_config.LinkColumn(display_text='Link'),
                    'Cover Letter': st.column_config.LinkColumn(display_text='Generate Cover Letter')  
                },
                use_container_width=True,
                hide_index=True
            )
            st.write(f'[View in Google Jobs]({st.session_state.google_jobs_link})')