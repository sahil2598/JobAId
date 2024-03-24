import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
import pyperclip

load_dotenv()
st.set_page_config(layout='centered')

model = ChatOpenAI(model_name='gpt-4')
# system message template
template = '''
You are a cover letter generator who generates a cover letter based on job applicant information and the job description.
The pre-requisites for the cover letter include:
1. The cover letter starts with the applicant address, email ID, phone number, date along with the hiring manager company and address. These details should be already be filled out according to the provided prompt.
2. Include a greeting for the hiring manager.
3. Contains only 3 paragraphs.
4. In the first paragraph, the applicant displays their interest in the role and company.
5. The second paragraph contains a brief overview of the applicant's background, including key achievements, skills and specialties that make them suited for the role. Show measurable impact.
6. The third paragraph elaborates on experiences with building qualitative skills such as teamwork and leadership.
7. Finish with a strong conclusion.
7. Contains a signature line at the end, including a closing and the name of the applicant.
'''
    
st.title('Cover Letter Generator')
def generate_cover_letter(applicant_details, job_description, resume=None):
    # if resume has been uploaded
    if resume:
        prompt = ChatPromptTemplate.from_messages([("system", template), ("human", 'Applicant Information:\n{applicant_details}\n\nApplicant Resume:\n{resume}\n\n Job Description: \n{job_details}')])
        chain = prompt | model  
        cover_letter = chain.invoke({'applicant_details': applicant_details, 'resume': resume, 'job_details': job_description}).content
    else:
        prompt = ChatPromptTemplate.from_messages([("system", template), ("human", 'Applicant Information:\n{applicant_details}\n\nJob Description: \n{job_details}')])
        chain = prompt | model  
        cover_letter = chain.invoke({'applicant_details': applicant_details, 'job_details': job_description}).content
    return cover_letter

params = ['resume_file', 'applicant_info', 'job_description']
if all(param in st.query_params for param in params):
    # load resume
    loader = PyPDFLoader(st.query_params['resume_file'])
    resume = loader.load()[0].page_content.replace('\n', ';')
    applicant_details = st.query_params['applicant_info']
    job_description = st.query_params['job_description']
    cover_letter = generate_cover_letter(applicant_details, job_description, resume)
    st.success("Text copied to clipboard!")
    # copy resume to clipboard (does not work on the hosted website)
    pyperclip.copy(cover_letter)
    st.write(cover_letter)
else:
    # generate cover letter based on custom input
    applicant_info = st.text_input('Enter Applicant Information')
    job_description = st.text_input('Enter Job Description')
    if st.button('Generate'):
        cover_letter = generate_cover_letter(applicant_info, job_description)
        st.success("Text copied to clipboard!")
        pyperclip.copy(cover_letter)
        st.write(cover_letter)