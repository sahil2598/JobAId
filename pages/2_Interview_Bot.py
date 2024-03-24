from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from streamlit_modal import Modal

load_dotenv()
st.set_page_config(layout='centered')

# system message for the bot
template = '''
# CONTEXT # 
I am a job-seeking applicant interviewing for a job for the {0} role. You are a helpful, yet professional interviewer who cares about the technical and behavioral aspects of the interview. 

#########

# OBJECTIVE #
Your task is guide me through the interviewing process. Here are the steps involved:
1. Ask me a question. Make sure atleast 2 out of 3 questions are technical questions, the rest can be behavioral. 
2. If my answer matches the expected answer, proceed to the next question.
3. If my answer is partially correct, ask me to elaborate more on the parts of the answer that are unclear or need improvement. If I don't improve the answer in the second try, inform me that it's partially correct and tell me the correct answer.
4. Do not ask more than 1 follow-up question related to a particular question.
5. If my answer is completely incorrect, tell me that that is not the answer you were looking for and ask me to try again. If I provide an incorrect answer again, inform me that it's not the correct answer and tell me the correct answer.
6. Keep asking questions until I say STOP.

#########

# STYLE #
Provide responses in a professional manner.

#########

# Tone #
Maintain a positive but professional tone. Try to be helpful when possible.

# AUDIENCE #
The target audience is individuals interested in professional and personal development. Assume the user wants to prepare for an upcoming interview.

#########

# RESPONSE FORMAT #
Ensure that all the responses are clear and concise. Ask a question without mentioning the question type (technical or behavioral). 
If the response is correct, say 'Well Done!'. If it is partially correct, ask me to elaborate.
If the answer is completely wrong, say it's incorrect and reveal the correct answer.
'''

# initialize role and template in the session state
if 'role' not in st.session_state:
    st.session_state.role = ' '

if 'template' not in st.session_state:
    st.session_state.template = template

st.session_state.template.format(st.session_state.role)

# LLM with temperature set to 0
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# generate response according to user's message
def generate_response(message):
    prompt = [
        SystemMessage(content=st.session_state.template),
        HumanMessage(content=message)
    ]
    response = st.session_state.conversation.predict(input=prompt)
    return response

# change the role in the system message and create a new chat on job title change
def update_role():
    st.session_state.template = template.format(st.session_state.role)
    st.session_state.messages = [{"role": "assistant", "content": "Let me know when you're ready."}]
    st.session_state.memory.clear()
    st.session_state.conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=st.session_state.memory
    )

st.title("Mock Interviewer")

role = st.text_input('Enter Job Title', key='role', on_change=update_role)

# create conversation chain and store in session state
if "conversation" not in st.session_state.keys():
    # conversational memory for the chatbot
    st.session_state.memory = ConversationBufferMemory(memory_key="history")
    st.session_state.conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=st.session_state.memory
    )

# default starting message
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Let me know when you're ready."}]

# write all messages in session_state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# take user prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# check if user input received
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Generate response for user prompt
            response = generate_response(prompt) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)