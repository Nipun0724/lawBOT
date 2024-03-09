import streamlit as st
import replicate
import os
import pandas as pd

st.set_page_config(page_title=" Llama 2 Chatbot (HR Interview)")

with st.sidebar:
    st.title(' Llama 2 Chatbot')
    st.write('This chatbot simulates HR interviews using the open-source Llama 2 LLM model.')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👌')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
    if selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
    st.markdown(' Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

csv_file = st.sidebar.file_uploader("Upload CSV file with skills (first column as skill name)", type="csv")

if csv_file is not None:
    skills_df = pd.read_csv(csv_file)
    skills_list = skills_df["skill name"].tolist()  
else:
    skills_list = []  


if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to your HR interview simulation. How may I assist you today?"}]

# 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to your HR interview simulation. How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response with skill-based questions
def generate_llama2_response(prompt_input):
  """Generates a response from the LLM model, potentially including HR interview questions.

  Args:
      prompt_input: The user's prompt to continue the conversation.

  Returns:
      A string containing the LLM's generated response.
  """

  # Build the conversation history, incorporating skills if available
  conversation = "You are an HR interviewer evaluating a candidate. "
  if skills_list:  # Check if skills list is populated from uploaded CSV
      conversation += f"The candidate's skills include: {', '.join(skills_list)}.\n\n"

  for message in st.session_state.messages:
      conversation += f"{message['role']}: {message['content']}\n\n"

  # Formulate the prompt for the LLM, including temperature, top_p, max_length, etc.
  output = replicate.run(llm,
                          input={
                              "prompt": f"{conversation}{prompt_input} Interviewer: ",
                              "temperature": temperature,
                              "top_p": top_p,
                              "max_length": max_length,
                              "repetition_penalty": 1
                          })

  # Process and return the LLM's response
  full_response = ''.join(output)
  return full_response

# Add an input field for user messages
user_input = st.text_input("Type your message here:")

# Check if the user has inputted a message
if user_input:
    # Generate response from Llama 2 model
    llama2_response = generate_llama2_response(user_input)
    
    # Append user message and Llama 2 response to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": llama2_response})
