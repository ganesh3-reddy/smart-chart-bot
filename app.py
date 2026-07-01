
## Entire RAG Chat Bot by Uploading PDF

import os
from google import genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
import chromadb
from google.genai import types

# load the api key and make client connection

api_key = st.secrets["AIzaSyBp2cP4B7LgiqeqYlk6eKvH-HzCbDovbLI"]
client = genai.Client(api_key=api_key)


# Initialize chromaDB(by using chache.resource we declare it here and use in entire app)
@st.cache_resource
def get_chroma_client():
    return chromadb.Client()

chroma_client=get_chroma_client()

# chunk function
def chunk_text(text):
    chunk_size = 500
    overlap = 50
    chunks = []

    if not text:
        return chunks

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)

    return chunks

# Page config( How you want the page look like)

st.set_page_config(page_title="FAQ BOT", page_icon="🤖",layout='wide')
st.title('🤖 FAQ BOT')
st.caption('upload pdf  and ask questions',width='stretch')

# slidebar for pdf uploads and stats
with st.sidebar:
    st.header("Document")
    file_uploaded = st.file_uploader("upload pdf",type='pdf')

    st.divider()
    st.header("📊 Token Stats")

    if "total_input_token" not in st.session_state:
        st.session_state.total_input_token = 0
    if "total_output_token" not in st.session_state:
        st.session_state.total_output_token =0
    col1, col2 = st.columns(2)
    col1.metric("Input Tokens",st.session_state.total_input_token)
    col2.metric("output Token",st.session_state.total_output_token)

    st.metric("Total Tokens", st.session_state.total_input_token+st.session_state.total_output_token)

    if st.button('reset'):
        st.session_state.total_input_token = 0
        st.session_state.total_output_token =0
        st.rerun()
# If pdf uploaded the process

if file_uploaded:
    # check the file is new or exsisting file
    if "current_file" not in st.session_state or st.session_state.current_file != file_uploaded.name:
        with st.spinner("processing file...."):

            # read the file
            fulltext = ""
            reader = PdfReader(file_uploaded)
            for page in reader.pages:
                fulltext+=page.extract_text()+ "\n"
            
            chunks= chunk_text(fulltext)

            # create new collection

            collection_name = f'doc_{hash(file_uploaded.name)% 10000}'

            # delete if the collection name already exsits
            try:
                chroma_client.delete_collection(collection_name)
            except:
                pass
            collection = chroma_client.create_collection(collection_name)

            # index the chunks
            for i,chunk in enumerate(chunks):
                embedding = client.models.embed_content(
                    model='gemini-embedding-001',
                    contents=chunk
                ).embeddings[0].values

                collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    ids = [f'doc{i}']
                )
            #save to session state
            st.session_state.current_file =file_uploaded.name
            st.session_state.collection_name = collection_name
            st.session_state.num_chunks = len(chunks)
            # reset the chat 
            st.session_state.messages = []
            st.session_state.chat_history = []
        st.success(f'✅ indexed chunks {len(chunks)} from pages is {len(reader.pages)}')


## initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

## Display the messages

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

        # show sources for assistant messages
        if message['role'] == 'assistant' and 'sources' in message:
            with st.expander("sources used",expanded=False):
                for i, source in enumerate(message['sources']):
                    st.markdown(f'chunks {i+1}')
                    st.text(source[:500] + "..." if len(source) > 500 else source)
                    if i < len(message['sources']) - 1:
                        st.divider()

# Chat input
if file_uploaded and "collection_name" in st.session_state:
    if prompt := st.chat_input(" Ask a question"):
        # add user  message to the UI
        st.session_state.messages.append({'role':'user','content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        
        #get collection
        collection = chroma_client.get_collection(st.session_state.collection_name)

        #embed query
        query_emb = client.models.embed_content(
            model='gemini-embedding-001',
            contents=prompt
        ).embeddings[0].values

        #get relevant chunks
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=3
        )
        context = "\n\n".join(results['documents'][0])

        # system instructions
        system_instruction = f'''You are a helpful assistant answering questions about a document.
Answer based ONLY on the context below. If the answer isn't in the context, say "I couldn't find that in the document."

Document Context:
{context} '''
        ## add messages to chat_history
        st.session_state.chat_history.append(
            types.Content(
            role = 'user',
            parts = [types.Part(text=prompt)]
            )
        )
        with st.chat_message('Assistant'):
            with st.spinner('thinking...'):
                ## model answering
                response = client.models.generate_content(
                    model= 'gemini-2.5-flash-lite',
                    contents=st.session_state.chat_history,
                    config= types.GenerateContentConfig(
                        system_instruction=system_instruction
                    )       
                )
                answer = response.text

                # get token usage  from meta data
                if hasattr(response,'usage_metadata') and response.usage_metadata:
                    st.session_state.total_input_token += response.usage_metadata.prompt_token_count
                    st.session_state.total_output_token += response.usage_metadata.candidates_token_count

                st.markdown(answer)
        # add assistant response to LLM
        st.session_state.chat_history.append(
            types.Content(
                role = 'model',
                parts = [types.Part(text=answer)]
            )
        )
        #add assistant messages to UI
        st.session_state.messages.append(
            {
                'role' : 'assistant',
                'content' : answer,
                'sources' : results['documents'][0]
            }
        )
        # Keep only last 10 exchanges in chat history to manage context window
        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = st.session_state.chat_history[-20:]
        st.rerun()
elif file_uploaded:
    st.info("👆 Upload a PDF document to get started")


        







