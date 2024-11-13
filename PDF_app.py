#!/usr/bin/env python
# coding: utf-8
# In[3]:
#!pip install streamlit
# In[5]:
#!pip install groq pymupdf
# In[15]:
import streamlit as st
import fitz  # PyMuPDF
from groq import Groq
from PIL import Image
import re
from collections import Counter


# In[16]:


client = Groq(api_key='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')


# In[39]:

def extract_text_by_page(pdf_path):
    """Extracts text from each page and returns it as a list of strings, one for each page."""
    doc = fitz.open("pdf", pdf_path.read())
    pages_text = [doc.load_page(i).get_text("text") for i in range(doc.page_count)]
    doc.close()
    return pages_text

def clean_text(text):
    # Basic cleaning as before
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text, max_chunk_size=512):
    # Initialize the summarization pipeline for contextual grouping
    summarizer = pipeline("summarization")
    
    # Split the text into sentences
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        # Check if adding the next sentence would exceed the chunk size
        current_chunk.append(sentence)
        chunk_text = " ".join(current_chunk)
        
        # Summarize to see if it forms a coherent chunk
        summary = summarizer(chunk_text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        
        if len(summary.split()) < max_chunk_size:
            continue
        else:
            chunks.append(chunk_text)
            current_chunk = []

    # Add any remaining sentences as the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_pdf_cleantext(path):
    texts_pages=extract_text_by_page(path)
    cleantext=''
    for text in texts_pages:
        temp=clean_text(text)
        cleantext+=temp
    return cleantext


def summarize_text(text, model_name):
    try:
        summary_response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": f"Summarize the following text: {text}"
                }
            ],
            model=model_name,
        )
        return summary_response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"
    
def ask_question(context, question, model_name):
    try:
        answer_response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": f"Context: {context} Question: {question}"
                }
            ],
            model=model_name,
        )
        return answer_response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"
def what_else(text, model_name):
    try:
        we_response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": f"Suggest prompt ideas for this text. What can be done? return only new line seperated answers nothing else : {text}"
                }
            ],
            model=model_name,
        )
        return we_response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

        # Extract each suggestion and store it in a list
        suggestions = [choice.message.content for choice in we_response.choices]
        return suggestions
    except Exception as e:
        return [f"An error occurred: {e}"]






# Set page configuration
st.set_page_config(page_title="PDF Explorer", page_icon="📄", layout="centered")

# Title and header image
st.title("📄 PDF Explorer")
st.markdown(
    """
    <style>
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px;}
    .stTextInput>div>div>input {border-radius: 8px; padding: 8px;}
    .stSelectbox>div>div>div>input {border-radius: 8px; padding: 8px;}
    </style>
    """,
    unsafe_allow_html=True
)
image = Image.open('Dum.png')
st.image(image, use_container_width='always')

# File uploader section
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=False)
if uploaded_file is not None:
    # Display extracted text preview
    pdf_text = get_pdf_cleantext(uploaded_file)
    st.subheader("🔍 Text Extracted from PDF:")
    st.write(pdf_text[:500])  # Display a snippet of the text for review

    # Summarization model selection
    st.markdown("---")
    st.subheader("Summarization Options")
    summarize_model = st.selectbox(
        "Choose a model for summarization:",
        ["mixtral-8x7b-32768", "llama-3.1-8b-instant", "gemma-7b-it", "llama3-70b-8192", "whisper-large-v3"]
    )
    col1, col2 = st.columns(2)
    with col1:
        summary_button = st.button("🔍 Summarize PDF")
    with col2:
        what_else_button = st.button("💡 Get Suggestions")

    # Generate summary on button click
    if summary_button:
        summary = summarize_text(pdf_text, summarize_model)
        st.subheader("📋 Summary:")
        st.write(summary)
        st.button("📋 Copy Summary", on_click=lambda: st.write(summary))  # Copy button for summary

    # Generate suggestions on button click
    if what_else_button:
        suggestions = what_else(pdf_text, model_name="llama-3.1-8b-instant")
        if suggestions:
            suggetion_prompt = st.selectbox(
                "Choose a suggestion to ask",
                options=suggestions.splitlines(),
                key="suggestion_selectbox"
            )
        if suggetion_prompt:
                    answer = ask_question(pdf_text, suggetion_prompt, model_name="llama-3.1-8b-instant")
                    st.subheader("🤖 Response to Suggestion:")
                    st.write(answer)
                    st.button("📋 Copy Response", on_click=lambda: st.write(answer))  # Copy button for suggestion response
           
                

    # Question-answering section
    st.markdown("---")
    st.subheader("Ask Questions About the PDF")
    st.markdown("Type your question below, and let the model answer it based on the PDF content.")
    question_model = "llama-3.1-8b-instant"
    question = st.text_input("Ask a question about the PDF:")
    if question:
        answer = ask_question(pdf_text, question, question_model)
        st.subheader("🤖 Answer:")
        st.write(answer)
        st.button("📋 Copy Answer", on_click=lambda: st.write(answer))  # Copy button for user-entered question response

    


