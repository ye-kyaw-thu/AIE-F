import os
import sys
import streamlit as st
# Ensure repo root is on sys.path for local imports
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# from scripts.eval import load_inference_bundle, predict_texts
from src.eliza import Eliza
from typing import Literal  

# model loading
@st.cache_resource
def load_models(checkpoint_path, language):
    # model, word2id, id2label, max_len = load_inference_bundle(checkpoint_path)
    model = None
    word2id = None
    id2label = None
    max_len = None
    eliza = Eliza(language=language)
    return model, word2id, id2label, max_len, eliza

model, word2id, id2label, max_len, eliza = load_models("../checkpoints/BiLSTM_model.pth", language="mm")
    
st.title("Group 2 - Hybrid ELiza")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response from Eliza & prediction from model
    response = eliza.respond(prompt)
    # pred_ids, pred_labels, pred_scores = predict_texts(
    #     model=model,
    #     word2id=word2id,
    #     id2label=id2label,
    #     max_len=max_len,
    #     texts=[prompt],
    #     stopwords_path="../data/stopwords.txt",
    # )
    # get predicted emotion and confidence
    # idx = pred_ids[0]
    # label = pred_labels[0]    
    # score = float(pred_scores[0])    
    label = "happy"
    score = 0.95
    # Display assistant response and emotion in chat UI    
    assistant_text = f"Eliza: {response}\n\nEmotion: {label} ({score:.2%})"
    with st.chat_message("assistant"):
        st.markdown(assistant_text)
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})


