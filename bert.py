import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# 1. Page Configuration
st.set_page_config(page_title="BERT QA Engine", page_icon="🤖")
st.title("🤖 Extractive Question-Answering Engine")
st.write("Provide a passage of text and ask a question. The BERT model will extract the answer.")

# 2. Robust Cache Logic (Bypasses pipeline string naming entirely)
@st.cache_resource
def load_qa_model():
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

with st.spinner("Loading BERT model... Please wait (this may take a minute the first time)."):
    tokenizer, model = load_qa_model()

# 3. Default Values
default_context = """The Nile River is the longest river in the world, flowing through northeastern Africa for over 6,600 kilometers. It is a vital water source for Egypt and Sudan. The river has two major tributaries: the White Nile and the Blue Nile. The White Nile is considered to be the headwaters and primary stream of the Nile itself."""
default_question = "What is the longest river in the world?"

# 4. User Inputs
context = st.text_area("Context / Source Text", value=default_context, height=150)
question = st.text_input("Your Question", value=default_question)

# 5. Model Inference Execution
if st.button("Extract Answer", type="primary"):
    if context and question:
        with st.spinner("Analyzing text..."):
            # Encode context and question together
            inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Find tokens with highest start and end scores
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            
            # Convert token IDs back to plain text string
            answer_tokens = inputs.input_ids[0][answer_start:answer_end]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            # Calculate an approximate confidence score via Softmax probabilities
            start_prob = torch.softmax(outputs.start_logits, dim=-1)[0][answer_start].item()
            end_prob = torch.softmax(outputs.end_logits, dim=-1)[0][answer_end - 1].item()
            confidence_score = start_prob * end_prob

            # 6. Display Results Nicely
            if answer.strip():
                st.success(f"**Answer:** {answer}")
                st.metric(label="Confidence Score", value=f"{confidence_score:.2%}")
            else:
                st.warning("The model could not extract a reliable answer from the text.")
    else:
        st.warning("Please fill in both the context and the question fields.")
