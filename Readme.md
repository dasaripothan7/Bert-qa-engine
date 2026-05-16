# BERT-Based Question Answering Engine 🤖

An interactive Extractive Question Answering (QA) Web Application built using BERT, PyTorch, Hugging Face Transformers, and Streamlit.

The application allows users to provide a context paragraph and ask natural language questions. The model extracts the most relevant answer directly from the text.

---

## 🚀 Features

- Interactive web interface using Streamlit
- Extractive Question Answering with BERT
- Uses Hugging Face Transformers
- Confidence score calculation
- Fast model loading with Streamlit caching
- Real-time inference on custom text passages

---

## 🧠 Model Used

bert-large-uncased-whole-word-masking-finetuned-squad

This model is fine-tuned on the SQuAD dataset for extractive QA tasks.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- NLP / Deep Learning

---

## 📂 Project Structure

```bash
bert-qa-engine/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com
   cd bert-qa-engine
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

3. Paste a text passage into the "Context" box, type your question, and click the submit button to see the extracted answer along with its confidence score.