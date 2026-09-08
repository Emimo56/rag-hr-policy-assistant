````markdown
# 📋 HR Policy Assistant

An AI-powered HR Policy Assistant built using Retrieval-Augmented Generation (RAG).

Users can upload an HR policy PDF and ask questions about company policies such as:

- Vacation and PTO
- Sick leave
- Holidays
- Remote work
- Working hours
- Benefits
- Reimbursements
- Employee responsibilities
- Leave policies
- Workplace policies

The application retrieves relevant information from the uploaded PDF and uses Groq's GPT-OSS 20B model to generate an answer.

---

# 🏗️ Architecture

```text
HR Policy PDF
     |
     v
PyMuPDF
     |
     v
Text Extraction
     |
     v
Text Chunking
     |
     v
Sentence Transformers
     |
     v
Embeddings
     |
     v
FAISS Vector Search
     |
     v
Top Relevant Chunks
     |
     v
Groq GPT-OSS 20B
     |
     v
HR Policy Answer
````

---

# 🛠️ Technology Stack

* Python
* Streamlit
* PyMuPDF
* Sentence Transformers
* FAISS
* NumPy
* Groq API
* OpenAI GPT-OSS 20B

---

# ✨ Features

* Upload HR policy PDF
* Extract PDF text
* Automatically split text into chunks
* Generate semantic embeddings
* Store embeddings in FAISS
* Retrieve the most relevant policy sections
* Ask natural-language questions
* Generate answers using GPT-OSS 20B
* Display retrieved policy sources
* Display source page numbers
* Chat-style interface
* No external database required

---

# 📁 Project Structure

```text
hr-policy-assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🔑 Groq API Key

Create a Groq API key from the Groq Console.

Do NOT put the API key directly into `app.py`.

For Streamlit Cloud, add the key through the application's Secrets settings.

The secret should be:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

---

# 🚀 Run Locally

Install Python 3.12.

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/hr-policy-assistant.git
```

Go into the project:

```bash
cd hr-policy-assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Groq API key.

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your-groq-api-key"
```

Run Streamlit:

```bash
streamlit run app.py
```

Open the URL shown by Streamlit.

---

# ☁️ Deploy to Streamlit Community Cloud

1. Push the project to GitHub.

2. Open Streamlit Community Cloud.

3. Connect your GitHub account.

4. Select the repository.

5. Select:

```text
app.py
```

as the entrypoint.

6. Add the following secret:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

7. Deploy the application.

---

# 🧠 RAG Process

## 1. Document ingestion

The uploaded PDF is processed using PyMuPDF.

## 2. Chunking

The extracted text is divided into smaller overlapping chunks.

## 3. Embeddings

Sentence Transformers converts each chunk into a numerical vector.

## 4. Vector database

FAISS stores the vectors and performs similarity search.

## 5. Retrieval

When the user asks a question, the question is converted into an embedding.

FAISS retrieves the most relevant policy chunks.

## 6. Generation

The retrieved chunks are provided to GPT-OSS 20B.

The model generates an answer based on the retrieved policy content.

---

# ⚠️ Important

This application is an informational HR policy assistant.

It should not be treated as legal advice.

If the policy does not contain the answer, the assistant should say that the information could not be found instead of inventing a policy.

---

# 🔮 Future Improvements

Possible improvements include:

* Multiple PDF support
* Persistent FAISS indexes
* Document metadata
* Better chunking
* Hybrid search
* Re-ranking
* Conversation memory
* Authentication
* Admin dashboard
* Policy version management
* Policy comparison
* Citation highlighting
* Feedback collection
* Document deletion
* Multi-language support

```
```
