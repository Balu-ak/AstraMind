# AstraMind
A guiding star through the world of AI and data science.

AstraMind is an interactive AI-powered tutor and technical Q&A assistant designed to make learning advanced technologies seamless and engaging.

With AstraMind, you can:

⚡ Ask technical questions and receive streaming, step-by-step answers

🎓 Use the built-in Tutor Tool to get structured lessons in Machine Learning, Deep Learning, Artificial Intelligence, Computer Vision, Data Science, Natural Language Processing, LLM Engineering, and Agentic AI

🔄 Seamlessly switch between models (OpenAI GPT models or local Ollama models) for flexibility and control

🧠 Customize the system prompt to tailor the assistant’s expertise and tone to your needs

🌐 Enjoy a Gradio-powered UI that’s lightweight, intuitive, and ready for both experimentation and deployment

Why AstraMind?
The name reflects the vision of a guiding “star mind” — a mentor that illuminates complex AI concepts and provides clarity, whether you’re a beginner exploring fundamentals or an advanced learner building real-world applications.

# AstraMind – Setup & Run Guide
## 🔧 Requirements

Make sure you have the following installed:

Python 3.9+

pip (Python package manager)

Ollama (if you want to use local LLaMA models) → install guide

Create a requirements.txt with these packages:

gradio>=4.0
openai>=1.30
requests
python-dotenv

## ⚙️ Setup Instructions

Clone or copy the project

git clone <your-repo-url>
cd astramind


## (Optional but recommended) Create a virtual environment

python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows


## Install dependencies

pip install -r requirements.txt


##Set your OpenAI API key
Create a .env file in the project root:

OPENAI_API_KEY=sk-xxxx-your-key


## ⚠️ If you’re only using Ollama, you can skip the API key.

(Optional) Pull an Ollama model
Make sure Ollama is running (ollama serve) and pull a model:

ollama pull llama3.2


## Run AstraMind

python astramind_app.py


## Open in browser
After running, you’ll see a message like:

Running on local URL:  http://127.0.0.1:7860


Open that URL to use AstraMind 🎓🌌

## 🎯 Usage

Ask technical questions → Type directly in the chat box.

Tutor mode → Prefix with tutor: (e.g., tutor: Explain backpropagation).

Switch models → Choose OpenAI (GPT-4o, GPT-4o-mini) or Ollama (local LLaMA).

Customize → Edit the System Prompt to change AstraMind’s persona/expertise.
