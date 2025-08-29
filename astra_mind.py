#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, json, requests, gradio as gr
from typing import List, Dict, Any, Generator, Optional
from dotenv import load_dotenv
from openai import OpenAI


# In[2]:


load_dotenv(override=True)

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert technical question answerer. "
    "Explain clearly, show steps for math when relevant, and keep answers concise."
)


# In[3]:


# --- Keys & Hosts ---
api_key = os.getenv("OPENAI_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"

openai_client = OpenAI()


# In[4]:


if not api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif not api_key.startswith("sk-proj-"):
    print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
elif api_key.strip() != api_key:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
else:
    print("API key found and looks good so far!")


# In[5]:


# -------------------- Tutor Tool --------------------
DOMAINS = [
    "Machine Learning", "Deep Learning", "Artificial Intelligence",
    "Computer Vision", "Data Science", "Natural Language Processing",
    "LLM Engineering", "Agentic AI"
]
LEVELS = ["Beginner", "Intermediate", "Advanced"]

def tutor_tool(domain: str, level: str, prompt: str) -> str:
    domain = domain if domain in DOMAINS else "Machine Learning"
    level = level if level in LEVELS else "Beginner"
    topic = prompt.strip() or f"Introduction to {domain}"

    def bullets(items): return "\n".join([f"- {x}" for x in items])

    overview = f"{domain} — {topic} ({level} level)"
    goals = [
        "Understand the core idea in plain language",
        "See where it’s used and why it matters",
        "Learn the minimal workflow you’d follow",
        "Get a small practice task and a quick quiz"
    ]
    key_concepts_map = {
        "Machine Learning": ["Data → Features → Model → Loss → Optimization → Evaluation",
                             "Bias/variance, overfitting, train/val/test splits",
                             "Supervised vs. Unsupervised vs. Reinforcement"],
        "Deep Learning": ["Neural nets, activation functions, backprop",
                          "Architectures: MLP, CNN, RNN/Transformer",
                          "Regularization: dropout, weight decay, early stopping"],
        "Artificial Intelligence": ["Goal-directed behavior, search/planning",
                                    "Knowledge representation, reasoning",
                                    "Learning as a subset of AI"],
        "Computer Vision": ["Images → tensors, convolutions, pooling",
                            "Classification, detection, segmentation",
                            "Data aug, transfer learning (pretrained CNNs)"],
        "Data Science": ["Question → Data → Clean → Explore → Model → Communicate",
                         "EDA, visualization, statistics",
                         "Experiment design, A/B testing, causality (basic)"],
        "Natural Language Processing": ["Tokenization, embeddings, sequence modeling",
                                        "Classical (n-grams) → Neural (RNN/Transformer)",
                                        "Tasks: classification, NER, QA, summarization"],
        "LLM Engineering": ["Prompting, system prompts, structured outputs",
                            "RAG pipelines (index, retrieve, augment)",
                            "Evaluation, guardrails, latency/cost tradeoffs"],
        "Agentic AI": ["Planning + Tools + Memory + Reflection",
                       "Function/tool calling, multi-step workflows",
                       "Safety, determinism, monitoring"]
    }
    key_concepts = key_concepts_map.get(domain, ["Core pipeline", "Common pitfalls", "Evaluation"])

    workflow_map = {
        "Beginner": ["Define the question", "Collect/inspect a toy dataset",
                     "Pick a baseline model", "Evaluate simply", "Iterate once"],
        "Intermediate": ["Feature engineering / architecture choice",
                         "Hyperparameter tuning", "Proper validation",
                         "Error analysis", "Deploy a minimal demo"],
        "Advanced": ["Data-centric iteration at scale",
                     "System constraints (latency/cost/safety)",
                     "Robust evaluation & monitoring", "Ablations", "Deployment & CI/CD"]
    }
    workflow = workflow_map[level]

    practice_map = {
        "Machine Learning": "Train a logistic regression on a small tabular dataset; report accuracy and confusion matrix.",
        "Deep Learning": "Fine-tune a small CNN on CIFAR-10 for 5 epochs; plot train/val loss.",
        "Artificial Intelligence": "Implement BFS/DFS for a toy maze; compare path lengths and runtime.",
        "Computer Vision": "Use a pretrained ResNet to classify 100 images; analyze top-5 errors.",
        "Data Science": "Do EDA on a CSV (missing values, outliers); create 3 actionable charts.",
        "Natural Language Processing": "Build a sentiment classifier with a small Transformer; evaluate F1.",
        "LLM Engineering": "Create a RAG toy over 5 markdown files; answer 5 queries and note failures.",
        "Agentic AI": "Wire a tool-calling demo and execute a 3-step plan."
    }
    practice = practice_map[domain]

    quiz = [
        "Q1: Name two common pitfalls for this topic.",
        "Q2: What metric(s) would you use and why?",
        "Q3: What would you change if data is noisy or limited?"
    ]

    next_steps_map = {
        "Beginner": ["Reproduce the practice task end-to-end", "Write a 200-word summary",
                     "Try a different dataset/model and compare"],
        "Intermediate": ["Tune 2 hyperparameters systematically", "Add proper validation",
                         "Document failure cases and fixes"],
        "Advanced": ["Design an eval suite incl. edge cases", "Measure latency/cost",
                     "Automate experiments (sweeps) and reporting"]
    }
    next_steps = next_steps_map[level]

    out = []
    out.append(f"### {overview}\n")
    out.append("**Goals:**\n" + bullets(goals) + "\n")
    out.append("**Key Concepts:**\n" + bullets(key_concepts) + "\n")
    out.append("**Typical Workflow:**\n" + bullets(workflow) + "\n")
    out.append(f"**Practice Task:**\n- {practice}\n")
    out.append("**Mini-Quiz:**\n" + bullets(quiz) + "\n")
    out.append("**Next Steps:**\n" + bullets(next_steps))
    return "\n".join(out)


# In[6]:


# -------------------- Message helpers (for messages-format) --------------------
def build_openai_messages_from_history(system_prompt: str, messages_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """history is a list of {'role': 'user'|'assistant', 'content': str} dicts"""
    msgs = [{"role": "system", "content": system_prompt}]
    for m in messages_history:
        if m.get("role") in ("user", "assistant") and m.get("content") is not None:
            msgs.append({"role": m["role"], "content": m["content"]})
    return msgs


# In[7]:


def openai_stream(model: str, temperature: float, system_prompt: str,
                  messages_history: List[Dict[str, str]]):
    msgs = build_openai_messages_from_history(system_prompt, messages_history)
    try:
        stream = openai_client.chat.completions.create(
            model=model, messages=msgs, temperature=temperature, stream=True
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
    except Exception as e:
        yield f"[OpenAI error] {e}"


# In[8]:


def ollama_stream(model: str, temperature: float, system_prompt: str,
                  messages_history: List[Dict[str, str]]):
    payload = {
        "model": model,
        "system": system_prompt,
        "messages": [],
        "stream": True,
        "options": {"temperature": temperature}
    }
    for m in messages_history:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and content is not None:
            payload["messages"].append({"role": role, "content": content})
    try:
        with requests.post(OLLAMA_CHAT_URL, json=payload, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    msg = data.get("message", {})
                    delta = msg.get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue
    except Exception as e:
        yield f"[Ollama error] {e}"


# In[9]:


def chat_fn(messages, user_text, system_prompt, provider, model, temperature,
            tutor_enabled, tutor_domain, tutor_level):
    user_text = (user_text or "").strip()
    if not user_text:
        yield messages, gr.update(value="")   # clear textbox even if empty
        return

    # Tutor Tool branch
    if tutor_enabled and user_text.lower().startswith("tutor:"):
        lesson_prompt = user_text[len("tutor:"):].strip()
        tool_answer = tutor_tool(tutor_domain, tutor_level, lesson_prompt)
        messages = messages + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": tool_answer},
        ]
        yield messages, gr.update(value="")
        return

    # LLM branch
    messages = messages + [{"role": "user", "content": user_text}, {"role": "assistant", "content": ""}]
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    if provider == "OpenAI":
        stream = openai_stream(model, temperature, sys_prompt, messages[:-1])
    else:
        stream = ollama_stream(model, temperature, sys_prompt, messages[:-1])

    full = []
    for token in stream:
        full.append(token)
        messages[-1]["content"] = "".join(full)
        yield messages, gr.update(value="")   # keep clearing txt each token


# In[10]:


# -------------------- UI --------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("Your Personal AI Tutor with OpenAI and Llama")

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            provider = gr.Radio(["OpenAI", "Ollama"], value="OpenAI", label="Provider")
            model = gr.Dropdown(["gpt-4o-mini", "gpt-4o"], value="gpt-4o-mini", label="Model")
            temperature = gr.Slider(0.0, 1.5, value=0.2, step=0.1, label="Temperature")
            system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, lines=4, label="System Prompt")

            gr.Markdown("**Tutor Tool** — type messages like `tutor: Explain backpropagation`")
            tutor_enabled = gr.Checkbox(value=True, label="Enable Tutor Tool for 'tutor:' messages")
            tutor_domain = gr.Dropdown(DOMAINS, value="Machine Learning", label="Tutor Domain")
            tutor_level = gr.Dropdown(LEVELS, value="Beginner", label="Tutor Level")

        with gr.Column(scale=2, min_width=480):
            chat = gr.Chatbot(height=440, type="messages")  # IMPORTANT: messages format
            with gr.Row():
                txt = gr.Textbox(placeholder="Ask a technical question… (or: tutor: Intro to CNNs)", scale=4)
                send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")

    def on_provider_change(p):
        if p == "OpenAI":
            return gr.update(choices=["gpt-4o-mini", "gpt-4o"], value="gpt-4o-mini")
        else:
            try:
                resp = requests.get(OLLAMA_TAGS_URL, timeout=1.5)
                names = [m.get("name") for m in resp.json().get("models", []) if m.get("name")]
                if not names: names = ["llama3.2"]
                return gr.update(choices=names, value=names[0])
            except Exception:
                return gr.update(choices=["llama3.2"], value="llama3.2")

    provider.change(fn=on_provider_change, inputs=provider, outputs=model)

    send.click(
    chat_fn,
    inputs=[chat, txt, system_prompt, provider, model, temperature, tutor_enabled, tutor_domain, tutor_level],
    outputs=[chat, txt]   # 👈 return chat + clear textbox
    )
    txt.submit(
    chat_fn,
    inputs=[chat, txt, system_prompt, provider, model, temperature, tutor_enabled, tutor_domain, tutor_level],
    outputs=[chat, txt]
    )
    clear.click(lambda: [], outputs=chat)


# In[11]:


demo.launch(share=True,inbrowser=True)


# In[ ]:




