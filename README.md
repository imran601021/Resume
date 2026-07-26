---
title: Resume Analyzer
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---
# 🎯 Resume Analyzer

An AI-powered resume matcher that scores how well your resume fits a job description — with skill gap analysis, keyword density, experience matching, and formatting feedback.

**New: AI Agent.** On top of the scoring engine, an agent (powered by Groq's free API running the open-weight OpenAI GPT-OSS 120B model) explains *why* your score is what it is, rewrites weak bullet points, and suggests job roles worth searching for based on your resume — with ready-to-click Indeed/LinkedIn/Naukri search links.

🔗 **Live Demo:** [huggingface.co/spaces/CallMeRolex/Resume-Analyzer](https://huggingface.co/spaces/CallMeRolex/Resume-Analyzer)

---

## 📸 Screenshots
![Input Screen](r1.png)

**Overall match, skill match and content match gauges**
![Score Gauges](r2.png)

**Quick summary and detailed skill breakdown**
![Detailed Analysis](r5.png)

**Keyword analysis — found vs missing**
![Keyword Analysis](r3.png)

**Formatting issues and recommendations**
![Recommendations](r4.png)

---

## ⚙️ Installation

**Prerequisites:** Python 3.9+

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Resume.git
cd Resume

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

> **Note:** First launch downloads the `all-mpnet-base-v2` model (~420MB). This only happens once.

### 🧠 Enable the AI Agent (optional, free)

The AI Agent uses [Groq](https://console.groq.com/keys)'s free API to run an open-weight LLM (OpenAI's GPT-OSS 120B). No paid plan needed.

1. Create a free account at [console.groq.com](https://console.groq.com/keys) and generate an API key.
2. Set it as an environment variable before running the app:
   ```bash
   export GROQ_API_KEY="your-key-here"     # Windows: set GROQ_API_KEY=your-key-here
   ```
   Or, if deploying on Streamlit Community Cloud / Hugging Face Spaces, add `GROQ_API_KEY` to your app's secrets.
3. In the sidebar, keep "Enable AI Agent" checked. If no key is set, the rest of the app still works — the AI Agent section just shows a note instead of feedback.

---

## 🐳 Run with Docker

```bash
docker compose up --build
```

Open your browser at **http://localhost:8501**

---

## 📖 How to Use

**Step 1 — Enter your skills**
In the left sidebar, type your skills as a comma-separated list.
Example: `Python, Docker, AWS, FastAPI`

**Step 2 — Upload your resume**
Click "Upload Resume PDF" and select your resume. Must be a text-based PDF (not a scanned image).

**Step 3 — Paste the job description**
Copy the full job description from any job portal and paste it into the text box.

**Step 4 — Read your results**
The app generates an overall match score broken down into:

| Score | What it measures |
|-------|-----------------|
| 📊 Overall Match | Weighted combination of all scores below |
| 🧠 Semantic Match | How closely your resume content aligns with the JD |
| 🎯 Skill Match | Which of your listed skills appear in the JD |
| 🔍 Keyword Density | How many JD keywords are present in your resume |
| 📅 Experience | Whether your years of experience meet the requirement |
| 💼 Job Title | How well your role history aligns with the position |
| 🧾 Formatting | Bullet usage, line length, and consistency |

You'll also get a **Recommendations** section at the bottom with specific actions to improve your score.

**Step 5 — Read the AI Agent's feedback**
If a `GROQ_API_KEY` is configured, the AI Agent section explains *why* you got that score, rewrites a couple of weak bullet points, and suggests 3-5 job roles worth searching for — with direct Indeed/LinkedIn/Naukri search links.

---

## 📁 Project Structure

```
Resume/
├── app.py               # Main Streamlit application
├── agent.py             # AI Agent: gap analysis, tips, job suggestions (Groq + GPT-OSS 120B)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Local Docker orchestration
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io) | Web UI |
| [sentence-transformers](https://www.sbert.net) | Semantic embeddings (`all-mpnet-base-v2`) |
| [PyMuPDF](https://pymupdf.readthedocs.io) | PDF text extraction |
| [Plotly](https://plotly.com) | Score gauge charts |
| [Groq](https://groq.com) + GPT-OSS 120B | AI Agent — gap analysis, bullet rewrites, job suggestions (free, open-source model) |

---

## 📄 License

MIT License
=======
---
title: Resume Analyzer 
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.57.0
app_file: app.py
pinned: false
---
