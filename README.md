- **Container runtime:** [Podman](https://podman.io) (daemonless, rootless — used in place of Docker)
- **CI/CD:** GitHub Actions builds and pushes the image on every push to `main`
- **Registry:** GitHub Container Registry (GHCR)
- **Hosting:** [Render](https://render.com) free tier
- **Image optimization:** uses [uv](https://github.com/astral-sh/uv) for fast dependency installs and a CPU-only PyTorch build to avoid bundling unused CUDA libraries, keeping the image significantly smaller

---

## ⚙️ Local Installation

**Prerequisites:** Python 3.9+

```bash
# 1. Clone the repo
git clone https://github.com/imran601021/Resume-Analyzer.git
cd Resume-Analyzer

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
   Or, if deploying on Render / Streamlit Community Cloud / Hugging Face Spaces, add `GROQ_API_KEY` to your app's environment variables/secrets.
3. In the sidebar, keep "Enable AI Agent" checked. If no key is set, the rest of the app still works — the AI Agent section just shows a note instead of feedback.

---

## 🐳 Run with Podman / Docker

```bash
podman build -t resume-analyzer .
podman run -p 8501:8501 -e GROQ_API_KEY="your-key-here" resume-analyzer
```

(Docker works identically — swap `podman` for `docker` if that's your runtime.)

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
The app generates an overall match score, shown as gauges with a plain-language verdict, broken down into tabs:

| Tab | What it shows |
|-----|----------------|
| Overview | Score breakdown (5 metrics) + quick summary |
| Skills Detail | Matched, missing, and partially-matched skills |
| Keywords | Job description keywords found vs missing in your resume |
| Experience & Title | Years of experience match + job title alignment |
| Formatting & Tips | Formatting issues, suggestions, and actionable recommendations |

| Score | What it measures |
|-------|-------------------|
| Overall Match | Weighted combination of all scores below |
| Semantic Match | How closely your resume content aligns with the JD |
| Skill Match | Which of your listed skills appear in the JD |
| Keyword Density | How many JD keywords are present in your resume |
| Experience | Whether your years of experience meet the requirement |
| Job Title | How well your role history aligns with the position |
| Formatting | Bullet usage, line length, and consistency |

**Step 5 — Read the AI Agent's feedback**
If a `GROQ_API_KEY` is configured, the AI Agent section explains *why* you got that score, rewrites a couple of weak bullet points, and suggests 3-5 job roles worth searching for — with direct Indeed/LinkedIn/Naukri search links.

---

## 📁 Project Structure
Resume-Analyzer/
├── app.py # Main Streamlit application (UI + scoring logic)
├── agent.py # AI Agent: gap analysis, tips, job suggestions (Groq + GPT-OSS 120B)
├── requirements.txt # Python dependencies
├── Dockerfile # Container image definition (uv + CPU-only torch)
├── docker-compose.yml # Local Docker/Podman orchestration
├── .github/workflows/
│ └── build-push.yml # CI/CD: builds and pushes image to GHCR on every push
└── README.md

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io) | Web UI |
| [sentence-transformers](https://www.sbert.net) | Semantic embeddings (`all-mpnet-base-v2`) |
| [PyMuPDF](https://pymupdf.readthedocs.io) | PDF text extraction |
| [Plotly](https://plotly.com) | Score gauge charts |
| [Groq](https://groq.com) + GPT-OSS 120B | AI Agent — gap analysis, bullet rewrites, job suggestions (free, open-source model) |
| [Podman](https://podman.io) | Containerization (daemonless Docker alternative) |
| [uv](https://github.com/astral-sh/uv) | Fast Python dependency installer, used in the Docker build |
| GitHub Actions | CI/CD — automated build & push to GHCR on every commit |
| [Render](https://render.com) | Hosting / deployment |

---

## 📄 License

MIT License
READMEEOF
