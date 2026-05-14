import fitz  # PyMuPDF
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Resume Analyzer Pro", layout="wide", initial_sidebar_state="expanded")

# ── Model ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

model = load_model()

# ── Text extraction ────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file, max_chars=50000):
    try:
        pdf_file = uploaded_file.read()
        doc = fitz.open(stream=pdf_file, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) > max_chars:
                st.warning(f"PDF too large. Using first {max_chars} characters.")
                text = text[:max_chars]
                break
        doc.close()
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            st.error("No text found in PDF. Ensure it's text-based, not scanned.")
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# ── Skill matching──────────────────────────
def extract_skills_advanced(resume_text, job_desc, skills_list, threshold=0.6):
    if not skills_list or not resume_text or not job_desc:
        return [], [], []

    matched, missing, partial_matches = [], [], []

    # Encode resume & job desc ONCE outside loop
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding    = model.encode(job_desc,    convert_to_tensor=True)

    # FIX: encode ALL skills in a single batched call
    skill_embeddings = model.encode(skills_list, convert_to_tensor=True)

    for i, skill in enumerate(skills_list):
        se = skill_embeddings[i]
        resume_score = util.cos_sim(se, resume_embedding).item()
        job_score    = util.cos_sim(se, job_embedding).item()

        if resume_score > threshold:
            matched.append((skill, round(resume_score, 2)))
        elif job_score > (threshold + 0.05) and resume_score < threshold:
            missing.append((skill, round(job_score, 2)))
        elif resume_score > 0.45:
            partial_matches.append((skill, round(resume_score, 2)))

    return matched, missing, partial_matches

# ── Experience extraction ──────────────────────────────────────────
def extract_experience_years(text):
    try:
        matches = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)', text, re.IGNORECASE)
        if matches:
            return sum(int(m) for m in matches)
        dates = re.findall(r'(\d{4})\s*-\s*(\d{4})', text)
        if dates:
            return sum(int(e) - int(s) for s, e in dates)
        return None
    except Exception:
        return None

def check_experience_match(resume_text, job_desc):
    resume_years = extract_experience_years(resume_text)
    job_match    = re.search(r'(\d+)\+?\s*years?', job_desc, re.IGNORECASE)
    if resume_years and job_match:
        required = int(job_match.group(1))
        match    = resume_years >= required
        return {
            'status':        "✅ MATCH" if match else "⚠️ BELOW REQUIREMENT",
            'resume_years':  resume_years,
            'required_years': required,
            'match':         match
        }
    return None

# ── Job title──────
def extract_job_title(job_desc):
    lines = [l.strip() for l in job_desc.strip().split('\n') if l.strip()]
    if not lines:
        return "Job Title Not Found"
    # First non-empty line is most likely the title
    if lines[0]:
        return lines[0]
    # Fallback: scan for a line with a common role keyword
    role_kw = ['engineer', 'developer', 'manager', 'analyst', 'designer',
               'scientist', 'lead', 'architect', 'director', 'consultant']
    for line in lines:
        if any(kw in line.lower() for kw in role_kw):
            return line
    return "Job Title Not Found"

def analyze_job_title_match(resume_text, job_desc):
    title = extract_job_title(job_desc)          # always a str now
    te = model.encode(title,       convert_to_tensor=True)
    re_ = model.encode(resume_text, convert_to_tensor=True)
    score = util.cos_sim(te, re_).item()
    return {'job_title': title, 'match_score': round(score, 2),
            'match_percentage': round(score * 100)}

# ── Keyword analysis ───────────────────────────────────────────────
def advanced_keyword_analysis(resume_text, job_desc):
    stop_words = {
        'the','a','an','and','or','but','in','on','at','to','for','of','with',
        'by','from','is','are','be','have','has','we','you','your','our','their',
        'this','that','which','who','will','would','should','could','must','may'
    }
    sections     = job_desc.lower().split('requirement')
    analysis_text = sections[1] if len(sections) > 1 else job_desc.lower()
    words        = re.findall(r'\b\w{4,}\b', analysis_text)
    keywords     = [w for w in set(words) if w not in stop_words][:30]
    resume_lower = resume_text.lower()
    found   = [kw for kw in keywords if kw in resume_lower]
    missing = [kw for kw in keywords if kw not in resume_lower]
    density = (len(found) / len(keywords) * 100) if keywords else 0
    return {
        'found': len(found), 'total': len(keywords),
        'density': round(density, 2),
        'found_keywords': found[:10], 'missing_keywords': missing[:10]
    }

# ── Formatting checks ──────────────────────────────────────────────
def check_formatting(text):
    issues, suggestions = [], []
    lines        = text.splitlines()
    bullet_count = text.count("•") + text.count("- ")

    if bullet_count < 3:
        issues.append("Few bullet points — add more for clarity")
    elif bullet_count > 50:
        issues.append("Too many bullet points — consolidate")

    caps_lines = sum(1 for l in lines if l.isupper() and len(l) > 10)
    if caps_lines > 3:
        issues.append(f"Excessive ALL CAPS ({caps_lines} lines)")

    long_lines = [l for l in lines if len(l) > 160]
    if long_lines:
        issues.append(f"Long lines ({len(long_lines)}) — break them up")

    if sum(b in text for b in ["●", "•", "■"]) > 1:
        suggestions.append("Mixed bullet styles — use a consistent style throughout")

    return issues, suggestions

# ── Scoring ────────────────────────────────────────────────────────
def calculate_advanced_scores(resume_text, job_desc, skills_list):
    scores = {}

    re_emb  = model.encode(resume_text[:5000], convert_to_tensor=True)
    jd_emb  = model.encode(job_desc[:2000],    convert_to_tensor=True)
    scores['content'] = round(util.cos_sim(re_emb, jd_emb).item() * 100)

    matched, missing, partial = extract_skills_advanced(resume_text, job_desc, skills_list)
    scores['skills'] = round((len(matched) / len(skills_list)) * 100) if skills_list else 0

    kw = advanced_keyword_analysis(resume_text, job_desc)
    scores['keywords'] = kw['density']

    exp = check_experience_match(resume_text, job_desc)
    scores['experience'] = (100 if exp['match'] else 50) if exp else 75

    title_match = analyze_job_title_match(resume_text, job_desc)
    scores['job_title'] = title_match['match_percentage']

    issues, suggestions = check_formatting(resume_text)
    scores['formatting'] = max(0, 100 - len(issues) * 10)

    scores['overall'] = round(
        scores['content']    * 0.35 +
        scores['skills']     * 0.25 +
        scores['keywords']   * 0.15 +
        scores['experience'] * 0.10 +
        scores['job_title']  * 0.10 +
        scores['formatting'] * 0.05
    )

    return scores, {
        'matched': matched, 'missing': missing, 'partial': partial,
        'experience': exp, 'title_match': title_match,
        'keywords': kw, 'formatting_issues': issues,
        'formatting_suggestions': suggestions
    }

# ── Gauge chart ────────────────────────────────────────────────────
def animated_gauge(label, value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": label, "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  50], "color": "#ffcccc"},
                {"range": [50, 75], "color": "#ffe699"},
                {"range": [75,100], "color": "#c6efce"}
            ]
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ── UI ─────────────────────────────────────────────────────────────
st.title("🎯 Resume Analyzer PRO")
st.write("**Advanced AI-powered resume matching** with semantic analysis, skill extraction, and detailed recommendations.")

with st.sidebar:
    st.header("⚙️ Configuration")
    user_skill_input = st.text_input("Enter your skills (comma-separated)", placeholder="Python, Flask, Docker, AWS")
    skills_list = [s.strip().lower() for s in user_skill_input.split(",") if s.strip()]
    if len(skills_list) > 20:
        st.warning("Limited to 20 skills. Using first 20.")
        skills_list = skills_list[:20]
    show_detailed = st.checkbox("Show Detailed Analysis", value=True)
    st.markdown("---")
    st.caption("🤖 all-mpnet-base-v2 (80–85% accuracy)")
    st.caption("📊 Semantic analysis enabled")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
with col2:
    job_desc = st.text_area("Paste Job Description", height=150)

if uploaded_file and job_desc:
    if not skills_list:
        st.warning("⚠️ Enter at least one skill to analyze.")
    else:
        progress_bar = st.progress(0)
        status_text  = st.empty()
        try:
            status_text.text("📄 Extracting resume...")
            progress_bar.progress(20)
            resume_text = extract_text_from_pdf(uploaded_file)

            if resume_text:
                status_text.text("🔍 Analysing with AI...")
                progress_bar.progress(60)
                scores, details = calculate_advanced_scores(resume_text, job_desc, skills_list)
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                progress_bar.empty()
                status_text.empty()

                st.divider()
                c1, c2, c3 = st.columns(3)
                with c1: animated_gauge("📊 Overall Match", scores['overall'], "#4CAF50")
                with c2: animated_gauge("🎯 Skill Match",   scores['skills'],  "#2196F3")
                with c3: animated_gauge("🧠 Content Match", scores['content'], "#FF9800")

                st.subheader("📈 Score Breakdown")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Semantic",    f"{scores['content']}%")
                c2.metric("Skills",      f"{scores['skills']}%")
                c3.metric("Keywords",    f"{scores['keywords']:.0f}%")
                c4.metric("Experience",  f"{scores['experience']}%")
                c5.metric("Formatting",  f"{scores['formatting']}%")

                st.divider()
                st.subheader("🎯 Quick Summary")
                c1, c2, c3 = st.columns(3)
                with c1: st.info(f"**✅ Matched Skills**\n{len(details['matched'])}/{len(skills_list)}")
                with c2: st.warning(f"**⚠️ Missing Skills**\n{len(details['missing'])}/{len(skills_list)}")
                with c3:
                    if details['experience']:
                        em = "✅" if details['experience']['match'] else "⚠️"
                        st.info(f"**{em} Experience**\n{details['experience']['resume_years']}+ years")
                    else:
                        st.info("**❓ Experience**\nNot found")

                if show_detailed:
                    st.divider()
                    st.subheader("📝 Detailed Analysis")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**✅ Matched Skills:**")
                        for skill, sc in details['matched'][:5]:
                            st.markdown(f"- {skill} ({sc*100:.0f}%)")
                        if not details['matched']: st.info("None matched")
                    with c2:
                        st.markdown("**❌ Missing Skills:**")
                        for skill, _ in details['missing'][:5]:
                            st.markdown(f"- {skill}")
                        if not details['missing']: st.success("All matched!")
                    with c3:
                        st.markdown("**⚠️ Partial Matches:**")
                        for skill, sc in details['partial'][:5]:
                            st.markdown(f"- {skill} ({sc*100:.0f}%)")
                        if not details['partial']: st.info("None")

                    st.divider()
                    st.markdown("**🔍 Keyword Analysis:**")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Found: {details['keywords']['found']}/{details['keywords']['total']}**")
                        st.caption(f"Density: {details['keywords']['density']:.1f}%")
                        for kw in details['keywords']['found_keywords'][:8]:
                            st.markdown(f"✓ {kw}")
                    with c2:
                        st.markdown(f"**Missing: {len(details['keywords']['missing_keywords'])}**")
                        for kw in details['keywords']['missing_keywords'][:8]:
                            st.markdown(f"✗ {kw}")

                    st.divider()
                    st.markdown("**💼 Job Title Match:**")
                    c1, c2 = st.columns(2)
                    c1.write(f"Position: **{details['title_match']['job_title']}**")
                    c2.write(f"Match: **{details['title_match']['match_percentage']}%**")

                    if details['experience']:
                        st.divider()
                        st.markdown("**📅 Experience Analysis:**")
                        exp = details['experience']
                        em  = "✅" if exp['match'] else "⚠️"
                        st.write(f"{em} {exp['status']}: {exp['resume_years']} vs {exp['required_years']} required years")

                    if details['formatting_issues']:
                        st.divider()
                        st.markdown("**🧾 Formatting Issues:**")
                        for issue in details['formatting_issues']:
                            st.warning(issue)
                    if details['formatting_suggestions']:
                        st.markdown("**💡 Suggestions:**")
                        for s in details['formatting_suggestions']:
                            st.info(s)

                    st.divider()
                    st.subheader("🎯 Recommendations")
                    recs = []
                    if scores['overall'] < 70:
                        recs.append("🔴 **Overall match is low** — Rewrite sections to better align with the JD")
                    if details['missing']:
                        recs.append(f"🟡 **Add missing skills** — {len(details['missing'])} required skills not found")
                    if scores['content'] < 70:
                        recs.append("📝 **Improve content** — Use more job description keywords naturally")
                    if scores['keywords'] < 60:
                        recs.append("🔍 **Increase keyword density** — Add specific technical terms")
                    if details['experience'] and not details['experience']['match']:
                        recs.append(f"⏳ **Experience gap** — You have {details['experience']['resume_years']} yrs; role needs {details['experience']['required_years']}")
                    if scores['formatting'] < 70:
                        recs.append("✨ **Improve formatting** — Better structure helps ATS scanning")
                    if recs:
                        for r in recs: st.markdown(f"• {r}")
                    else:
                        st.success("✅ Your resume looks great for this position!")

        except Exception as e:
            st.error(f"Error: {e}")
            progress_bar.empty()
            status_text.empty()
else:
    st.info("👈 **To get started:**\n\n1. Enter your skills in the sidebar\n2. Upload your resume (PDF)\n3. Paste the job description\n4. View your analysis!")

st.divider()
st.caption("⚡ **PRO Version** | 🤖 Advanced AI | 📊 Semantic Matching | 💼 Career Intelligence")