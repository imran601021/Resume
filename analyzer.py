import re
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _model


def extract_text_from_pdf(file_stream, max_chars=50000):
    try:
        pdf_bytes = file_stream.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) > max_chars:
                text = text[:max_chars]
                break
        doc.close()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        raise ValueError(f"Error reading PDF: {e}")


def extract_skills_advanced(resume_text, job_desc, skills_list, threshold=0.55):
    if not skills_list or not resume_text or not job_desc:
        return [], [], []

    model = get_model()
    matched, missing, partial_matches = [], [], []

    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding    = model.encode(job_desc,    convert_to_tensor=True)
    skill_embeddings = model.encode(skills_list, convert_to_tensor=True)

    for i, skill in enumerate(skills_list):
        se = skill_embeddings[i]
        resume_score = util.cos_sim(se, resume_embedding).item()
        job_score    = util.cos_sim(se, job_embedding).item()

        if resume_score > threshold:
            matched.append((skill, round(resume_score, 2)))
        elif job_score > (threshold + 0.05) and resume_score < threshold:
            missing.append((skill, round(job_score, 2)))
        elif resume_score > 0.40:
            partial_matches.append((skill, round(resume_score, 2)))

    return matched, missing, partial_matches


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
            'status':        "MATCH" if match else "BELOW REQUIREMENT",
            'resume_years':  resume_years,
            'required_years': required,
            'match':         match
        }
    return None


def extract_job_title(job_desc):
    lines = [l.strip() for l in job_desc.strip().split('\n') if l.strip()]
    if not lines:
        return "Job Title Not Found"
    if lines[0]:
        return lines[0]
    role_kw = ['engineer', 'developer', 'manager', 'analyst', 'designer',
               'scientist', 'lead', 'architect', 'director', 'consultant']
    for line in lines:
        if any(kw in line.lower() for kw in role_kw):
            return line
    return "Job Title Not Found"


def analyze_job_title_match(resume_text, job_desc):
    model = get_model()
    title = extract_job_title(job_desc)
    te = model.encode(title,       convert_to_tensor=True)
    re_ = model.encode(resume_text, convert_to_tensor=True)
    score = util.cos_sim(te, re_).item()
    return {'job_title': title, 'match_score': round(score, 2),
            'match_percentage': round(score * 100)}


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


def calculate_advanced_scores(resume_text, job_desc, skills_list):
    model = get_model()
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


def score_verdict(value):
    if value >= 85:
        return "Excellent match — you're well aligned for this role", "success"
    elif value >= 70:
        return "Good match — minor tailoring recommended", "info"
    elif value >= 50:
        return "Moderate match — some real gaps to address", "warning"
    else:
        return "Weak match — significant rework needed", "error"