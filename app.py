import fitz  # PyMuPDF
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import uuid
import plotly.graph_objects as go
from fuzzywuzzy import fuzz

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Resume Analyzer",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==================== CACHING: CRITICAL FOR PERFORMANCE ====================

@st.cache_resource
def load_model():
    """Load SentenceTransformer model once and reuse"""
    return SentenceTransformer("intfloat/e5-large-v2")

# Load models once
model = load_model()

# ==================== TEXT EXTRACTION & PREPROCESSING ====================

def extract_text_from_pdf(uploaded_file, max_chars=50000):
    """
    Extract text from PDF with size limit
    """
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
            st.error("No text found in PDF. Ensure it's a text-based PDF, not scanned image.")
        
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def preprocess_text(text):
    """Preprocess text efficiently"""
    return text.lower().strip()

# ==================== SKILLS MATCHING ====================

def extract_skills(resume_text, job_desc, skills_list, threshold=70):
    """
    Match skills efficiently
    """
    if not skills_list or not resume_text or not job_desc:
        return [], []
    
    resume_text = resume_text.lower()
    job_desc = job_desc.lower()
    matched = []
    missing = []

    for skill in skills_list:
        skill_lower = skill.lower()
        score_resume = fuzz.partial_ratio(skill_lower, resume_text)
        
        if score_resume >= threshold:
            matched.append(skill)
        else:
            score_jd = fuzz.partial_ratio(skill_lower, job_desc)
            if score_jd >= threshold:
                missing.append(skill)
    
    return matched, missing

# ==================== FORMATTING CHECKS ====================

def check_formatting(text):
    """Check formatting with better logic"""
    issues = []
    
    # Optimize bullet point detection
    bullet_count = text.count("•") + text.count("- ")
    if bullet_count < 3:
        issues.append("Use more bullet points for better readability.")
    elif bullet_count > 50:
        issues.append("Too many bullet points. Consolidate where possible.")
    
    # Better ALL CAPS detection
    lines = text.splitlines()
    caps_lines = sum(1 for line in lines if line.isupper() and len(line) > 10)
    if caps_lines > 3:
        issues.append(f"Avoid using ALL CAPS excessively ({caps_lines} lines found).")
    

    long_lines = [line for line in lines if len(line) > 160]
    if long_lines:
        issues.append(f"{len(long_lines)} lines are too long (>160 chars). Break them up.")
    
    return issues

# ==================== SCORING ====================

def calculate_style_score(text):
    """Calculate style score"""
    bullets = text.count("•") + text.count("- ")
    if bullets >= 10:
        return 100
    elif bullets >= 5:
        return 75
    elif bullets >= 3:
        return 50
    else:
        return 25

def calculate_ats_score(text, job_desc):
    """
    Calculate ATS score
    """

    keywords = [
        word.lower() for word in job_desc.split() 
        if len(word) > 4 and word.isalpha()
    ]
    
    if not keywords:
        return 0
    
    text_lower = text.lower()
    matches = sum(1 for k in keywords if k in text_lower)
    density = (matches / len(keywords)) * 100
    
    return min(100, int(density))

def calculate_section_score(text):
    """Calculate section score"""
    sections = [
        'education', 'experience', 'skills', 
        'projects', 'summary', 'certifications'
    ]
    text_lower = text.lower()
    count = sum(1 for sec in sections if sec in text_lower)
    return round((count / len(sections)) * 100)

# ==================== SEMANTIC MATCHING ====================

def compare_resume_with_job(resume_text, job_desc, skills_list):
    """
    Compare resume with job description
    """
    if not resume_text or not job_desc:
        return None, [], []
    
    resume_text = resume_text.strip()
    job_desc = job_desc.strip()
    
    # Limit text for faster processing
    resume_clean = resume_text[:5000]
    jd_clean = job_desc[:2000]
    
    try:
        # More efficient embedding strategy
        resume_embedding = model.encode(
            "passage: " + resume_clean.lower(),
            convert_to_tensor=True
        )
        jd_embedding = model.encode(
            "query: " + jd_clean.lower(),
            convert_to_tensor=True
        )
        
        similarity = util.cos_sim(resume_embedding, jd_embedding).item()
        matched, missing = extract_skills(resume_text, job_desc, skills_list)
        
        return similarity, matched, missing
    except Exception as e:
        st.error(f"Similarity calculation failed: {e}")
        return None, [], []

# ==================== UI COMPONENTS ====================

def custom_animated_bar(label, value, color_from, color_to):
    """Animated progress bar"""
    bar_id = str(uuid.uuid4()).replace('-', '')
    animation_name = f"fillBar{bar_id}"
    
    keyframes = f"""
    <style>
    @keyframes {animation_name} {{
        from {{ width: 0%; }}
        to {{ width: {value}%; }}
    }}
    .bar-{bar_id} {{
        animation: {animation_name} 1.5s ease-in-out forwards;
    }}
    </style>
    """
    
    st.markdown(f"<b>{label}</b>", unsafe_allow_html=True)
    st.markdown(keyframes, unsafe_allow_html=True)
    st.markdown(f'''
    <div style="background-color: #e0e0e0; border-radius: 10px; height: 24px; margin-bottom: 12px;">
        <div class="bar-{bar_id}" style="
            height: 100%;
            background: linear-gradient(to right, {color_from}, {color_to});
            border-radius: 10px;
            text-align: center;
            color: white;
            font-weight: bold;
            line-height: 24px;">
            {value}%
        </div>
    </div>
    ''', unsafe_allow_html=True)

def animated_gauge(label, value, color):
    """Gauge meter visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": label, "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#ffcccc"},
                {"range": [50, 80], "color": "#ffe699"},
                {"range": [80, 100], "color": "#c6efce"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================== MAIN STREAMLIT UI ====================

st.title("📄 Resume to Job Description Matcher")
st.write(
    "Upload your resume (PDF) and paste a job description to analyze compatibility. "
    "**Performance optimized - instant results!**"
)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    user_skill_input = st.text_input(
        "Enter your skills (comma-separated)",
        placeholder="Python, Flask, Docker, AWS"
    )
    skills_list = [
        s.strip().lower() for s in user_skill_input.split(",") 
        if s.strip()
    ]
    
    # Add skill limit validation
    if len(skills_list) > 20:
        st.warning(f"⚠️ Limited to 20 skills. Using first 20 of {len(skills_list)}.")
        skills_list = skills_list[:20]
    
    enable_formatting_feedback = st.checkbox("Enable Formatting Feedback", value=True)
    
    st.markdown("---")
    st.caption("💡 Tip: Formatting suggestions based on ATS best practices")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

with col2:
    job_desc = st.text_area("Paste Job Description", height=150)

# ==================== ANALYSIS LOGIC ====================

if uploaded_file and job_desc:
    if not skills_list:
        st.warning("⚠️ Please enter at least one skill in the sidebar to analyze.")
    else:
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Extract PDF
            status_text.text("📄 Extracting resume text...")
            progress_bar.progress(20)
            resume_text = extract_text_from_pdf(uploaded_file)
            
            if not resume_text:
                st.error("Failed to extract text from PDF.")
            else:
                # Step 2: Compare
                status_text.text("🔍 Analyzing resume vs job description...")
                progress_bar.progress(50)
                score, matched_skills, missing_skills = compare_resume_with_job(
                    resume_text, job_desc, skills_list
                )
                
                if score is not None:
                    # Step 3: Calculate scores
                    status_text.text("📊 Computing scores...")
                    progress_bar.progress(75)
                    
                    overall_score = round(score * 100)
                    style_score = calculate_style_score(resume_text)
                    ats_score = calculate_ats_score(resume_text, job_desc)
                    section_score = calculate_section_score(resume_text)
                    skill_score = (
                        round(len(matched_skills) / len(skills_list) * 100) 
                        if skills_list else 0
                    )
                    
                    scores = {
                        "🎯 Skill Match": skill_score,
                        "🧠 Content": overall_score,
                        "🎨 Style": style_score,
                        "⚙️ ATS Compatibility": ats_score,
                        "📄 Sections": section_score
                    }
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    st.divider()
                    animated_gauge("📊 Overall Resume Score", overall_score, "#4CAF50")
                    
                    st.subheader("📈 Detailed Scores")
                    gradients = [
                        ("#f2709c", "#ff9472"),
                        ("#00c6ff", "#0072ff"),
                        ("#f7971e", "#ffd200"),
                        ("#56ab2f", "#a8e063"),
                        ("#e96443", "#904e95")
                    ]
                    
                    for (label, value), (color_from, color_to) in zip(
                        scores.items(), gradients
                    ):
                        custom_animated_bar(label, value, color_from, color_to)
                    
                    # Detailed report button
                    if st.button("🔓 Unlock Full Report", use_container_width=True):
                        st.divider()
                        st.subheader("📝 Detailed Analysis")
                        
                        # Skills analysis
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**✅ Matched Skills ({len(matched_skills)}):**")
                            if matched_skills:
                                for skill in matched_skills:
                                    st.markdown(f"- {skill}")
                            else:
                                st.info("No skills matched. Consider adding related skills.")
                        
                        with col2:
                            st.markdown(f"**❌ Missing Skills ({len(missing_skills)}):**")
                            if missing_skills:
                                for skill in missing_skills:
                                    st.markdown(f"- {skill}")
                            else:
                                st.success("All skills matched!")
                        
                        st.divider()
                        
                        # Formatting feedback
                        if enable_formatting_feedback:
                            st.subheader("🧾 Formatting & Structure")
                            formatting_issues = check_formatting(resume_text)
                            
                            if formatting_issues:
                                st.warning(f"Found {len(formatting_issues)} suggestions:")
                                for issue in formatting_issues:
                                    st.markdown(f"- {issue}")
                            else:
                                st.success("✅ Formatting looks great!")
                        
                        st.divider()
                        st.subheader("💡 Recommendations")
                        st.markdown("""
                        ✅ **To improve your match score:**
                        1. Add missing technical skills that match the job description
                        2. Use quantifiable achievements (e.g., "improved performance by 40%")
                        3. Include relevant keywords from the job posting naturally
                        4. Ensure all major sections (Education, Experience, Skills) are present
                        5. Keep formatting clean with consistent bullet points
                        
                        ✅ **For ATS optimization:**
                        - Use standard section headers
                        - Avoid tables and images (unless necessary)
                        - Use simple, professional formatting
                        - Include relevant keywords naturally throughout
                        - Maintain consistent date formatting
                        
                        ✅ **General Tips:**
                        - Tailor your resume for each job application
                        - Keep it to 1-2 pages
                        - Use power words and action verbs
                        - Quantify your achievements with metrics
                        """)
                else:
                    st.error("❌ Could not compute similarity score. Please try again.")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            progress_bar.empty()
            status_text.empty()

else:
    st.info(
        "👈 **Get started:**\n\n"
        "1. Enter your skills in the sidebar\n"
        "2. Upload your resume (PDF)\n"
        "3. Paste the job description\n"
        "4. Click analyze!"
    )

# ==================== FOOTER ====================
st.divider()
st.caption(
    "⚡ **Performance Optimized** | "
    "🤖 Powered by Sentence Transformers | "
    "📊 Made for job seekers"
)
