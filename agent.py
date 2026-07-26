"""
AI Agent module for Resume Analyzer.

Uses Groq's free API to run an open-weight LLM (OpenAI GPT-OSS 120B) on top of the
scores already computed by app.py. It does three things:

  1. Explains *why* the resume is lagging (gap analysis)
  2. Gives concrete, actionable tips — including rewritten bullet points
  3. Suggests job titles/roles to search for, based on the resume

No paid API is required. Groq's free tier is generous enough for personal
project use. Get a free key at https://console.groq.com/keys and set it as
the GROQ_API_KEY environment variable (or in Streamlit secrets).
"""

import json
import os
import time
import urllib.parse

from groq import Groq

MODEL = "openai/gpt-oss-120b"  # open-weight model (Apache 2.0), free on Groq
# Note: llama-3.3-70b-versatile was deprecated by Groq (June 2026) in favor
# of this model. If you hit a "model_decommissioned" error, check
# https://console.groq.com/docs/deprecations and update MODEL accordingly.


def _get_client():
    """Create a Groq client from env var or Streamlit secrets. Returns None if no key."""
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        return None

    api_key = api_key.strip()  # guard against trailing newline/whitespace from copy-paste
    if not api_key:
        return None

    return Groq(api_key=api_key)


def _build_prompt(resume_text, job_desc, scores, details, skills_list):
    missing_skills = [s for s, _ in details.get("missing", [])]
    matched_skills = [s for s, _ in details.get("matched", [])]
    missing_keywords = details.get("keywords", {}).get("missing_keywords", [])

    return f"""You are a career coach reviewing a resume against a job description.
You are given scores that were already calculated by a separate scoring engine —
do not recompute them, just use them as context to explain and advise.

SCORES (0-100):
- Overall match: {scores.get('overall')}
- Content/semantic match: {scores.get('content')}
- Skill match: {scores.get('skills')}
- Keyword density: {scores.get('keywords')}
- Experience match: {scores.get('experience')}
- Formatting: {scores.get('formatting')}

Skills the candidate listed but that are MISSING from the resume text: {missing_skills}
Skills the candidate listed that ARE present in the resume: {matched_skills}
Job-description keywords missing from the resume: {missing_keywords}

RESUME TEXT (may be truncated):
{resume_text[:4000]}

JOB DESCRIPTION:
{job_desc[:2000]}

Respond with ONLY a JSON object (no markdown fences, no commentary) in exactly this shape:
{{
  "gap_summary": "2-3 sentence plain-English explanation of the main reasons this resume is lagging behind what the JD wants",
  "improvement_tips": ["short, specific, actionable tip", "..."],
  "bullet_rewrites": [
    {{"original": "a weak bullet point pulled from the resume text if one exists, else a generic weak example", "improved": "a stronger rewritten version tailored to the JD"}}
  ],
  "suggested_roles": ["job title 1", "job title 2", "job title 3"]
}}

Rules:
- improvement_tips: 4-6 items, each one specific and actionable (not generic advice like "improve your resume").
- bullet_rewrites: 2-3 items, based on real content from the resume where possible.
- suggested_roles: 3-5 realistic job titles this candidate's skills/experience actually fit, not just the JD title repeated.
"""


def generate_ai_feedback(resume_text, job_desc, scores, details, skills_list):
    """
    Calls the AI agent and returns a dict with keys:
    gap_summary, improvement_tips, bullet_rewrites, suggested_roles.
    Returns None (with an error message) if no API key is configured.
    """
    client = _get_client()
    if client is None:
        return None, (
            "No GROQ_API_KEY found. Get a free key at "
            "https://console.groq.com/keys and set it as an environment "
            "variable or in Streamlit secrets to enable the AI agent."
        )

    prompt = _build_prompt(resume_text, job_desc, scores, details, skills_list)

    last_error = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1600,
                reasoning_effort="low",  # this task doesn't need deep reasoning; keeps it fast & within free-tier token limits
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            data = _parse_json_response(raw)
            if data is None:
                return None, "AI agent error: model returned a response that wasn't valid JSON."
            return data, None
        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))  # brief backoff before retrying
                continue

    return None, f"AI agent error: {_describe_error(last_error)}"


def _describe_error(e):
    """Dig past SDK wrapper exceptions to show the real underlying cause."""
    parts = [f"{type(e).__name__}: {e}"]
    cause = e.__cause__
    seen = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        parts.append(f"caused by {type(cause).__name__}: {cause}")
        cause = cause.__cause__
    return " | ".join(parts)


def _parse_json_response(raw):
    """Parse the model's JSON output, tolerating stray text or code fences."""
    if not raw:
        return None
    text = raw.strip()
    # Strip markdown code fences if the model added them despite instructions
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: grab the first {...} block in the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def build_job_search_links(suggested_roles, location="Chennai, Tamil Nadu"):
    """
    Turns suggested role titles into ready-to-click job search links.
    No job-search API needed, so this stays completely free — it just
    builds correctly-formatted search URLs.
    """
    links = []
    loc_q = urllib.parse.quote(location)
    for role in suggested_roles:
        role_q = urllib.parse.quote(role)
        links.append({
            "role": role,
            "indeed": f"https://in.indeed.com/jobs?q={role_q}&l={loc_q}",
            "linkedin": f"https://www.linkedin.com/jobs/search/?keywords={role_q}&location={loc_q}",
            "naukri": f"https://www.naukri.com/{role_q.replace('%20', '-')}-jobs-in-{loc_q.replace('%20', '-')}",
        })
    return links
