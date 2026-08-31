const skillsInput = document.getElementById('skills');
const skillsChips = document.getElementById('skills-chips');
const analyzeBtn = document.getElementById('analyze-btn');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const emptyStateEl = document.getElementById('empty-state');
const gaugesRow = document.getElementById('gauges-row');

skillsInput.addEventListener('input', () => {
    const skills = skillsInput.value.split(',').map(s => s.trim()).filter(Boolean);
    skillsChips.innerHTML = skills.map(s => `<span class="chip">${s}</span>`).join('');
});

function showStatus(msg) {
    statusEl.textContent = msg;
    statusEl.classList.remove('hidden');
}
function hideStatus() { statusEl.classList.add('hidden'); }

function gaugeCard(label, value, colorVar) {
    const colors = { green: '#4CAF50', blue: '#2196F3', orange: '#FF9800' };
    return `
    <div class="gauge-card">
        <div class="gauge-label">${label}</div>
        <div class="gauge-circle" style="--gauge-value:${value}; --gauge-color:${colors[colorVar]}">
            <span>${value}</span>
        </div>
    </div>`;
}

function verdictHtml(verdict) {
    const [msg, kind] = verdict;
    return `<div class="verdict ${kind}">${msg}</div>`;
}

analyzeBtn.addEventListener('click', async () => {
    const fileInput = document.getElementById('resume-file');
    const jobDesc = document.getElementById('job-desc').value.trim();
    const skills = skillsInput.value.trim();
    const useAiAgent = document.getElementById('use-ai-agent').checked;
    const jobLocation = document.getElementById('job-location').value.trim();

    if (!fileInput.files[0]) { alert('Please upload a resume PDF'); return; }
    if (!jobDesc) { alert('Please paste a job description'); return; }
    if (!skills) { alert('Please enter at least one skill'); return; }

    const formData = new FormData();
    formData.append('resume', fileInput.files[0]);
    formData.append('job_description', jobDesc);
    formData.append('skills', skills);
    formData.append('use_ai_agent', useAiAgent);
    formData.append('job_location', jobLocation);

    analyzeBtn.disabled = true;
    emptyStateEl.classList.add('hidden');
    resultsEl.classList.add('hidden');
    showStatus('Extracting resume and running semantic analysis — this may take a moment...');

    try {
        const res = await fetch('/api/analyze', { method: 'POST', body: formData });
        const data = await res.json();

        if (!res.ok) {
            showStatus(data.error || 'Something went wrong.');
            analyzeBtn.disabled = false;
            return;
        }

        hideStatus();
        renderResults(data);
        resultsEl.classList.remove('hidden');
    } catch (e) {
        showStatus('Network error: ' + e.message);
    } finally {
        analyzeBtn.disabled = false;
    }
});

function renderResults(data) {
    const { scores, details, verdicts, ai_feedback } = data;

    gaugesRow.innerHTML = `
        <div>${gaugeCard('Overall Match', scores.overall, 'green')}${verdictHtml(verdicts.overall)}</div>
        <div>${gaugeCard('Skill Match', scores.skills, 'blue')}${verdictHtml(verdicts.skills)}</div>
        <div>${gaugeCard('Content Match', scores.content, 'orange')}${verdictHtml(verdicts.content)}</div>
    `;

    document.getElementById('tab-overview').innerHTML = `
        <h3>Score Breakdown</h3>
        <div class="metric-row">
            <div class="metric-box"><div class="value">${scores.content}%</div><div class="label">Semantic</div></div>
            <div class="metric-box"><div class="value">${scores.skills}%</div><div class="label">Skills</div></div>
            <div class="metric-box"><div class="value">${scores.keywords.toFixed(0)}%</div><div class="label">Keywords</div></div>
            <div class="metric-box"><div class="value">${scores.experience}%</div><div class="label">Experience</div></div>
            <div class="metric-box"><div class="value">${scores.formatting}%</div><div class="label">Formatting</div></div>
        </div>
        <h3>Quick Summary</h3>
        <div class="info-box info"><strong>Matched Skills</strong> ${details.matched.length}</div>
        <div class="info-box warning"><strong>Missing Skills</strong> ${details.missing.length}</div>
        <div class="info-box info"><strong>Experience</strong> ${details.experience ? details.experience.resume_years + '+ years' : 'Not found'}</div>
    `;

    document.getElementById('tab-skills').innerHTML = `
        <div class="col-3">
            <div class="col"><h4>Matched Skills</h4><ul>${details.matched.map(m => `<li>${m.skill} (${(m.score*100).toFixed(0)}%)</li>`).join('') || '<li>None matched</li>'}</ul></div>
            <div class="col"><h4>Missing Skills</h4><ul>${details.missing.map(m => `<li>${m.skill}</li>`).join('') || '<li>All matched!</li>'}</ul></div>
            <div class="col"><h4>Partial Matches</h4><ul>${details.partial.map(m => `<li>${m.skill} (${(m.score*100).toFixed(0)}%)</li>`).join('') || '<li>None</li>'}</ul></div>
        </div>
    `;

    document.getElementById('tab-keywords').innerHTML = `
        <div class="col-3">
            <div class="col"><h4>Found: ${details.keywords.found}/${details.keywords.total}</h4><ul>${details.keywords.found_keywords.map(k => `<li>✓ ${k}</li>`).join('')}</ul></div>
            <div class="col"><h4>Missing: ${details.keywords.missing_keywords.length}</h4><ul>${details.keywords.missing_keywords.map(k => `<li>✗ ${k}</li>`).join('')}</ul></div>
        </div>
    `;

    document.getElementById('tab-experience').innerHTML = `
        <h4>Job Title Match</h4>
        <p>Position: <strong>${details.title_match.job_title}</strong> — Match: <strong>${details.title_match.match_percentage}%</strong></p>
        ${details.experience ? `<h4>Experience</h4><p>${details.experience.resume_years} vs ${details.experience.required_years} required years — ${details.experience.match ? 'On track' : 'Below requirement'}</p>` : ''}
    `;

    document.getElementById('tab-formatting').innerHTML = `
        ${details.formatting_issues.map(i => `<div class="info-box warning">${i}</div>`).join('')}
        ${details.formatting_suggestions.map(s => `<div class="info-box info">${s}</div>`).join('')}
        <h4>Recommendations</h4>
        ${buildRecommendations(scores, details)}
    `;

    const aiSection = document.getElementById('ai-agent-section');
    if (ai_feedback) {
        if (ai_feedback.error) {
            aiSection.innerHTML = `<div class="ai-section"><h3>AI Agent</h3><p>${ai_feedback.error}</p></div>`;
        } else {
            aiSection.innerHTML = `
                <div class="ai-section">
                    <h3>AI Agent — Deeper Feedback & Job Suggestions</h3>
                    <p class="caption">AI-generated analysis — treat as a second opinion alongside the scores above.</p>
                    <p><strong>Why you're lagging:</strong> ${ai_feedback.gap_summary}</p>
                    ${ai_feedback.improvement_tips.length ? `<h4>Tips</h4><ul>${ai_feedback.improvement_tips.map(t => `<li>${t}</li>`).join('')}</ul>` : ''}
                    ${ai_feedback.bullet_rewrites.length ? `<h4>Suggested rewrites</h4>` + ai_feedback.bullet_rewrites.map(r => `<p><s>${r.original}</s><br>→ <strong>${r.improved}</strong></p>`).join('') : ''}
                    ${ai_feedback.job_links.length ? `<h4>Roles worth exploring</h4><ul>${ai_feedback.job_links.map(jl => `<li><strong>${jl.role}</strong> — <a href="${jl.indeed}" target="_blank">Indeed</a> · <a href="${jl.linkedin}" target="_blank">LinkedIn</a> · <a href="${jl.naukri}" target="_blank">Naukri</a></li>`).join('')}</ul>` : ''}
                </div>`;
        }
    } else {
        aiSection.innerHTML = '';
    }
}

function buildRecommendations(scores, details) {
    const recs = [];
    if (scores.overall < 70) recs.push("Overall match is low — Rewrite sections to better align with the JD");
    if (details.missing.length) recs.push(`Add missing skills — ${details.missing.length} required skills not found`);
    if (scores.content < 70) recs.push("Improve content — Use more job description keywords naturally");
    if (scores.keywords < 60) recs.push("Increase keyword density — Add specific technical terms");
    if (details.experience && !details.experience.match) recs.push(`Experience gap — You have ${details.experience.resume_years} yrs; role needs ${details.experience.required_years}`);
    if (scores.formatting < 70) recs.push("Improve formatting — Better structure helps ATS scanning");
    if (!recs.length) return '<div class="info-box success">Your resume looks great for this position!</div>';
    return recs.map(r => `<div class="info-box info">${r}</div>`).join('');
}

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    });
});