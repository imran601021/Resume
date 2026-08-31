import os
from flask import Flask, request, jsonify, render_template

import analyzer
import agent

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit


def serialize_details(details):
    """Convert tuple-based results into JSON-friendly lists of dicts."""
    return {
        'matched':  [{'skill': s, 'score': sc} for s, sc in details['matched']],
        'missing':  [{'skill': s, 'score': sc} for s, sc in details['missing']],
        'partial':  [{'skill': s, 'score': sc} for s, sc in details['partial']],
        'experience': details['experience'],
        'title_match': details['title_match'],
        'keywords': details['keywords'],
        'formatting_issues': details['formatting_issues'],
        'formatting_suggestions': details['formatting_suggestions'],
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        uploaded_file = request.files.get('resume')
        job_desc = request.form.get('job_description', '').strip()
        skills_raw = request.form.get('skills', '').strip()
        use_ai_agent = request.form.get('use_ai_agent', 'false') == 'true'
        job_location = request.form.get('job_location', 'Chennai, Tamil Nadu')

        if not uploaded_file:
            return jsonify({'error': 'No resume file uploaded'}), 400
        if not job_desc:
            return jsonify({'error': 'Job description is required'}), 400

        skills_list = [s.strip().lower() for s in skills_raw.split(',') if s.strip()][:20]
        if not skills_list:
            return jsonify({'error': 'Enter at least one skill'}), 400

        resume_text = analyzer.extract_text_from_pdf(uploaded_file)
        if not resume_text:
            return jsonify({'error': 'Could not extract text from PDF. Ensure it is text-based, not scanned.'}), 400

        scores, details = analyzer.calculate_advanced_scores(resume_text, job_desc, skills_list)

        response = {
            'scores': scores,
            'details': serialize_details(details),
            'verdicts': {
                'overall': analyzer.score_verdict(scores['overall']),
                'skills': analyzer.score_verdict(scores['skills']),
                'content': analyzer.score_verdict(scores['content']),
            }
        }

        if use_ai_agent:
            try:
                feedback, err = agent.generate_ai_feedback(
                    resume_text, job_desc, scores, details, skills_list
                )
                if err:
                    response['ai_feedback'] = {'error': err}
                elif feedback:
                    roles = feedback.get('suggested_roles', [])
                    job_links = agent.build_job_search_links(roles, location=job_location) if roles else []
                    response['ai_feedback'] = {
                        'gap_summary': feedback.get('gap_summary', ''),
                        'improvement_tips': feedback.get('improvement_tips', []),
                        'bullet_rewrites': feedback.get('bullet_rewrites', []),
                        'job_links': job_links,
                    }
            except Exception as e:
                response['ai_feedback'] = {'error': f'AI Agent error: {e}'}

        return jsonify(response)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Something went wrong: {e}'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501)