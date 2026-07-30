import datetime
import pytz
import json
import ollama  

class AIReporter:
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name

    def generate_report(self, transcript_text, vision_metrics, syllabus_text="", start_time_utc=None, duration_seconds=1800):
        ist = pytz.timezone('Asia/Kolkata')
        if start_time_utc is None:
            start_time_ist = datetime.datetime.now(ist)
        else:
            start_time_ist = start_time_utc.astimezone(ist)
            
        end_time_ist = start_time_ist + datetime.timedelta(seconds=duration_seconds)

        session_start_str = start_time_ist.strftime("%I:%M %p")
        session_end_str = end_time_ist.strftime("%I:%M %p")
        session_date_str = start_time_ist.strftime("%Y-%m-%d")

        boards_count = vision_metrics.get("extracted_boards_count", 0)

        if not transcript_text or len(transcript_text.strip()) < 10:
            transcript_text = "No intelligible speech was detected in this video file."

        syllabus_prompt_part = ""
        if syllabus_text:
            syllabus_prompt_part = f"\nREFERENCE SYLLABUS / PPT SLIDES:\n\"{syllabus_text[:4000]}\"\n"

        json_schema = {
            "type": "object",
            "properties": {
                "english_usage_pct": {"type": "integer"},
                "other_language_pct": {"type": "integer"},
                "english_comm_score": {"type": "number"},
                "english_comm_feedback": {"type": "string"},
                "technical_relevance_score": {"type": "number"},
                "technical_coverage_feedback": {"type": "string"},
                "pedagogical_effectiveness": {"type": "string"},
                "syllabus_coverage_pct": {"type": "integer"},
                "missing_syllabus_topics": {"type": "string"},
                "out_of_syllabus_topics": {"type": "string"},
                "class_summary": {"type": "string"},
                "key_notes": {"type": "string"}
            },
            "required": [
                "english_usage_pct", "other_language_pct", "english_comm_score",
                "english_comm_feedback", "technical_relevance_score",
                "technical_coverage_feedback", "pedagogical_effectiveness",
                "syllabus_coverage_pct", "missing_syllabus_topics",
                "out_of_syllabus_topics", "class_summary", "key_notes"
            ]
        }

        prompt = f"""
You are an uncompromising academic audit assistant. 
Compare the provided Lecture Transcript against the Reference Syllabus/PPT text.

CRITICAL DIRECTIVES:
1. First, check subject-matter alignment. If the lecture covers Physics/Forces and the reference PPT is about an unrelated topic, the coverage score MUST BE 0%.
2. Do NOT award points for matching generic words like "Introduction", "Overview", "Part 1", or "Slide".

Syllabus Content:
\"\"\"{syllabus_text[:2000]}\"\"\"

Lecture Transcript:
\"\"\"{transcript_text[:2000]}\"\"\"

Return ONLY valid JSON:
{{
    "coverage_score": <integer 0-100>,
    "reasoning": "<explanation>",
    "matched_topics": [],
    "missing_topics": []
}}
"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                format=json_schema
            )
            
            raw_output = response['message']['content'].strip()
            parsed_metrics = json.loads(raw_output)
            
            parsed_metrics["session_date"] = session_date_str
            parsed_metrics["start_time"] = session_start_str
            parsed_metrics["end_time"] = session_end_str
            parsed_metrics["boards_count"] = boards_count
            parsed_metrics["transcript"] = transcript_text
            
            return parsed_metrics

        except Exception as e:
            print(f"[AIReporter Error] Failed to generate LLM report: {e}")
            
            return {
                "english_usage_pct": 0,
                "other_language_pct": 0,
                "english_comm_score": 0,
                "english_comm_feedback": f"⚠️ LLM Processing Error: {e}",
                "technical_relevance_score": 0,
                "technical_coverage_feedback": "Unable to evaluate technical content due to model error.",
                "pedagogical_effectiveness": "Interactive Lecture",
                "syllabus_coverage_pct": 0,
                "missing_syllabus_topics": "• Analysis unavailable.",
                "out_of_syllabus_topics": "• None detected.",
                "class_summary": "• Could not generate summary. Check Ollama logs.",
                "key_notes": "### Error\nFailed to parse model response.",
                "session_date": session_date_str,
                "start_time": session_start_str,
                "end_time": session_end_str,
                "boards_count": boards_count,
                "transcript": transcript_text
            }