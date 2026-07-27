# src/report_generator.py
import datetime
import pytz
import json
import ollama  

class AIReporter:
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name

    def generate_report(self, transcript_text, vision_metrics, start_time_utc=None, duration_seconds=1800):
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

        # Define JSON Schema to force Ollama to return valid structural JSON
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
                "class_summary": {"type": "string"},
                "key_notes": {"type": "string"}
            },
            "required": [
                "english_usage_pct", "other_language_pct", "english_comm_score",
                "english_comm_feedback", "technical_relevance_score",
                "technical_coverage_feedback", "pedagogical_effectiveness",
                "class_summary", "key_notes"
            ],
            "technical_relevance_score": {
    "type": "number",
    "description": "Score between 0.0 and 10.0 evaluating technical depth."
}
        }

        prompt = f"""
You are an expert faculty evaluation AI analyzing a classroom lecture transcript.

TRANSCRIPT:
"{transcript_text}"

BOARD METRICS:
- Visual Board Snapshots Extracted: {boards_count}

Analyze the transcript and evaluate the lecturer's performance. Return valid scores, communication feedback, bullet points for class summary, and key notes.
"""

        try:
            # Force Ollama to structure its response according to JSON Schema
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                format=json_schema
            )
            
            raw_output = response['message']['content'].strip()
            parsed_metrics = json.loads(raw_output)
            
            # Inject IST metadata
            parsed_metrics["session_date"] = session_date_str
            parsed_metrics["start_time"] = session_start_str
            parsed_metrics["end_time"] = session_end_str
            parsed_metrics["boards_count"] = boards_count
            parsed_metrics["transcript"] = transcript_text
            
            return parsed_metrics

        except Exception as e:
            # Print actual error to terminal console for debugging
            print(f"[AIReporter Error] Failed to generate LLM report: {e}")
            
            return {
                "english_usage_pct": 0,
                "other_language_pct": 0,
                "english_comm_score": 0.0,
                "english_comm_feedback": f"⚠️ LLM Processing Error: {e}",
                "technical_relevance_score": 0.0,
                "technical_coverage_feedback": "Unable to evaluate technical content due to model error.",
                "pedagogical_effectiveness": "Analysis unavailable.",
                "class_summary": "• Could not generate summary. Check Ollama terminal logs.",
                "key_notes": "### Error\nFailed to parse model response.",
                "session_date": session_date_str,
                "start_time": session_start_str,
                "end_time": session_end_str,
                "boards_count": boards_count,
                "transcript": transcript_text
            }