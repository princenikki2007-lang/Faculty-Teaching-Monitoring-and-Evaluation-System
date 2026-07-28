import streamlit as st
import tempfile
import os
import shutil
from pathlib import Path
from datetime import datetime
import pytz

# Import your underlying modules
from src.report_generator import AIReporter
from src.audio_processor import AudioProcessor
from src.vision_analyzer import VisionProcessor

st.set_page_config(
    page_title="Faculty Teaching Assessment Dashboard",
    page_icon="🎓",
    layout="wide"
)

# --- CUSTOM CSS FOR METRICS & UI POLISH ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Faculty Teaching Assessment & Analytics")
st.markdown("Automated AI evaluation of classroom delivery, board usage, language breakdown, and technical depth.")

# --- SIDEBAR: VIDEO UPLOAD & SETTINGS ---
st.sidebar.header("📹 Video Upload")

uploaded_video = st.sidebar.file_uploader(
    "Upload Classroom Recording", 
    type=["mp4", "avi", "mov", "mkv"],
    help="Select a lecture video file to process audio and board visual extraction."
)

st.sidebar.divider()
st.sidebar.header("⚙️ Model Configuration")

selected_model = st.sidebar.selectbox(
    "Local Ollama Model", 
    ["llama3.2", "mistral", "qwen2.5-coder"],
    index=0
)

# Store video temporary path across runs
video_path = None

if uploaded_video is not None:
    # Save uploaded bytes to temporary file for moviepy / cv2 / whisper
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_video.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path = tmp_file.name

    st.sidebar.video(uploaded_video)
    st.sidebar.success(f"Loaded File: {uploaded_video.name}")
else:
    st.sidebar.info("Upload a video to enable assessment.")


# --- MAIN ACTION BUTTON ---
run_button = st.button(
    "🚀 Run Full Session Assessment", 
    type="primary", 
    disabled=(uploaded_video is None)
)

if run_button and video_path is not None:
    # 1. Prepare directory for fresh extracted board images
    boards_output_dir = "data/extracted_boards"
    if os.path.exists(boards_output_dir):
        shutil.rmtree(boards_output_dir)
    os.makedirs(boards_output_dir, exist_ok=True)

    start_time_utc = datetime.now(pytz.utc)

    # 2. Instantiate local AI engines
    audio_proc = AudioProcessor()
    vision_proc = VisionProcessor(output_dir=boards_output_dir)
    reporter = AIReporter(model_name=selected_model)

    # STEP A: Extract REAL audio transcript using Whisper GPU
    with st.spinner("🎙️ Transcribing audio using local Whisper..."):
        audio_result = audio_proc.transcribe(video_path)
        real_transcript = audio_result["text"]  # <-- Real text from video!

    # STEP B: Process REAL frames using OpenCV
    with st.spinner("🖼️ Analyzing video frames & board content..."):
        real_vision_metrics = vision_proc.analyze_video(video_path) # <-- Real board snapshot count!

    # Calculate actual video duration (seconds) based on frame count
    estimated_duration = real_vision_metrics.get("frames_analyzed", 0) * vision_proc.sample_rate / 30

    # STEP C: Send real audio transcript & board metrics to Ollama
    with st.spinner(f"🧠 Generating dynamic pedagogical metrics with {selected_model}..."):
        data = reporter.generate_report(
            transcript_text=real_transcript,       # <-- Pass real transcript
            vision_metrics=real_vision_metrics,     # <-- Pass real vision metrics
            start_time_utc=start_time_utc,
            duration_seconds=estimated_duration
        )

    # Cleanup temporary uploaded video
    if os.path.exists(video_path):
        os.remove(video_path)

    st.success("Analysis Completed Successfully!")
    st.divider()

    # --- TOP LEVEL KPI METRICS ---
    st.subheader("📌 Executive Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🕒 Session Timing (IST)",
            value=f"{data['start_time']} - {data['end_time']}",
            delta=data['session_date']
        )

    with col2:
    # Safely get the score
     raw_tech = float(data.get('technical_relevance_score', 0))
    
    # If the LLM outputs a number out of 10 (like 8.0), multiply it by 10 to make it 80.
     tech_score = raw_tech * 10 if raw_tech <= 10 else raw_tech
    
     st.metric(
        label="🎯 Technical Relevance Score",
        value=f"{int(tech_score)} / 100",
        delta="Very High" if tech_score >= 80 else "Needs Review",
        delta_color="off"  # Removes the confusing green up-arrow
     )

     with col3: 
        raw_comm = float(data.get('english_comm_score', 0))
    
        # Scale out of 10 up to 100
        comm_score = raw_comm * 10 if raw_comm <= 10 else raw_comm
    
        st.metric(
        label="🗣️ Communication Score",
        value=f"{int(comm_score)} / 100",
        delta="Proficient" if comm_score >= 65 else "Need to improve",
        delta_color="off"  # Removes the confusing green up-arrow
     )

     with col4:
        st.metric(
            label="🖼️ Board Snapshots Captured",
            value=f"{data['boards_count']} Snapshots",
            delta="Active Board Use" if data['boards_count'] > 0 else "No Board Content"
        )

     st.divider()

    # --- TWO COLUMN DETAILED BREAKDOWN ---
    col_left, col_right = st.columns(2)

    with col_left:
            st.subheader("🌐 Language Usage & Communication")
            
            eng_pct = float(data.get("english_usage_pct", 80))
            oth_pct = float(data.get("other_language_pct", 20))

            st.write(f"**English Usage:** {int(eng_pct)}%")
            # Divide by 100 to scale percentage (0-100) down to (0.0-1.0)
            st.progress(min(max(eng_pct / 100.0, 0.0), 1.0))

            st.write(f"**Other Language / Vernacular Usage:** {int(oth_pct)}%")
            st.progress(min(max(oth_pct / 100.0, 0.0), 1.0))

            st.info(f"**Communication Feedback:** {data['english_comm_feedback']}")

    with col_right:
            st.subheader("🔬 Technical Coverage & Pedagogy")
            
            tech_score = float(data.get("technical_relevance_score", 8.0))
            
            st.write(f"**Topic Focus & Depth Score:** {tech_score}/10")
            # Divide score by 10.0 to scale (0-10) down to (0.0-1.0)
            st.progress(min(max(tech_score / 10.0, 0.0), 1.0))

            st.markdown(f"**Technical Alignment:** {data['technical_coverage_feedback']}")
            st.markdown(f"**Pedagogical Style:** {data['pedagogical_effectiveness']}")

    st.divider()

   # --- EXTRACTED BOARD SNAPSHOTS GALLERY ---
    st.subheader("🖼️ Extracted Clean Board Snapshots")
    board_paths = real_vision_metrics.get("saved_paths", [])

    if board_paths:
        # Render extracted images in a 3-column grid
        cols = st.columns(3)
        for idx, img_path in enumerate(board_paths):
            with cols[idx % 3]:
                st.image(img_path, caption=f"Board Snapshot {idx + 1}", width='stretch')
    else:
        st.info("No distinct board writing changes were detected during this session.")

    # --- TABBED DETAIL SECTION ---
    tab_notes, tab_summary, tab_transcript = st.tabs([
        "📝 Class Notes & Key Takeaways", 
        "📋 Lecture Summary", 
        "📜 Full Audio Transcript"
    ])

    with tab_notes:
        st.markdown(data['key_notes'])

    with tab_summary:
        st.markdown(data['class_summary'])

    with tab_transcript:
        st.code(data['transcript'], language="text")