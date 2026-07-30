import streamlit as st
import tempfile
import os
import shutil
from pathlib import Path
from datetime import datetime
import pytz

# Import core modules
from src.report_generator import AIReporter
from src.audio_processor import AudioProcessor
from src.vision_analyzer import VisionProcessor
from src.rag_verifier import SyllabusRAGVerifier

st.set_page_config(
    page_title="Faculty Teaching Quality & Syllabus RAG System",
    page_icon="🎓",
    layout="wide"
)

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

st.title("🎓 Faculty Teaching Quality & Syllabus Audit System")
st.markdown("Automated AI evaluation of classroom delivery, board usage, speech cadence, and RAG syllabus verification.")

# --- SIDEBAR: INPUTS & CONFIGS ---
st.sidebar.header("📹 Video Upload")
uploaded_video = st.sidebar.file_uploader(
    "Upload Classroom Recording", 
    type=["mp4", "avi", "mov", "mkv"],
    help="Select a lecture video file to process audio and board visual extraction."
)

st.sidebar.divider()
st.sidebar.header("📚 Syllabus / Slide Upload (RAG)")
uploaded_syllabus = st.sidebar.file_uploader(
    "Upload PPTX or PDF Syllabus",
    type=["pdf", "pptx", "ppt"],
    help="Upload official course syllabus or PPT slides to verify lecture topic coverage."
)

st.sidebar.divider()
st.sidebar.header("⚙️ Model Configuration")
selected_model = st.sidebar.selectbox(
    "Local Ollama Model", 
    ["llama3.2", "mistral", "qwen2.5-coder"],
    index=0
)

video_path = None
syllabus_path = None

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_video.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path = tmp_file.name
    st.sidebar.video(uploaded_video)
    st.sidebar.success(f"Video Loaded: {uploaded_video.name}")
else:
    st.sidebar.info("Upload a video to enable assessment.")

if uploaded_syllabus is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_syllabus.name.split('.')[-1]}") as tmp_syl:
        tmp_syl.write(uploaded_syllabus.read())
        syllabus_path = tmp_syl.name
    st.sidebar.success(f"Syllabus Loaded: {uploaded_syllabus.name}")

# --- MAIN ACTION BUTTON ---
run_button = st.button(
    "🚀 Run Full Session Assessment", 
    type="primary", 
    disabled=(uploaded_video is None)
)

if run_button and video_path is not None:
    boards_output_dir = "data/extracted_boards"
    if os.path.exists(boards_output_dir):
        shutil.rmtree(boards_output_dir)
    os.makedirs(boards_output_dir, exist_ok=True)

    start_time_utc = datetime.now(pytz.utc)

    # Instantiate local AI engines
    audio_proc = AudioProcessor()
    vision_proc = VisionProcessor(output_dir=boards_output_dir)
    rag_verifier = SyllabusRAGVerifier()
    reporter = AIReporter(model_name=selected_model)

    # 1. Parse Syllabus (RAG) if provided
    syllabus_text = ""
    if syllabus_path:
        with st.spinner("📖 Extracting syllabus content for RAG verification..."):
            syllabus_text = rag_verifier.extract_syllabus_text(syllabus_path)

    # 2. Extract Transcript
    with st.spinner("🎙️ Transcribing audio using local Whisper..."):
        audio_result = audio_proc.transcribe(video_path)
        real_transcript = audio_result["text"]
        pacing_metrics = audio_proc.calculate_pacing(audio_result["segments"])

    # 3. Analyze Video Frames
    with st.spinner("🖼️ Analyzing video frames & board content..."):
        real_vision_metrics = vision_proc.analyze_video(video_path)

    estimated_duration = real_vision_metrics.get("frames_analyzed", 0) * vision_proc.sample_rate / 30

    # 4. Generate AI Report
    with st.spinner(f"🧠 Generating pedagogical metrics with {selected_model}..."):
        data = reporter.generate_report(
            transcript_text=real_transcript,
            vision_metrics=real_vision_metrics,
            syllabus_text=syllabus_text,
            start_time_utc=start_time_utc,
            duration_seconds=estimated_duration
        )

    # Cleanup temporary files
    if os.path.exists(video_path):
        os.remove(video_path)
    if syllabus_path and os.path.exists(syllabus_path):
        os.remove(syllabus_path)

    st.success("Analysis Completed Successfully!")
    st.divider()

    # --- TOP LEVEL KPI METRICS ---
    st.subheader("📌 Executive Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="🕒 Session Timing (IST)",
            value=f"{data['start_time']} - {data['end_time']}",
            delta=data['session_date']
        )

    with col2:
        raw_tech = float(data.get('technical_relevance_score', 0))
        tech_score = raw_tech * 10 if raw_tech <= 10 else raw_tech
        st.metric(
            label="🎯 Technical Score",
            value=f"{int(tech_score)} / 100",
            delta="High Depth" if tech_score >= 80 else "Needs Review",
            delta_color="off"
        )

    with col3: 
        raw_comm = float(data.get('english_comm_score', 0))
        comm_score = raw_comm * 10 if raw_comm <= 10 else raw_comm
        st.metric(
            label="🗣️ Communication",
            value=f"{int(comm_score)} / 100",
            delta="Proficient" if comm_score >= 65 else "Improvement Needed",
            delta_color="off"
        )

    with col4:
        syl_cov = int(data.get("syllabus_coverage_pct", 0))
        st.metric(
            label="📚 Syllabus Coverage",
            value=f"{syl_cov}%" if uploaded_syllabus else "N/A",
            delta="Matched against PPT" if uploaded_syllabus else "No PPT Uploaded",
            delta_color="off"
        )

    with col5:
        st.metric(
            label="⏱️ Speaking Cadence",
            value=f"{pacing_metrics['wpm']} WPM",
            delta=pacing_metrics['status'],
            delta_color="off"
        )

    st.divider()

    # --- DETAILED BREAKDOWN ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🌐 Language Usage & Cadence")
        eng_pct = float(data.get("english_usage_pct", 80))
        oth_pct = float(data.get("other_language_pct", 20))

        st.write(f"**English Usage:** {int(eng_pct)}%")
        st.progress(min(max(eng_pct / 100.0, 0.0), 1.0))

        st.write(f"**Vernacular Usage:** {int(oth_pct)}%")
        st.progress(min(max(oth_pct / 100.0, 0.0), 1.0))

        st.info(f"**Communication Feedback:** {data.get('english_comm_feedback', 'N/A')}")

    with col_right:
        st.subheader("🔬 Technical Coverage & Pedagogy")
        
        # Clean Pedagogical Style String (Fixes '],' UI artifact)
        raw_style = data.get("pedagogical_effectiveness", "Standard Lecture")
        if isinstance(raw_style, list):
            style_text = ", ".join(str(item) for item in raw_style if item)
        else:
            style_text = str(raw_style)
        style_text = style_text.replace("[", "").replace("]", "").replace('"', '').replace("'", '').strip(" ,")
        if not style_text:
            style_text = "Standard Lecture"

        st.markdown(f"**Technical Alignment:** {data.get('technical_coverage_feedback', 'N/A')}")
        st.markdown(f"**Pedagogical Style:** {style_text}")

    st.divider()

    # --- BOARD SNAPSHOTS ---
    st.subheader("🖼️ Extracted Board Snapshots")
    board_paths = real_vision_metrics.get("saved_paths", [])

    if board_paths:
        cols = st.columns(3)
        for idx, img_path in enumerate(board_paths):
            with cols[idx % 3]:
                st.image(img_path, caption=f"Board Snapshot {idx + 1}", use_container_width=True)
    else:
        st.info("No distinct board writing changes were detected during this session.")

    # --- TABBED DETAIL SECTION ---
    tab_rag, tab_notes, tab_summary, tab_transcript = st.tabs([
        "📚 Syllabus & RAG Audit",
        "📝 Class Notes & Key Takeaways", 
        "📋 Lecture Summary", 
        "📜 Full Audio Transcript"
    ])

    with tab_rag:
        if uploaded_syllabus:
            st.markdown(f"### Syllabus Alignment Score: {data.get('syllabus_coverage_pct', 0)}%")
            st.markdown("#### ❌ Missing Syllabus Topics")
            st.markdown(data.get('missing_syllabus_topics', '• None identified.'))
            st.markdown("#### ⚠️ Out-of-Syllabus / Extra Topics Covered")
            st.markdown(data.get('out_of_syllabus_topics', '• None identified.'))
        else:
            st.info("Upload a syllabus or PPT slide file in the sidebar to enable automated curriculum verification.")

    with tab_notes:
        st.markdown(data.get('key_notes', ''))

    with tab_summary:
        st.markdown(data.get('class_summary', ''))

    with tab_transcript:
        st.code(data.get('transcript', ''), language="text")