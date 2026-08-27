import os
import time
import random
import shutil
import gradio as gr
from faster_whisper import WhisperModel
import ollama
import pandas as pd

# 1. Directories & AI Setup
AUDIO_STORAGE_DIR = "candidate_audios"
os.makedirs(AUDIO_STORAGE_DIR, exist_ok=True)
CSV_FILE = "extempore_evaluations.csv"

print("Loading Whisper speech engine...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("Ready!")

# 2. Selected Professional Topics
EXTEMPORE_TOPICS = [
    "Describe your ideal professional workspace setup.",
    "Walk me through organizing your daily tasks.",
    "The most useful software you use daily.",
    "How you prepare for an important meeting.",
    "Working from home versus the office environment.",
    "Describe your preferred style of team communication."
]

def get_initial_topic():
    return random.choice(EXTEMPORE_TOPICS)

# 3. Topic Refresh Logic (Strict 2-Skip Limit)
def refresh_topic(current_topic, skips_left):
    if skips_left <= 0:
        return current_topic, skips_left, gr.update(interactive=False), "⚠️ You have reached the maximum limit of 2 topic changes."
    
    available_topics = [t for t in EXTEMPORE_TOPICS if t != current_topic]
    new_topic = random.choice(available_topics)
    new_skips = skips_left - 1
    
    status_msg = f"Topic refreshed. You have {new_skips} change(s) remaining." if new_skips > 0 else "Topic refreshed. No changes remaining."
    btn_state = gr.update(interactive=(new_skips > 0))
    
    return new_topic, new_skips, btn_state, status_msg

# 4. Strict 9-Point Grading Rubric
SYSTEM_PROMPT = """
You are a strict, expert English communication assessor evaluating a 2-minute Extempore speech.
You will be provided with the candidate's transcript, the assigned topic, and a fluency report showing speech pace and pauses.

Evaluate the candidate strictly on a scale of 1 to 10 across each of the following 9 parameters:
- Spoken English
- Grammar
- Pronunciation (Penalize if transcript contains broken, fragmented, or phonetically garbled words)
- Sentence Formation
- Vocabulary
- Fluency (Evaluate pacing and lack of long, awkward pauses)
- Neutral Accent (Penalize if speech recognition struggled with heavy localized inflections)
- Confidence (Evaluate based on continuous assertive structure and absence of filler hesitation)
- Ability to think on an unknown topic (Evaluate relevance, flow of thought, and depth on the given prompt)

Output EXACTLY in this format:

Spoken English: [Score]/10
Grammar: [Score]/10
Pronunciation: [Score]/10
Sentence Formation: [Score]/10
Vocabulary: [Score]/10
Fluency: [Score]/10
Neutral Accent: [Score]/10
Confidence: [Score]/10
Ability to think on an unknown topic: [Score]/10

Overall Extempore Rating: [Average Score]/10

Detailed Feedback: [2-3 concise sentences detailing specific strengths, grammatical errors, or structural pacing issues.]
"""

def evaluate_candidate(name, email, phone, current_topic, audio_filepath):
    # Validation checks
    if not name.strip() or not email.strip() or not phone.strip():
        return "⚠️ Error: Please fill in your Name, Email, and Phone Number.", "", "", gr.update()
    if not audio_filepath:
        return "⚠️ Error: No audio recording detected. Please record your answer.", "", "", gr.update()

    # Save audio permanently using a clean identifier
    clean_identifier = email.replace("@", "_").replace(".", "_").strip()
    saved_audio_filename = f"{clean_identifier}_{phone.strip()}_{int(time.time())}.wav"
    saved_audio_path = os.path.join(AUDIO_STORAGE_DIR, saved_audio_filename)
    shutil.copy(audio_filepath, saved_audio_path)

    # Transcription & Speech Timing
    segments, info = whisper_model.transcribe(audio_filepath, beam_size=5)
    transcript = ""
    total_speech_time = 0.0
    for segment in segments:
        transcript += segment.text + " "
        total_speech_time += (segment.end - segment.start)

    transcript = transcript.strip()
    total_duration = info.duration

    # Fluency Metrics
    if total_duration > 0:
        active_ratio = (total_speech_time / total_duration) * 100
        pause_ratio = max(0.0, 100.0 - active_ratio)
        fluency_report = f"Total Time: {round(total_duration, 1)}s | Speaking: {round(active_ratio, 1)}% | Pauses: {round(pause_ratio, 1)}%"
    else:
        fluency_report = "Audio too short."

    # Ollama Evaluation
    user_prompt = f"Topic Given: \"{current_topic}\"\nFluency Stats: {fluency_report}\nTranscript: \"{transcript}\""
    
    try:
        response = ollama.chat(
            model='llama3',
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt}
            ]
        )
        ai_evaluation = response['message']['content']
    except Exception as e:
        ai_evaluation = f"AI Evaluation Error: {str(e)}"

    # Save to CSV Database
    log_entry = {
        "Timestamp": [time.strftime("%Y-%m-%d %H:%M:%S")],
        "Name": [name.strip()],
        "Email": [email.strip()],
        "Phone Number": [phone.strip()],
        "Topic": [current_topic],
        "Audio File": [saved_audio_path],
        "Fluency Stats": [fluency_report],
        "Transcript": [transcript],
        "Evaluation": [ai_evaluation]
    }
    df = pd.DataFrame(log_entry)
    if not os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, index=False)
    else:
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)

    return (
        transcript if transcript else "(No clear speech recognized)",
        fluency_report,
        ai_evaluation,
        gr.update(interactive=False)  # Lock the refresh button upon submission
    )

# 5. Web Interface Layout
with gr.Blocks(theme=gr.themes.Soft(), title="Extempore Assessment Portal") as demo:
    skips_state = gr.State(value=2)
    
    gr.Markdown(
        """
        # 🎙️ Extempore Communication Assessment
        Please enter your details below, check your assigned topic, and record your spoken answer for **up to 2 minutes**.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            name_input = gr.Textbox(label="Full Name", placeholder="e.g. Alex Johnson")
            email_input = gr.Textbox(label="Email Address", placeholder="e.g. alex@example.com")
            phone_input = gr.Textbox(label="Phone Number", placeholder="e.g. +1 555-0199")
            
            gr.Markdown("### 📋 Your Extempore Topic:")
            topic_display = gr.Textbox(
                value=get_initial_topic, 
                interactive=False, 
                label="Assigned Topic"
            )
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Change Topic (Max 2)", size="sm")
            skip_status = gr.Markdown("*(You can randomize the topic up to 2 times before recording)*")
            
            audio_input = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath", 
                label="Record Your Speech (Target: 2 minutes)"
            )
            submit_btn = gr.Button("Submit Assessment", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Assessment Summary")
            status_box = gr.Textbox(label="Transcript", interactive=False, lines=4)
            fluency_box = gr.Textbox(label="Speech Pacing & Pause Ratio", interactive=False)
            eval_box = gr.Textbox(label="Score Breakdown (9 Criteria)", interactive=False, lines=16)

    # Wire Refresh Logic
    refresh_btn.click(
        fn=refresh_topic,
        inputs=[topic_display, skips_state],
        outputs=[topic_display, skips_state, refresh_btn, skip_status]
    )

    # Wire Submission Logic
    submit_btn.click(
        fn=evaluate_candidate,
        inputs=[name_input, email_input, phone_input, topic_display, audio_input],
        outputs=[status_box, fluency_box, eval_box, refresh_btn]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)