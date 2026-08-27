import os
import time
import random
import shutil
import gradio as gr
from faster_whisper import WhisperModel
from groq import Groq
import pandas as pd

# Set your private Admin PIN here
ADMIN_PIN = "icdtad@1945"  # Change this to any password you want

AUDIO_STORAGE_DIR = "candidate_audios"
os.makedirs(AUDIO_STORAGE_DIR, exist_ok=True)
CSV_FILE = "extempore_evaluations.csv"

if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["Timestamp", "Name", "Email", "Phone Number", "Topic", "Audio File", "Fluency Stats", "Transcript", "Evaluation"]).to_csv(CSV_FILE, index=False)

whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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

def refresh_topic(current_topic, skips_left):
    if skips_left <= 0:
        return current_topic, skips_left, gr.update(interactive=False), "⚠️ Maximum limit of 2 topic changes reached."
    
    available_topics = [t for t in EXTEMPORE_TOPICS if t != current_topic]
    new_topic = random.choice(available_topics)
    new_skips = skips_left - 1
    
    status_msg = f"Topic refreshed. You have {new_skips} change(s) remaining." if new_skips > 0 else "Topic refreshed. No changes remaining."
    btn_state = gr.update(interactive=(new_skips > 0))
    
    return new_topic, new_skips, btn_state, status_msg

SYSTEM_PROMPT = """
You are a fair, objective, and supportive English communication assessor evaluating a 2-minute Extempore speech.
You will be provided with the candidate's transcript, the assigned topic, and a fluency report showing speech pace and pauses.

CRITICAL INSTRUCTION: "Topic Relevance" is the absolute highest priority metric. If a candidate speaks significantly off-topic or ignores the prompt, their Overall Extempore Rating must be heavily penalized, regardless of how perfect their grammar, vocabulary, or fluency is.

Evaluate the candidate on a scale of 1 to 10 across each of the following 7 parameters:

- Spoken English (Be reasonably forgiving of minor mistakes)
- Grammar (Do not heavily penalize minor conversational slips)
- Vocabulary
- Fluency (Evaluate pacing, but allow for natural conversational pauses)
- Neutral Accent (Score fairly; only deduct points if heavy localized inflections make the transcript severely fragmented)
- Confidence (Evaluate based on continuous structure and tone, but allow for minor hesitations)
- Topic Relevance (HIGHEST PRIORITY: Evaluate how accurately the candidate's response addresses the assigned topic. If the response is entirely off-topic, this score must be a 1 or 2, and the Overall Rating must absolutely not exceed 4/10).

Output EXACTLY in this format:
Spoken English: [Score]/10
Grammar: [Score]/10
Vocabulary: [Score]/10
Fluency: [Score]/10
Neutral Accent: [Score]/10
Confidence: [Score]/10
Topic Relevance: [Score]/10
Overall Extempore Rating: [Weighted Score heavily anchored by Topic Relevance]/10

Detailed Feedback: [2-3 concise, encouraging sentences. If the candidate was off-topic, state this explicitly as the primary reason for a lower score.]
"""

def evaluate_candidate(name, email, phone, current_topic, audio_filepath):
    if not name.strip() or not email.strip() or not phone.strip():
        return "⚠️ Error: Please fill in your Name, Email, and Phone Number.", "", "", gr.update()
    if not audio_filepath:
        return "⚠️ Error: No audio recording detected. Please record your answer.", "", "", gr.update()

    clean_identifier = email.replace("@", "_").replace(".", "_").strip()
    saved_audio_filename = f"{clean_identifier}_{phone.strip()}_{int(time.time())}.wav"
    saved_audio_path = os.path.join(AUDIO_STORAGE_DIR, saved_audio_filename)
    shutil.copy(audio_filepath, saved_audio_path)

    segments, info = whisper_model.transcribe(audio_filepath, beam_size=5)
    transcript = ""
    total_speech_time = 0.0
    for segment in segments:
        transcript += segment.text + " "
        total_speech_time += (segment.end - segment.start)

    transcript = transcript.strip()
    total_duration = info.duration

    if total_duration > 0:
        active_ratio = (total_speech_time / total_duration) * 100
        pause_ratio = max(0.0, 100.0 - active_ratio)
        fluency_report = f"Total Time: {round(total_duration, 1)}s | Speaking: {round(active_ratio, 1)}% | Pauses: {round(pause_ratio, 1)}%"
    else:
        fluency_report = "Audio too short."

    user_prompt = f"Topic Given: \"{current_topic}\"\nFluency Stats: {fluency_report}\nTranscript: \"{transcript}\""
    
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        ai_evaluation = completion.choices[0].message.content
    except Exception as e:
        ai_evaluation = f"AI Evaluation Error: {str(e)}"

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
    df.to_csv(CSV_FILE, mode='a', header=False, index=False)

    return (
        transcript if transcript else "(No clear speech recognized)",
        fluency_report,
        ai_evaluation,
        gr.update(interactive=False)
    )

# Admin verification logic
def unlock_admin_download(entered_pin):
    if entered_pin == ADMIN_PIN:
        return gr.update(value=CSV_FILE, visible=True), "✅ Access granted."
    else:
        return gr.update(visible=False), "❌ Incorrect Admin PIN."

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
            eval_box = gr.Textbox(label="Score Breakdown (6 Criteria)", interactive=False, lines=12)

    # Protected Admin Download Section
    with gr.Accordion("🔒 Admin Portal (Download CSV Database)", open=False):
        pin_input = gr.Textbox(label="Enter Admin PIN", type="password", placeholder="Enter PIN")
        unlock_btn = gr.Button("Unlock CSV Download", size="sm")
        admin_status = gr.Markdown("")
        admin_download_file = gr.File(label="Extempore Database Export", visible=False)

    refresh_btn.click(
        fn=refresh_topic,
        inputs=[topic_display, skips_state],
        outputs=[topic_display, skips_state, refresh_btn, skip_status]
    )

    submit_btn.click(
        fn=evaluate_candidate,
        inputs=[name_input, email_input, phone_input, topic_display, audio_input],
        outputs=[status_box, fluency_box, eval_box, refresh_btn]
    )

    unlock_btn.click(
        fn=unlock_admin_download,
        inputs=[pin_input],
        outputs=[admin_download_file, admin_status]
    )

if __name__ == "__main__":
    demo.launch()
