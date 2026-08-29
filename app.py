import os
import time
import random
import shutil
import gradio as gr
from faster_whisper import WhisperModel
from groq import Groq
import pandas as pd

# Set your private Admin PIN here
ADMIN_PIN = "icdtad@1945"

AUDIO_STORAGE_DIR = "candidate_audios"
os.makedirs(AUDIO_STORAGE_DIR, exist_ok=True)
CSV_FILE = "extempore_evaluations.csv"

if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["Timestamp", "Name", "Email", "Phone Number", "Topic", "Audio File",
                          "Fluency Stats", "Transcript", "Evaluation"]).to_csv(CSV_FILE, index=False)

whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ==========================================
# 📝 CHANGE YOUR QUESTIONS HERE
# ==========================================
EXTEMPORE_TOPICS = [
    "If you could exchange lives with any person for one day, who would you choose?",
    "If you were given ₹10 crore but had to spend it all in 24 hours, how would you spend it?",
    "Would you choose a high-paying job you hate or a low-paying job you love? Why?",
    "Can money buy happiness?",
    "Would you rather know your future or be able to change your past?",
    "If there were no internet for one month, what would you do?",
    "If you could go back to being 10 years old for one day, what would you do?"
]
# ==========================================


# ---------- Reveal / Refresh (timer itself now runs in the browser) ----------

def reveal_topic():
    topic = random.choice(EXTEMPORE_TOPICS)
    # Topic box (value + visible), hide Reveal button, show Change button, start timer
    return (
        gr.update(value=topic, visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
        f"START_{time.time()}"
    )


def refresh_topic(current_topic, skips_left):
    if skips_left <= 0:
        return (current_topic, skips_left, gr.update(interactive=False),
                "⚠️ Maximum limit of 2 topic changes reached.", gr.update())

    available_topics = [t for t in EXTEMPORE_TOPICS if t != current_topic]
    if not available_topics:
        available_topics = EXTEMPORE_TOPICS

    new_topic = random.choice(available_topics)
    new_skips = skips_left - 1

    status_msg = f"Topic refreshed. You have {new_skips} change(s) remaining." if new_skips > 0 \
        else "Topic refreshed. No changes remaining."
    btn_state = gr.update(interactive=(new_skips > 0))

    # Every successful skip restarts the 10-second timer
    return new_topic, new_skips, btn_state, status_msg, f"START_{time.time()}"


def unlock_recorder():
    # Called by the browser when the 10s countdown hits zero
    return (
        "### 🔴 PREPARATION OVER! Recording started automatically. You have 2 minutes.",
        gr.update(visible=True),
        gr.update(visible=False)
    )


# ==========================================
# 🧠 NEW UPDATED SYSTEM PROMPT
# ==========================================
SYSTEM_PROMPT = """
You are a fair, objective, and supportive English communication assessor evaluating a 2-minute Extempore speech.
You will be provided with the candidate's transcript, the assigned topic, and a fluency report showing speech pace and pauses.

CRITICAL INSTRUCTION: "Topic Relevance" is the absolute highest priority metric. If a candidate speaks significantly off-topic or ignores the prompt, their Overall Extempore Rating must be heavily penalized, regardless of how perfect their grammar, vocabulary, or fluency is.

Evaluate the candidate on a scale of 1 to 10 across each of the following 7 parameters:

* Spoken English (Be reasonably forgiving of minor mistakes)
* Grammar (Do not heavily penalize minor conversational slips)
* Vocabulary
* Fluency (Evaluate pacing, but allow for natural conversational pauses)
* Neutral Accent (Score fairly; only deduct points if heavy localized inflections make the transcript severely fragmented)
* Confidence (Evaluate based on continuous structure and tone, but allow for minor hesitations)
* Topic Relevance (HIGHEST PRIORITY: Evaluate how accurately the candidate's response addresses the assigned topic. If the response is entirely off-topic, this score must be a 1 or 2, and the Overall Rating must absolutely not exceed 4/10).

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
            model="llama-3.1-8b-instant",
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


def unlock_admin_download(entered_pin):
    if entered_pin == ADMIN_PIN:
        return gr.update(value=CSV_FILE, visible=True), "✅ Access granted."
    else:
        return gr.update(visible=False), "❌ Incorrect Admin PIN."


# ---------- Browser-side helpers ----------

# Hide pause/resume buttons entirely, and hide the two helper textboxes
CUSTOM_CSS = """
#audio_input .pause-button,
#audio_input .resume-button { display: none !important; }
.hidden-box { display: none !important; }
"""

# Runs once on page load:
#  1. asks for microphone permission immediately
#  2. keeps renaming "Stop" -> "Save" and hiding Pause/Resume, whatever Gradio re-renders
ON_LOAD_JS = """
function() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => { stream.getTracks().forEach(t => t.stop()); })
        .catch(err => console.log('Mic permission error:', err));

    const fixButtons = () => {
        const root = document.querySelector('#audio_input');
        if (!root) return;
        root.querySelectorAll('button').forEach(b => {
            const txt = (b.innerText || '').trim();
            if (txt === 'Pause' || txt === 'Resume') { b.style.display = 'none'; return; }
            b.childNodes.forEach(n => {
                if (n.nodeType === 3 && n.textContent.trim() === 'Stop') n.textContent = 'Save';
            });
        });
    };
    fixButtons();
    new MutationObserver(fixButtons).observe(document.body, { childList: true, subtree: true });
}
"""

# Runs every time the server sends a new START_<timestamp> value.
# Restarts the 10s countdown from scratch (any previous countdown / pending auto-save is cancelled).
COUNTDOWN_JS = """
function(val) {
    if (!val || !val.startsWith('START')) return;
    if (window.__prepInterval) clearInterval(window.__prepInterval);
    if (window.__saveTimeout)  clearTimeout(window.__saveTimeout);

    const timerEl = document.querySelector('#timer_display');
    let n = 10;
    const render = () => {
        if (timerEl) timerEl.innerHTML = '<h3>⏳ Preparation Time: ' + n + ' seconds remaining...</h3>';
    };
    render();

    window.__prepInterval = setInterval(() => {
        n -= 1;
        if (n > 0) { render(); return; }
        clearInterval(window.__prepInterval);
        window.__prepInterval = null;

        // Tell the server the countdown finished so it can show the recorder and hide Change Topic
        const box = document.querySelector('#phase_trigger textarea, #phase_trigger input');
        if (box) {
            box.value = 'UNLOCK_' + Date.now();
            box.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }, 1000);
}
"""

# Runs right after the server has made the recorder visible:
# clicks Record, then clicks Save exactly 2 minutes later.
AUTO_RECORD_JS = """
function() {
    const findBtn = (cls, words) => {
        const root = document.querySelector('#audio_input');
        if (!root) return null;
        let b = root.querySelector('button.' + cls);
        if (b) return b;
        return Array.from(root.querySelectorAll('button')).find(x => {
            const t = (x.innerText || '').trim();
            return words.includes(t);
        }) || null;
    };

    const tryRecord = (attempt) => {
        const rec = findBtn('record-button', ['Record']);
        if (rec) {
            rec.click();
            window.__saveTimeout = setTimeout(() => {
                const save = findBtn('stop-button', ['Save', 'Stop']);
                if (save) save.click();
            }, 120000);
        } else if (attempt < 20) {
            setTimeout(() => tryRecord(attempt + 1), 250);
        }
    };
    tryRecord(0);
}
"""


with gr.Blocks(theme=gr.themes.Soft(), title="Extempore Assessment Portal", css=CUSTOM_CSS) as demo:
    skips_state = gr.State(value=2)

    # Server -> browser: restart countdown
    js_trigger = gr.Textbox(elem_id="js_trigger", elem_classes=["hidden-box"])
    # Browser -> server: countdown finished
    phase_trigger = gr.Textbox(elem_id="phase_trigger", elem_classes=["hidden-box"])

    demo.load(fn=None, js=ON_LOAD_JS)

    gr.Markdown(
        """
        # 🎙️ Extempore Communication Assessment
        Please enter your details below. Once you reveal your topic, a **10-second preparation timer** will begin.
        **The recording will start automatically** after the timer ends.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            name_input = gr.Textbox(label="Full Name", placeholder="e.g. Alex Johnson")
            email_input = gr.Textbox(label="Email Address", placeholder="e.g. alex@example.com")
            phone_input = gr.Textbox(label="Phone Number", placeholder="e.g. +1 555-0199")

            gr.Markdown("### 📋 Your Extempore Topic:")
            reveal_btn = gr.Button("👀 Reveal Topic & Start Prep Timer", variant="primary")

            topic_display = gr.Textbox(label="Assigned Topic", interactive=False, visible=False)

            with gr.Row():
                refresh_btn = gr.Button("🔄 Change Topic (Max 2)", size="sm", visible=False)
            skip_status = gr.Markdown("*(You can randomize the topic up to 2 times during your 10s prep time)*")

            timer_display = gr.Markdown("### ⏳ Awaiting Topic Reveal...", elem_id="timer_display")
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record Your Speech (Target: 2 minutes)",
                visible=False,
                elem_id="audio_input"
            )
            submit_btn = gr.Button("Submit Assessment", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Assessment Summary")
            status_box = gr.Textbox(label="Transcript", interactive=False, lines=4)
            fluency_box = gr.Textbox(label="Speech Pacing & Pause Ratio", interactive=False)
            eval_box = gr.Textbox(label="Score Breakdown (7 Criteria)", interactive=False, lines=13)

    with gr.Accordion("🔒 Admin Portal (Download CSV Database)", open=False):
        pin_input = gr.Textbox(label="Enter Admin PIN", type="password", placeholder="Enter PIN")
        unlock_btn = gr.Button("Unlock CSV Download", size="sm")
        admin_status = gr.Markdown("")
        admin_download_file = gr.File(label="Extempore Database Export", visible=False)

    # ---- Wiring ----
    reveal_btn.click(
        fn=reveal_topic,
        outputs=[topic_display, reveal_btn, refresh_btn, js_trigger]
    )

    refresh_btn.click(
        fn=refresh_topic,
        inputs=[topic_display, skips_state],
        outputs=[topic_display, skips_state, refresh_btn, skip_status, js_trigger]
    )

    # Countdown runs in the browser; restarts on every new START_ value
    js_trigger.change(fn=None, inputs=[js_trigger], js=COUNTDOWN_JS)

    # Countdown finished -> show recorder, hide Change Topic, then auto-Record / auto-Save
    phase_trigger.change(
        fn=unlock_recorder,
        outputs=[timer_display, audio_input, refresh_btn]
    ).then(fn=None, js=AUTO_RECORD_JS)

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
