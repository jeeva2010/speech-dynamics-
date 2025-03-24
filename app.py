import os
import uuid  # For generating unique IDs
from datetime import datetime
from flask import Flask, request, render_template, send_file, redirect, url_for
import speech_recognition as sr
import parselmouth
import matplotlib.pyplot as plt
import pronouncing

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

recognizer = sr.Recognizer()

def generate_unique_name():
    """Generates a unique name using timestamp and UUID."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:8]
    return f"{timestamp}_{unique_id}"

def phonetic_transcription(word):
    """Get the phonetic transcription for a word."""
    phones = pronouncing.phones_for_word(word)
    return phones[0] if phones else "No transcription found"

def transcribe_audio_to_text(audio_path):
    """Transcribes audio to text using Google Speech Recognition."""
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Error with the API: {e}"

def prosody_analysis(audio_path, output_name):
    """Performs prosody analysis and saves the plot."""
    sound = parselmouth.Sound(audio_path)

    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_times = pitch.xs()
    voiced_frames = pitch_values > 0
    pitch_values = pitch_values[voiced_frames]
    pitch_times = pitch_times[voiced_frames]

    intensity = sound.to_intensity()
    intensity_values = intensity.values.T.flatten()
    intensity_times = intensity.xs()

    min_len = min(len(pitch_times), len(pitch_values), len(intensity_times), len(intensity_values))
    pitch_times, pitch_values = pitch_times[:min_len], pitch_values[:min_len]
    intensity_times, intensity_values = intensity_times[:min_len], intensity_values[:min_len]

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(pitch_times, pitch_values, color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (Hz)")
    plt.title("Pitch Analysis")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(intensity_times, intensity_values, color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity (dB)")
    plt.title("Intensity Analysis")
    plt.grid()

    plt.tight_layout()
    plot_path = os.path.join('static', f'{output_name}_prosody.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return "No file part"
        file = request.files['audio_file']
        if file.filename == '':
            return "No selected file"

        # Generate a unique name for this upload
        unique_name = generate_unique_name()

        # Save uploaded file with the unique name
        file_path = os.path.join(UPLOAD_FOLDER, f'{unique_name}_{file.filename}')
        file.save(file_path)

        # Perform analysis
        text = transcribe_audio_to_text(file_path)
        prosody_plot_path = prosody_analysis(file_path, unique_name)

        # Generate phonetic transcription
        words = text.split()
        phonetic_list = [phonetic_transcription(word) for word in words]
        phonetic = ' '.join(phonetic_list)

        # Save transcription and phonetic content
        transcription_path = os.path.join(OUTPUT_FOLDER, f'{unique_name}.txt')
        with open(transcription_path, 'w') as f:
            f.write(f"Transcribed Text:\n{text}\n\nPhonetic Transcription:\n{phonetic}")

        return render_template('index.html', text=text, phonetic=phonetic, 
                               plot_url=url_for('static', filename=f'{unique_name}_prosody.png'), 
                               download_url=url_for('download_file', filename=f'{unique_name}.txt'))

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
