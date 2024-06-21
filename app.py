from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from collections import deque
import tensorflow as tf

app = Flask(__name__)

# Load the model
LRCN_model = tf.keras.models.load_model('LRCN_MODEL.h5')

CLASSES_LIST = ["HorseRace", "MilitaryParade", "PushUps", "BaseballPitch", "PullUps", "Drumming", "Diving", "JavelinThrow", "PlayingTabla", "BenchPress"]

# Define the directory to save videos and images
SAVE_DIR = 'static/test_videos'
IMAGE_DIR = 'static/test_images'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Define the height of the frames
IMAGE_HEIGHT = 64  # Adjust this value according to your requirements
IMAGE_WIDTH = 64

def predict_on_video(video_file_path, output_image_path, SEQUENCE_LENGTH):
    video_reader = cv2.VideoCapture(video_file_path)
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    total_predictions = []
    frame_list = []
    frame_count = 0

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break

        resized_frame = cv2.resize(frame, (64, 64))
        normalized_frame = resized_frame / 255.0
        frames_queue.append(normalized_frame)
        frame_list.append(frame)
        frame_count += 1

        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            total_predictions.append(predicted_labels_probabilities)

    video_reader.release()

    if total_predictions:
        average_probabilities = np.mean(total_predictions, axis=0)
        predicted_label = np.argmax(average_probabilities)
        predicted_class_name = CLASSES_LIST[predicted_label] if 0 <= predicted_label < len(CLASSES_LIST) else "Unknown"
    else:
        predicted_class_name = "Unknown"

    if frame_list:
        middle_frame_index = len(frame_list) // 2
        middle_frame = frame_list[middle_frame_index]

        cv2.putText(middle_frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(output_image_path, middle_frame)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video_file' not in request.files:
        return redirect(url_for('index'))

    video_file = request.files['video_file']
    if video_file.filename == '':
        return redirect(url_for('index'))

    if video_file:
        video_filename = video_file.filename
        input_video_file_path = os.path.join(SAVE_DIR, video_filename)
        video_file.save(input_video_file_path)

        SEQUENCE_LENGTH = 20  # Set the sequence length for prediction
        output_image_file_path = os.path.join(IMAGE_DIR, f'{video_filename}-Output.png')

        predict_on_video(input_video_file_path, output_image_file_path, SEQUENCE_LENGTH)

        return redirect(url_for('show_result', image_filename=f'{video_filename}-Output.png'))

@app.route('/result/<path:image_filename>')
def show_result(image_filename):
    image_path = url_for('static', filename=f'test_images/{image_filename}')
    return render_template('result.html', image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
