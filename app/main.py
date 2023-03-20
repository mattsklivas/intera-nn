# Flask specific imports
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

# Python lib/pip modules
from datetime import datetime
from os import environ as env
from dotenv import find_dotenv, load_dotenv
from pymongo import errors, results
import cv2 
import mediapipe as mp
import torch
import random
import math
import os
import openai
import sys
import tempfile

# Project imports
from asl_model import AslNeuralNetwork
from config import Database

# load the environment variables from the .env file
load_dotenv(find_dotenv())

# OpenAPI key
openai.api_key = env.get('OPENAI_API_KEY')

app = Flask(__name__)
CORS(app)

# get a reference to the databases
intera_calls_db = Database.client[Database.intera_calls_db]

# get a reference to the collections
try: 
    messages = intera_calls_db[Database.messages_collection]
    rooms = intera_calls_db[Database.rooms_collection]
except errors.CollectionInvalid as err:
    print(err)

# Load model
current_dir = os.getcwd()
model = AslNeuralNetwork()
model_state_dict = torch.load(os.path.join(current_dir, 'asl_model_v6.2.pth'), map_location=model.device)
model.load_state_dict(model_state_dict)

# Dictionary of all words here
# signs = ['bad', 'bye', 'easy', 'good', 'happy', 'hello', 'like', 'me', 'meet', 'more', 'no', 'please', 'sad', 'she', 'sorry', 'thank you', 'want', 'why', 'yes', 'you']
signs =  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'bad', 'bye', 'easy', 'good', 'happy', 'hello', 'how', 'like', 'me', 'meet', 'more', 'no', 'please', 'sad', 'she', 'sorry', 'thank you', 'want', 'what', 'when', 'where', 'which', 'who', 'why', 'yes', 'you']

# Temporal fit constants
INPUT_SIZE = 201
# INPUT_SIZE = 201 #226
NUM_SEQUENCES = 48

# -------------------------------- UTIL FUNCTIONS ---------------------------------

def processing_frame(frame, holistic):
    # Initialize pose and left/right hand tensoqs
    # left_hand, right_hand, pose = torch.zeros(21 * 3), torch.zeros(21 * 3), torch.zeros(25 * 3)
    left_hand, right_hand, pose = torch.zeros(21 * 3), torch.zeros(21 * 3), torch.zeros(25 * 3)

    # Pass frame to model by reference (not writeable) for improving performance
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process image using Holistic model (Detect and predict keypoints)
    results = holistic.process(frame)

    # Ignore frames with no detection of both hands
    if not results.left_hand_landmarks and not results.right_hand_landmarks:
        return []

    # No hand detected
    if not results.left_hand_landmarks:
        left_hand = torch.zeros(21 * 3)
    # Hand detected
    else:
        # Add left hand keypoints (21 w/ 3d coordinates)
        lh = results.left_hand_landmarks
        for i, landmark in enumerate(lh.landmark):
            shift_ind = i * 3
            left_hand[shift_ind] = landmark.x
            left_hand[shift_ind + 1] = landmark.y
            left_hand[shift_ind + 2] = landmark.z            

    if not results.right_hand_landmarks:
        right_hand = torch.zeros(21 * 3)
    else:
        # Add right hand keypoints (21 w/ 3d coordinates)
        rh = results.right_hand_landmarks
        for j, landmark in enumerate(rh.landmark):
            shift_ind = j * 3
            right_hand[shift_ind] = landmark.x
            right_hand[shift_ind + 1] = landmark.y
            right_hand[shift_ind + 2] = landmark.z

    # No pose detected
    if not results.pose_landmarks:
        pose = torch.zeros(25 * 3)
        # pose = torch.zeros(25 * 4)
    # Pose detected
    else:
        # Add pose keypoints (25 w/ 3d coordinates plus visbility probability)
        pl = results.pose_landmarks
        for k, landmark in enumerate(pl.landmark):
            # Ignore lower body (landmarks #25-33)
            if k >= 25:
                break

            shift_ind = k * 3
            pose[shift_ind] = landmark.x
            pose[shift_ind + 1] = landmark.y
            pose[shift_ind + 2] = landmark.z  
            # pose[shift_ind + 3] = landmark.visibility

    # Concatenate processed frame
    return torch.cat([left_hand, right_hand, pose])

# Binary search on buffer frames (bf) using bit value (bv)
# Applicable to finding starting frame (bv = 1) and ending frame (bv = 0)
def bin_search(bf, bv, hm):
    # Search variables
    l, m, h = 0, 0, len(bf) - 1

    while l <= h:
        m = (h + l) // 2

        # Pass frame to mediapipe
        mv = processing_frame(bf[m], hm)

        # Found start frame with landmarks or end frame with no landmarks continue left
        if (bv == 1 and mv != []) or (bv == 0 and mv == []): 
            # Search left
            h = m - 1  
        else:
            # Search right
            l = m + 1

    return m

def get_holistic_model():
    # Get Mediapipe holistic solution
    mp_holistic = mp.solutions.holistic

    # Instantiate holistic model, specifying minimum detection and tracking confidence levels
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) 
    
    return holistic

import traceback
def live_video_temporal_fit(frames):
    # Calculate num frames over or under data frames input limit 
    try:
        print(f'Frames: {len(frames)}')
        num_frames = len(frames)
    except Exception as e:
        print(f'Error: {e}')
    missing_frames = NUM_SEQUENCES - num_frames

    if missing_frames == 0:
        print("Data already fitted to 48 frames")
        
    is_over_limit = missing_frames < 0
    # print(f'Problem: {num_frames}')

    # Must select frames manually 
    missing_frames = abs(missing_frames)

    # Calculate num of times each frame and update frame population to properly sample missing frames
    frame_pop = range(num_frames)
    if num_frames == 0:
        # No frames in file
        print('Error - Empty video provided')
        return []

    if num_frames < missing_frames:
        factor = math.ceil(NUM_SEQUENCES / num_frames)
        frame_pop = list(frame_pop) * factor

    # Pick frames randomly to remove or duplicate based on data size
    frame_indices = sorted(random.sample(frame_pop, missing_frames), reverse=True)

    # Data temporal fit
    if is_over_limit:
        # Delete frames over limit
        for frame_index in frame_indices:
            frames.pop(frame_index)
    else:
        # Duplicate missing frames
        for frame_index in frame_indices:
            curr_frame = frames[frame_index]
            frames.insert(frame_index, curr_frame)

    # Adjust format to torch tensors
    torch_frames = torch.zeros([1, NUM_SEQUENCES, INPUT_SIZE], dtype=torch.float)
    for seq, frame in enumerate(frames):
        torch_frames[0][seq] = frame
    
    return torch_frames

def quick_fit(frames):
    num_frames = len(frames)
    missing_frames = abs(NUM_SEQUENCES - num_frames)
    frame_pop = range(num_frames)

    frame_indices = sorted(random.sample(frame_pop, missing_frames), reverse=True)

    for frame_index in frame_indices:
        frames.pop(frame_index)

    return frames

def softmax(output):
    e = torch.exp(output)
    return e / e.sum()

# Predict multiple live sign from video call
def predict_live_sign(video):
    word_signs = [[]]
    curr_sign_index = 0
    next_word_started = False

    no_sign_count = 0
    NEXT_SIGN_BUFF = 6

    # List of predicted words
    predictions = []
    conf_vals = []

    try:
        holistic = get_holistic_model()

        try:
            with tempfile.NamedTemporaryFile(suffix = '.webm') as temp:
                temp.write(video)            

                # Collect frames until no more frames remain 
                cap = cv2.VideoCapture(temp.name)
                while True:
                    # Capture frame from camera
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Process every frame
                    processed_frame = processing_frame(frame, holistic)

                    # Landmarks detected in frame
                    if processed_frame != []:
                        # Reset count and set next sign started to true
                        no_sign_count = 0     
                        if not next_word_started: 
                            print('New word started!')
                            next_word_started = True

                        # Add processed frame to current sign frames
                        word_signs[curr_sign_index].append(processed_frame)
                        print(' + Frame Added')
                    # Empty frame
                    else:
                        # Count num consecutive empty frames
                        no_sign_count += 1

                        # Start new sign word after n consecutive empty frames
                        if no_sign_count >= NEXT_SIGN_BUFF:
                            no_sign_count = 0

                            # Only add new word after previous word is complete
                            if next_word_started:
                                print('End of sign detected! Waiting for start of next word!')
                                curr_sign_index += 1
                                word_signs.append([])
                                next_word_started = False

                # Release the camera and close the window
                cap.release()
        except Exception as e:
            print('Video Error: ', e)
            return 0, 'N/A', 0, f'Video Error: {str(e.args[0])}', []

        try:
            # For each word, fit the word to 48 frames then pass to model
            for sign_word_frames in word_signs:
                if len(sign_word_frames) == 0:
                    continue

                fitted_sign_frames = live_video_temporal_fit(sign_word_frames)

                if not isinstance(fitted_sign_frames, list):
                    # Pass to model and add to prediction sentence
                    y_pred = model(fitted_sign_frames)
                    _, predicted = torch.max(y_pred.data, 1)

                    predicted_word = signs[predicted]
                    predictions.append(predicted_word)

                    # Get the confidence %
                    y_prob = softmax(y_pred)
                    confidence = y_prob[0][predicted] 
                    conf_vals.append(confidence.item())

                print(f'Word prediction/Confidence %: {predicted_word}/{confidence.item()}')
        except Exception as e:
            print('Prediction Error: ', e)
            return 0, 'N/A', 0, f'Prediction Error: {str(e.args[0])}', []

    except Exception as e:
        print('NN Error: ', e)
        return 0, 'N/A', 0, f'NN Error: {str(e.args[0])}', []


    # # Append to list of predicted words and confidence percentages
    prediction = predictions[0]
    if len(predictions) > 1:
        prediction = " ".join(predictions)
        response = openai.Completion.create(
            model="text-curie-001",
            prompt=f"Give the grammatically-correct version of this phrase : {prediction}",
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
        prediction = response.choices[0].text.strip()
        
    confidence = sum(conf_vals)/len(conf_vals)

    # Return result
    return 1, prediction, confidence, None, predictions


# Predict single sign from practice module
def predict_single_sign(video):
    # Processed frames
    mp_frames = []

    try:
        holistic = get_holistic_model()

        with tempfile.NamedTemporaryFile(suffix = '.webm') as temp:
            temp.write(video)

            # Collect frames until video is complete
            cap = cv2.VideoCapture(temp.name)
            while True:
                # Capture frame from camera
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every frame
                processed_frame = processing_frame(frame, holistic)
                    
                # Landmarks detected in frame
                if processed_frame != []:
                    mp_frames.append(processed_frame)

            # Release the camera and close the window
            cap.release()

            # Fit
            y_pred = None
            predicted = None
            if len(mp_frames) > 0:
                keypoints = live_video_temporal_fit(mp_frames)

                # Neural network model prediction
                y_pred = model(keypoints)
                _, predicted = torch.max(y_pred.data, 1) # Apply softmax here to have percentage

    except Exception as e:
        print('NN Error: ', e)
        return 0, 'N/A', 0, f'Error: {str(e.args[0])}'

    try:
        # Get the confidence %
        y_prob = softmax(y_pred)
        confidence = y_prob[0][predicted]
        predicted_word = signs[predicted]
    except Exception as e:
        predicted_word = 'N/A'

    print(f'Word prediction/Confidence %: {predicted_word}/{confidence.item()}')

    # Return result
    return 1, predicted_word, confidence.item(), None

# -------------------------------- CONTROLLERS ---------------------------------

def process_video(video, word=None):
    # Predict one sign (practice module)
    signs = []
    if word:
        success, prediction, confidence, error = predict_single_sign(video)
    # Predict multiple (video calls)
    else:
        success, prediction, confidence, error, signs = predict_live_sign(video)

    if success == 0:
        return (0, error, f'Incorrect', confidence)

    # Practice module
    if word:
        result = 'Incorrect'
        if prediction == word:
            result = 'Correct'
        return (1, f'Sign attempt processed successfully', result, confidence)

    # Video calls
    else:
        prediction = f'{signs}\n{prediction}' if len(signs) > 1 else prediction
        return (1, f'Sign attempt processed successfully', prediction, confidence)

def create_message_entry(room_id, to_user, from_user, prediction):
    message = {'date_created': datetime.now(), 'room_id': room_id, 'to': to_user, 'from': from_user, 'text': prediction,\
        'edited': False, 'message_type': 'ASL', 'corrected': ''}

    result = messages.insert_one(message)

    if isinstance(result, results.InsertOneResult):
        if result.inserted_id:

            # Add message reference to corresponding room document
            rooms.find_one_and_update({'room_id': room_id}, {'$push': {'messages': result.inserted_id}})

            return (1, 'Message created successfully')

    return (0, 'Error creating message entry')

# -------------------------------- ROUTES ---------------------------------

# Health check route
@app.route("/can_connect")
def can_connect():
    print("endpoint reached")
    status_code = Response(status=200)
    return status_code

@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    video = request.files.get('video', None)
    if video is not None:
        video = video.read()
    else:
        return jsonify(error='No video provided', status=400)

    word = request.form.get('word', None)
    if word is None:
        return jsonify(error='No word provided', status=400)

    status, message, result, confidence = process_video(video, word)

    if status == 0:
        return jsonify(error=message, status=401)
    else:
        return jsonify(message=message, data={'word': word, 'result': result, 'confidence': confidence}, status=200)

@app.route("/process_sign", methods=["POST"])
def process_sign():
    video = request.files.get('video', None)
    if video is not None:
        video = video.read()
    else:
        return jsonify(error='No video provided', status=400)

    room_id = request.form.get('room_id')
    from_user = request.form.get('from_user')
    to_user = request.form.get('to_user')
    if room_id is None:
        return jsonify(error='Room ID not provided', status=400)
    if to_user is None:
        return jsonify(error='To User not provided', status=400)
    if to_user is None:
        return jsonify(error='To User not provided', status=400)

    status, message, prediction, confidence = process_video(video)

    if status == 0:
        prediction = f'[ERROR: Prediction unsuccessful. Please invalidate this message with your intended message.]'
    elif confidence < 0.6:
        prediction = f'{prediction} [INFO: Low confidence in ASL sign(s) predicted ({round(confidence * 100, 2)}%)]'

    # Append message to chat
    status, m_ = create_message_entry(room_id, to_user, from_user, prediction)

    if status == 0:
        return jsonify(error=m_, status=401)

    return jsonify(message=m_, data={'room_id': room_id, 'prediction': prediction, 'confidence': confidence}, status=200)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
