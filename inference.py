import torch
import os
import cv2
import librosa
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from moviepy.editor import VideoFileClip
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input
import warnings
import tempfile
import soundfile as sf
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import base64
import io

# ---- Attention Module ----
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):  # (batch, seq, hidden)
        weights = self.attn(lstm_output)            # (batch, seq, 1)
        weights = torch.softmax(weights, dim=1)     # across seq
        context = (weights * lstm_output).sum(dim=1) # (batch, hidden)
        return context

# ---- Multimodal Detector ----
class MultimodalDeepfakeDetector(nn.Module):
    def __init__(self, hidden_dim=256, audio_hidden_dim=128, use_attention=True):
        super().__init__()
        # Visual processing
        self.frame_norm = nn.LayerNorm(2048)
        self.frame_proj = nn.Linear(2048, 512)
        self.visual_lstm = nn.LSTM(512, hidden_dim, batch_first=True, bidirectional=True)

        # Audio processing (using pre-extracted features)
        self.audio_norm = nn.LayerNorm(768)  # Wav2Vec2 feature dimension
        self.audio_proj = nn.Linear(768, 512)  # Increased to match visual dimension
        self.audio_lstm = nn.LSTM(512, hidden_dim, batch_first=True, bidirectional=True)  # Same hidden dim as visual

        self.use_attention = use_attention
        if use_attention:
            self.visual_attn = Attention(2 * hidden_dim)
            self.audio_attn = Attention(2 * hidden_dim)
            # Add cross-modal attention
            self.cross_attn = nn.MultiheadAttention(2 * hidden_dim, num_heads=4)

        # Feature balancing
        self.visual_balance = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.audio_balance = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

        fusion_dim = 2*hidden_dim + 2*hidden_dim  # Now both modalities have same dimension
        self.fc1 = nn.Linear(fusion_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, frame_features, audio_features):
        # Visual
        frame_features = self.frame_norm(frame_features)
        frame_features = self.frame_proj(frame_features)
        visual_out, _ = self.visual_lstm(frame_features)
        visual_feat = self.visual_attn(visual_out) if self.use_attention else visual_out[:, -1, :]

        # Audio
        audio_features = self.audio_norm(audio_features)
        audio_features = self.audio_proj(audio_features)
        audio_out, _ = self.audio_lstm(audio_features)
        audio_feat = self.audio_attn(audio_out) if self.use_attention else audio_out[:, -1, :]

        # Cross-modal attention
        if self.use_attention:
            visual_feat = visual_feat.unsqueeze(0)  # [1, batch, dim]
            audio_feat = audio_feat.unsqueeze(0)    # [1, batch, dim]
            visual_feat, _ = self.cross_attn(visual_feat, audio_feat, audio_feat)
            audio_feat, _ = self.cross_attn(audio_feat, visual_feat, visual_feat)
            visual_feat = visual_feat.squeeze(0)    # [batch, dim]
            audio_feat = audio_feat.squeeze(0)      # [batch, dim]

        # Feature balancing
        visual_feat = self.visual_balance(visual_feat)
        audio_feat = self.audio_balance(audio_feat)

        # Fusion with equal weights
        fused = torch.cat([visual_feat, audio_feat], dim=1)
        x = self.fc1(fused)
        x = self.dropout(x)
        x = self.bn1(x)
        out = torch.sigmoid(self.fc2(x))
        return out



# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging

# Memory optimization settings
torch.backends.cudnn.benchmark = True

# Configure TensorFlow
try:
    # Disable GPU for TensorFlow to avoid conflicts with PyTorch
    tf.config.set_visible_devices([], 'GPU')
    print("ℹ️ Using CPU for TensorFlow operations")
except Exception as e:
    print(f"⚠️ Error setting TensorFlow device: {e}")


# --- Global variables and Model Loading ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Constants matching training configuration
MAX_FRAMES_FOR_INFERENCE = 20  # Match the sequence length used in training
BATCH_SIZE_FOR_FRAME_PROCESSING = 32  # Increased batch size for faster processing
AUDIO_SEGMENT_LENGTH = 320000  # ~20 seconds at 16kHz

# Cache for processed features
feature_cache = {}

# Create a TensorFlow function for batch prediction
@tf.function(reduce_retracing=True)
def predict_batch(batch_array):
    return xception_model_inference(batch_array, training=False)

# Initialize face cascade with error handling
face_cascade = None
try:
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        alt_base_path = os.path.join(cv2.__path__[0], 'data')
        face_cascade_path_alt = os.path.join(alt_base_path, 'haarcascade_frontalface_default.xml')
        if os.path.exists(face_cascade_path_alt):
            face_cascade_path = face_cascade_path_alt
        else:
            raise FileNotFoundError(f"Haarcascade file not found at {face_cascade_path} or {face_cascade_path_alt}")
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise ValueError("Failed to load face cascade classifier")
    print(f"✅ Face cascade loaded successfully from {face_cascade_path}")
except Exception as e:
    print(f"❌ Error loading face cascade: {e}")
    print("⚠️ Face detection will not be available")

# Initialize Wav2Vec2 with error handling
print("Loading Wav2Vec2 model for inference...")
wav2vec_processor_inference = None
wav2vec_model_inference = None
try:
    wav2vec_processor_inference = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model_inference = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE).eval()
    print("✅ Wav2Vec2 model loaded successfully")
except Exception as e:
    print(f"❌ Error loading Wav2Vec2 model: {e}")
    print("⚠️ Audio feature extraction might fail")

# Initialize Xception with error handling
print("Loading Xception model for inference...")
xception_model_inference = None
try:
    # Create Xception model with explicit input shape and CPU device
    with tf.device('/CPU:0'):
        xception_model_inference = Xception(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(299, 299, 3)
        )
        # Test the model with a dummy input
        dummy_input = np.zeros((1, 299, 299, 3))
        _ = xception_model_inference.predict(dummy_input, verbose=0)
    print("✅ Xception model loaded and tested successfully")
except Exception as e:
    print(f"❌ Error loading Xception model: {e}")
    print("⚠️ Visual feature extraction might fail")

# --- Feature Extraction Functions (adapted from preprocess.py) ---

def extract_audio_features_inference(audio_path):
    """Extract audio features from WAV file with optimized processing"""
    if wav2vec_processor_inference is None or wav2vec_model_inference is None:
        return np.zeros(768)
    try:
        # Check cache first
        cache_key = f"audio_{audio_path}"
        if cache_key in feature_cache:
            return feature_cache[cache_key]

        # Use larger chunks for audio processing
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) == 0: return np.zeros(768)

        # Process in larger segments for better efficiency
        if len(y) > AUDIO_SEGMENT_LENGTH:
            segments = []
            for i in range(0, len(y), AUDIO_SEGMENT_LENGTH):
                segment = y[i:i+AUDIO_SEGMENT_LENGTH]
                inputs = wav2vec_processor_inference(segment, return_tensors="pt", sampling_rate=16000).input_values.to(DEVICE)
                with torch.no_grad():
                    features = wav2vec_model_inference(inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                segments.append(features)

            # Average all segment features
            final_features = np.mean(segments, axis=0)
        else:
            inputs = wav2vec_processor_inference(y, return_tensors="pt", sampling_rate=16000).input_values.to(DEVICE)
            with torch.no_grad():
                final_features = wav2vec_model_inference(inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        if final_features.ndim == 0:
            final_features = np.array([final_features.item()])
        if final_features.shape[0] != 768:
            padded_features = np.zeros(768)
            current_len = final_features.shape[0]
            padded_features[:min(current_len, 768)] = final_features[:min(current_len, 768)]
            final_features = padded_features

        # Cache the result
        feature_cache[cache_key] = final_features.flatten()
        return feature_cache[cache_key]
    except Exception as e:
        print(f"Error extracting audio features for {audio_path}: {e}")
        return np.zeros(768)

def extract_frame_features_inference(video_path):
    """Extracts face features from video frames for inference with optimized processing"""
    if face_cascade is None or xception_model_inference is None:
        print("❌ Required models not loaded")
        return np.zeros((MAX_FRAMES_FOR_INFERENCE, 2048)), 0

    try:
        # Check cache first
        cache_key = f"visual_{video_path}"
        if cache_key in feature_cache:
            return feature_cache[cache_key]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video file {video_path}")
            return np.zeros((MAX_FRAMES_FOR_INFERENCE, 2048)), 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0:
            print("❌ Video has no frames")
            cap.release()
            return np.zeros((MAX_FRAMES_FOR_INFERENCE, 2048)), 0

        print(f"ℹ️ Video info: {total_frames} frames, {fps} FPS")

        # Calculate frame sampling interval
        if fps == 0:
            frame_interval = max(1, int(total_frames / MAX_FRAMES_FOR_INFERENCE)) if total_frames > MAX_FRAMES_FOR_INFERENCE else 1
        elif total_frames > MAX_FRAMES_FOR_INFERENCE:
            frame_interval = max(1, int(total_frames / MAX_FRAMES_FOR_INFERENCE))
        else:
            frame_interval = 1

        features_list = []
        frame_count = 0
        actual_frames_extracted = 0
        max_frames_to_read = max(2000, MAX_FRAMES_FOR_INFERENCE * frame_interval * 2 if frame_interval > 0 else 2000)
        batch_frames_for_xception = []
        
        # Create a figure for displaying frames
        plt.figure(figsize=(15, 5))
        plt.ion()  # Enable interactive mode

        # Pre-allocate face detection parameters
        face_params = {
            'scaleFactor': 1.1,
            'minNeighbors': 5,
            'minSize': (30, 30),
            'flags': cv2.CASCADE_SCALE_IMAGE
        }

        while cap.isOpened() and actual_frames_extracted < MAX_FRAMES_FOR_INFERENCE and frame_count < max_frames_to_read:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, **face_params)

                    if len(faces) > 0:
                        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                        x, y, w, h = largest_face

                        margin = int(0.2 * max(w, h))
                        x1, y1 = max(0, x - margin), max(0, y - margin)
                        x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)

                        face = frame[y1:y2, x1:x2]
                        if face.size == 0:
                            continue

                        face_resized = cv2.resize(face, (299, 299))
                        batch_frames_for_xception.append(face_resized)
                        actual_frames_extracted += 1

                        # Display the extracted face
                        # plt.subplot(1, 5, (actual_frames_extracted - 1) % 5 + 1)
                        # plt.imshow(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
                        # plt.title(f'Frame {actual_frames_extracted}')
                        # plt.axis('off')
                        
                        # # Update display every 5 frames or when we reach MAX_FRAMES_FOR_INFERENCE
                        # if actual_frames_extracted % 5 == 0 or actual_frames_extracted == MAX_FRAMES_FOR_INFERENCE:
                        #     plt.tight_layout()
                        #     plt.draw()
                        #     plt.pause(0.1)

                        if len(batch_frames_for_xception) >= BATCH_SIZE_FOR_FRAME_PROCESSING or actual_frames_extracted >= MAX_FRAMES_FOR_INFERENCE:
                            if batch_frames_for_xception:
                                try:
                                    # Process batch with optimized TensorFlow function
                                    batch_array = np.array([preprocess_input(f.astype(np.float32)) for f in batch_frames_for_xception])
                                    batch_features = predict_batch(batch_array)
                                    for feat in batch_features:
                                        features_list.append(feat.numpy().flatten())
                                except Exception as e:
                                    print(f"⚠️ Error processing batch: {e}")
                                finally:
                                    batch_frames_for_xception = []
                except Exception as e:
                    print(f"⚠️ Error processing frame {frame_count}: {e}")

            frame_count += 1

        cap.release()
        plt.ioff()  # Disable interactive mode
        plt.close()  # Close the figure

        # Process remaining frames
        if batch_frames_for_xception:
            try:
                batch_array = np.array([preprocess_input(f.astype(np.float32)) for f in batch_frames_for_xception])
                batch_features = predict_batch(batch_array)
                for feat in batch_features:
                    features_list.append(feat.numpy().flatten())
            except Exception as e:
                print(f"⚠️ Error processing final batch: {e}")

        if not features_list:
            print("⚠️ No valid frames extracted")
            return np.zeros((MAX_FRAMES_FOR_INFERENCE, 2048)), 0

        features_array = np.array(features_list)
        num_extracted_in_list = features_array.shape[0]
        feature_dim = features_array.shape[1] if num_extracted_in_list > 0 else 2048

        # Pad or truncate to MAX_FRAMES_FOR_INFERENCE
        if num_extracted_in_list < MAX_FRAMES_FOR_INFERENCE:
            padding = np.zeros((MAX_FRAMES_FOR_INFERENCE - num_extracted_in_list, feature_dim))
            features_array = np.concatenate((features_array, padding), axis=0)
        elif num_extracted_in_list > MAX_FRAMES_FOR_INFERENCE:
            features_array = features_array[:MAX_FRAMES_FOR_INFERENCE, :]

        # Cache the result
        feature_cache[cache_key] = (features_array, num_extracted_in_list)
        print(f"✅ Successfully extracted {num_extracted_in_list} frames")
        return features_array, num_extracted_in_list

    except Exception as e:
        print(f"❌ Error in frame extraction: {e}")
        return np.zeros((MAX_FRAMES_FOR_INFERENCE, 2048)), 0

def extract_features(video_path):
    """
    Extracts visual and audio features from a single video file for inference.
    Returns:
        visual_features (np.ndarray): Shape (1, MAX_FRAMES_FOR_INFERENCE, 2048)
        audio_features (np.ndarray): Shape (1, MAX_FRAMES_FOR_INFERENCE, 768)
        visual_length (int): Actual number of visual frames extracted.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return np.zeros((1, MAX_FRAMES_FOR_INFERENCE, 2048)), np.zeros((1, MAX_FRAMES_FOR_INFERENCE, 768)), 0

    temp_dir_obj = None
    visual_features_np = np.zeros((MAX_FRAMES_FOR_INFERENCE, 2048))
    audio_features_np = np.zeros(768)
    actual_visual_len = 0

    try:
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir_path = temp_dir_obj.name
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_audio_path = os.path.join(temp_dir_path, f"{base_name}.wav")

        with VideoFileClip(video_path, verbose=False) as video_clip:
            if video_clip.audio is not None:
                try:
                    video_clip.audio.write_audiofile(temp_audio_path,
                                                   verbose=False,
                                                   logger=None,
                                                   codec='pcm_s16le')
                except Exception as e_audio_extract:
                    print(f"MoviePy failed to extract audio: {e_audio_extract}. Creating silent audio.")
                    sf.write(temp_audio_path, np.zeros(16000 * 1, dtype=np.int16), 16000, subtype='PCM_16')
            else:
                sf.write(temp_audio_path, np.zeros(16000 * 1, dtype=np.int16), 16000, subtype='PCM_16')

        audio_features_np = extract_audio_features_inference(temp_audio_path)
        visual_features_np, actual_visual_len = extract_frame_features_inference(video_path)

        # Ensure correct dimensions
        if visual_features_np.ndim == 2:
            visual_features_np = np.expand_dims(visual_features_np, axis=0)

        # Reshape audio features to match visual sequence length
        if audio_features_np.ndim == 1:
            audio_features_np = np.expand_dims(audio_features_np, axis=0)
            audio_features_np = np.expand_dims(audio_features_np, axis=0)
            audio_features_np = np.repeat(audio_features_np, MAX_FRAMES_FOR_INFERENCE, axis=1)
        elif audio_features_np.ndim == 2:
            audio_features_np = np.expand_dims(audio_features_np, axis=1)
            audio_features_np = np.repeat(audio_features_np, MAX_FRAMES_FOR_INFERENCE, axis=1)

    except Exception as e:
        print(f"Error during feature extraction for {video_path}: {e}")
        visual_features_np = np.zeros((1, MAX_FRAMES_FOR_INFERENCE, 2048))
        audio_features_np = np.zeros((1, MAX_FRAMES_FOR_INFERENCE, 768))
        actual_visual_len = 0

    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()

    return visual_features_np, audio_features_np, actual_visual_len


# --- Model Definition and Loading ---
application_model = None
try:
    # Initialize model with same parameters as training
    application_model = MultimodalDeepfakeDetector(
        hidden_dim=256,
        audio_hidden_dim=128,
        use_attention=True
    )

    # Load model weights
    model_path = "models/DFModel_b64_ep200_ac93.pth"
    if os.path.exists(model_path):
        # Load state dict and ensure it's on the correct device
        state_dict = torch.load(model_path, map_location=DEVICE)
        application_model.load_state_dict(state_dict)
        application_model.to(DEVICE)
        application_model.eval()  # Set to evaluation mode
        print(f"✅ Model loaded successfully from {model_path}")
        print(f"Model device: {next(application_model.parameters()).device}")
    else:
        print(f"❌ Model weights not found at {model_path}")
        application_model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    application_model = None


# --- Inference function ---
def predict_video(video_path_param):
    """
    Predict if a video is real or fake with optimized processing.
    Returns:
        dict with keys: predicted_label, confidence, frames (base64), frames_analyzed, processing_time, model_confidence
    """
    if application_model is None:
        print("❌ Model not loaded. Cannot make predictions.")
        return None

    if not os.path.exists(video_path_param):
        print(f"❌ Video file not found: {video_path_param}")
        return None

    try:
        start_time = time.time()
        # Extract features
        print(f"📥 Extracting features from {video_path_param}...")
        visual_features, audio_features, actual_visual_len = extract_features(video_path_param)

        if actual_visual_len == 0:
            print("⚠️ No valid frames extracted from video")
            return None

        # Ensure correct dimensions and move to device
        visual_features_tensor = torch.tensor(visual_features, dtype=torch.float32).to(DEVICE)
        audio_features_tensor = torch.tensor(audio_features, dtype=torch.float32).to(DEVICE)

        # Make prediction with torch.no_grad() for efficiency
        with torch.no_grad():
            output = application_model(visual_features_tensor, audio_features_tensor)
            probability = output.item()
            predicted_class = 1 if probability > 0.5 else 0
            confidence = probability if predicted_class == 1 else 1 - probability

        # Extract frames as images (base64)
        # We'll extract the same faces as in extract_frame_features_inference
        frames_b64 = []
        cap = cv2.VideoCapture(video_path_param)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            frame_interval = max(1, int(total_frames / MAX_FRAMES_FOR_INFERENCE)) if total_frames > MAX_FRAMES_FOR_INFERENCE else 1
        elif total_frames > MAX_FRAMES_FOR_INFERENCE:
            frame_interval = max(1, int(total_frames / MAX_FRAMES_FOR_INFERENCE))
        else:
            frame_interval = 1
        frame_count = 0
        actual_frames_extracted = 0
        while cap.isOpened() and actual_frames_extracted < MAX_FRAMES_FOR_INFERENCE:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                    if len(faces) > 0:
                        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                        x, y, w, h = largest_face
                        margin = int(0.2 * max(w, h))
                        x1, y1 = max(0, x - margin), max(0, y - margin)
                        x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
                        face = frame[y1:y2, x1:x2]
                        if face.size == 0:
                            continue
                        face_resized = cv2.resize(face, (160, 160))
                        _, buffer = cv2.imencode('.jpg', face_resized)
                        b64 = base64.b64encode(buffer).decode('utf-8')
                        frames_b64.append(f"data:image/jpeg;base64,{b64}")
                        actual_frames_extracted += 1
                except Exception as e:
                    print(f"⚠️ Error extracting frame for API: {e}")
            frame_count += 1
        cap.release()
        processing_time = time.time() - start_time
        # Pad/truncate frames_b64
        if len(frames_b64) < MAX_FRAMES_FOR_INFERENCE:
            frames_b64 += [None] * (MAX_FRAMES_FOR_INFERENCE - len(frames_b64))
        elif len(frames_b64) > MAX_FRAMES_FOR_INFERENCE:
            frames_b64 = frames_b64[:MAX_FRAMES_FOR_INFERENCE]
        return {
            "predicted_label": predicted_class,
            "confidence": confidence,
            "frames": frames_b64,
            "frames_analyzed": actual_frames_extracted,
            "processing_time": processing_time,
            "model_confidence": confidence
        }
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None
    
# --- Example usage ---
if __name__ == "__main__":
    # Check if all required models are loaded
    feature_models_loaded = all([
        face_cascade is not None,
        wav2vec_model_inference is not None,
        xception_model_inference is not None
    ])

    if not feature_models_loaded:
        print("❌ One or more feature extraction models failed to load")
        exit(1)

    if application_model is None:
        print("❌ Application model could not be loaded")
        exit(1)

    # Test prediction
    sample_video_path = "/content/fv5.mp4"
    if not os.path.exists(sample_video_path):
        print(f"⚠️ Sample video not found: {sample_video_path}")
        print("Attempting to create a dummy video for testing...")
        try:
            from moviepy.editor import ColorClip, AudioArrayClip
            clip_duration, clip_fps = 2, 10
            dummy_audio_array = np.zeros((int(clip_duration * 16000), 1), dtype=np.int16)
            video_clip_dummy = ColorClip(size=(640, 480), color=(0,0,0), duration=clip_duration)
            audio_clip_dummy = AudioArrayClip(dummy_audio_array, fps=16000)
            final_clip_dummy = video_clip_dummy.set_audio(audio_clip_dummy)
            final_clip_dummy.write_videofile(sample_video_path, codec="libx264", audio_codec="aac", fps=clip_fps, logger=None)
            for clip_obj in [video_clip_dummy, audio_clip_dummy, final_clip_dummy]:
                clip_obj.close()
            print(f"✅ Dummy video created: {sample_video_path}")
        except Exception as e:
            print(f"❌ Could not create dummy video: {e}")
            exit(1)

    if os.path.exists(sample_video_path):
        print(f"\n🔍 Starting prediction for video: {sample_video_path}")
        predicted_result = predict_video(sample_video_path)

        if predicted_result is not None:
            label_map = {0: "Real", 1: "Fake"}
            print("\n📊 Prediction Results:")
            print(f"Predicted Label: {label_map.get(predicted_result['predicted_label'], 'Unknown')}")
            print(f"Confidence: {predicted_result['confidence']:.2%}")
            print(f"Frames Analyzed: {predicted_result['frames_analyzed']}")
            print(f"Processing Time: {predicted_result['processing_time']:.2f} seconds")
            print(f"Model Confidence: {predicted_result['model_confidence']:.2%}")
        else:
            print("❌ Prediction failed")
    else:
        print(f"❌ Video file {sample_video_path} not found. Cannot proceed with prediction.")
