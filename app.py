from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, datasets
from PIL import Image
import os, random, io, base64, glob


from PIL import ImageEnhance, ImageOps
import base64, io

# Flask app
app = Flask(__name__)
CORS(app)  # allow frontend JS to call API

# Model & preprocessing
MODEL_SAVE_PATH = "final_model/Final_ASL_Model.pth"
DATASET_DIR = "asl_alphabet_data/asl_alphabet_train/asl_alphabet_train"

# Load dataset class names 
dataset = datasets.ImageFolder(DATASET_DIR)
class_names = dataset.classes
num_classes = len(class_names)

# Build ResNet18 architecture 
model = models.resnet18(weights="IMAGENET1K_V1")  # same as pretrained=True
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load trained weights
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
model.eval()

# Define transforms 
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Routes

@app.route("/")
def home():
    # Serve the HTML file
    return send_from_directory('.', 'index.html')

@app.route("/random", methods=["GET"])
def random_image():
    # Recursively get all images from subfolders
    all_files = glob.glob(os.path.join(DATASET_DIR, "*/*.jpg")) + \
                glob.glob(os.path.join(DATASET_DIR, "*/*.png"))

    if not all_files:
        return jsonify({"error": "No images found in dataset directory."})

    img_path = random.choice(all_files)
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        pred_label = class_names[predicted.item()]

    # Convert image -> base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({
        "image": img_str,
        "prediction": pred_label,
        "filename": os.path.basename(img_path)
    })

#########################################
# 🆕 Real-time Webcam Prediction Route
#########################################
from flask import request  # make sure this import is at the top too!

@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    """
    Accepts a base64-encoded image from frontend webcam.
    Returns the predicted ASL letter.
    Example JSON: {"image": "data:image/png;base64,<...>"}
    """
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No 'image' field found in request"}), 400

        img_b64 = data["image"]
        # Strip prefix like "data:image/png;base64,"
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # --- Improved preprocessing for webcam frames ---
        w, h = img.size
        crop_size = min(w, h)
        left = (w - crop_size) / 2
        top = (h - crop_size) / 2
        right = (w + crop_size) / 2
        bottom = (h + crop_size) / 2
        img = img.crop((left, top, right, bottom)).resize((200, 200))

        # Enhance clarity and contrast slightly
        img = ImageEnhance.Brightness(img).enhance(1.2)
        img = ImageEnhance.Contrast(img).enhance(1.3)
        img = ImageEnhance.Color(img).enhance(1.1)

        # Optional small denoise (helps on webcams)
        img = ImageOps.autocontrast(img)

        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            pred_label = class_names[predicted.item()]

        return jsonify({"prediction": pred_label})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
