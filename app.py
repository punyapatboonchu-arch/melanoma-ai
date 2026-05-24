from flask import Flask, render_template, request
import os
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm

# =========================================
# Flask App
# =========================================

app = Flask(__name__)

# =========================================
# Upload Folder
# =========================================

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================================
# Device
# =========================================

device = torch.device("cpu")

# =========================================
# Load Model
# =========================================

model = timm.create_model(
    "efficientnet_b4",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(
    torch.load(
        "model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("✅ Model loaded successfully!")

# =========================================
# Transform
# =========================================

transform = transforms.Compose([

    transforms.Resize((400, 400)),

    transforms.CenterCrop(380),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )

])

# =========================================
# Settings
# =========================================

TEMPERATURE = 8.0

# =========================================
# Prediction Function
# =========================================

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image_tensor)

        probs = torch.softmax(
            output / TEMPERATURE,
            dim=1
        )

        melanoma_prob = probs[0][0].item()

    return melanoma_prob

# =========================================
# Landing Page
# =========================================

@app.route("/")
def landing():

    disclaimer = """
    This system is intended to support early screening
    and encourage users to seek medical advice
    from specialists.
    """

    return render_template(
        "landing.html",
        disclaimer=disclaimer
    )

# =========================================
# Questionnaire
# =========================================

@app.route("/questionnaire", methods=["GET", "POST"])
def questionnaire():

    result = None

    disclaimer = """
    This system is intended to support early screening
    and encourage users to seek medical advice
    from specialists.
    """

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename != "":

            # =========================================
            # Save File
            # =========================================

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(filepath)

            # =========================================
            # Predict
            # =========================================

            melanoma_prob = predict_image(filepath)

            # =========================================
            # Risk Score
            # =========================================

            percent = melanoma_prob * 100

            # =========================================
            # Risk Assessment
            # =========================================

            if melanoma_prob >= 0.70:

                risk_level = "High Risk"
                color = "red"

            elif melanoma_prob >= 0.40:

                risk_level = "Moderate Risk"
                color = "orange"

            else:

                risk_level = "Low Risk"
                color = "green"

            # =========================================
            # Result HTML
            # =========================================

            result = f"""

            <div style='line-height:1.8; text-align:center;'>

                <div style='
                    font-size:52px;
                    font-weight:bold;
                    color:{color};
                '>
                    {percent:.2f}%
                </div>

                <br>

                <div style='
                    font-size:32px;
                    font-weight:bold;
                    color:{color};
                '>
                    {risk_level}
                </div>

            </div>

            """

    return render_template(
        "questionnaire.html",
        result=result,
        disclaimer=disclaimer
    )

# =========================================
# Hospital Page
# =========================================

@app.route("/hospital")
def hospital():

    disclaimer = """
    This system is intended to support early screening
    and encourage users to seek medical advice
    from specialists.
    """

    with open(
        "hospitals.json",
        "r",
        encoding="utf-8"
    ) as f:

        hospitals = json.load(f)

    return render_template(
        "hospital.html",
        hospitals=hospitals,
        disclaimer=disclaimer
    )

# =========================================
# Info Page
# =========================================

@app.route("/info")
def info():

    disclaimer = """
    This system is intended to support early screening
    and encourage users to seek medical advice
    from specialists.
    """

    return render_template(
        "info.html",
        disclaimer=disclaimer
    )

# =========================================
# Run Flask
# =========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )