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
# Temperature Scaling
# =========================================

TEMPERATURE = 3.0

# =========================================
# Prediction Function
# =========================================

def predict_image(image_path):

    # โหลดภาพ
    image = Image.open(image_path).convert("RGB")

    # transform
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        # Predict
        output = model(image_tensor)

        # =========================================
        # Temperature Scaling
        # =========================================

        probs = torch.softmax(
            output / TEMPERATURE,
            dim=1
        )

        # IMPORTANT
        # class 0 = melanoma
        # class 1 = non_melanoma

        melanoma_prob = probs[0][0].item()
        non_melanoma_prob = probs[0][1].item()

        pred_class = torch.argmax(probs, dim=1).item()

        confidence = probs[0][pred_class].item()

    # =========================================
    # Debug
    # =========================================

    print("\n===================================")
    print("RAW OUTPUT:", output)
    print("PROBS:", probs)
    print("Melanoma Probability:", melanoma_prob)
    print("Non-Melanoma Probability:", non_melanoma_prob)
    print("Predicted Class:", pred_class)
    print("Confidence:", confidence)
    print("===================================\n")

    return melanoma_prob, non_melanoma_prob, confidence

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
            # Save Image
            # =========================================

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(filepath)

            # =========================================
            # Predict
            # =========================================

            melanoma_prob, non_melanoma_prob, confidence = predict_image(filepath)

            percent = melanoma_prob * 100

            # =========================================
            # Risk Category
            # =========================================

            if melanoma_prob < 0.30:

                risk_level = "Low Risk"
                recommendation = "ลักษณะความเสี่ยงต่ำ"
                color = "green"

            elif melanoma_prob < 0.60:

                risk_level = "Moderate Risk"
                recommendation = "ควรติดตามอาการหรือปรึกษาแพทย์"
                color = "orange"

            else:

                risk_level = "High Risk"
                recommendation = "ควรพบแพทย์ผู้เชี่ยวชาญ"
                color = "red"

            # =========================================
            # Result HTML
            # =========================================

            result = f"""

            <div style='line-height:1.8;'>

                <h2 style='color:#0a66c2;'>
                    AI Skin Cancer Assessment
                </h2>

                <br>

                ระดับความเสี่ยงจาก AI:

                <br><br>

                <span style='
                    font-size:42px;
                    font-weight:bold;
                    color:{color};
                '>
                    {risk_level}
                </span>

                <br><br>

                คะแนนความเสี่ยง:

                <br>

                <span style='
                    font-size:34px;
                    font-weight:bold;
                    color:{color};
                '>
                    {percent:.2f}%
                </span>

                <br><br>

                คำแนะนำ:

                <br>

                <b style='color:{color};'>
                    {recommendation}
                </b>

                <br><br>

                AI Confidence:
                <b>
                    {confidence * 100:.2f}%
                </b>

                <br><br>

                Temperature Scaling:
                <b>
                    {TEMPERATURE}
                </b>

                <br><br>

                <div style='
                    margin-top:20px;
                    padding:14px;
                    border-radius:12px;
                    background:#f4f8ff;
                    border-left:5px solid #0a66c2;
                    font-size:14px;
                    color:#333;
                '>

                    <b>Disclaimer:</b><br>

                    This AI system is intended to support
                    preliminary screening only and should
                    not be used as a final medical diagnosis.

                    <br><br>

                    Users are encouraged to consult
                    medical professionals for further evaluation.

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