from flask import Flask, render_template, request
import os
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm

app = Flask(__name__)

# -------------------------------
# Upload Folder
# -------------------------------
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------
# AI Model Setup
# -------------------------------
device = torch.device("cpu")

model = timm.create_model(
    "efficientnet_b4",
    pretrained=False,
    num_classes=2
)

# โหลดโมเดล
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully!")

# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Prediction Function
# -------------------------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

        # แปลง logits -> probability
        probs = torch.softmax(output, dim=1)

        # class index 1 = Melanoma
        risk_prob = probs[0][1].item()

    return risk_prob

# -------------------------------
# Routes
# -------------------------------

@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/questionnaire", methods=["GET", "POST"])
def questionnaire():

    result = None

    # Threshold สำหรับคัดกรอง
    threshold = 0.225

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename != "":

            # Save file
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(filepath)

            # Predict
            risk_prob = predict_image(filepath)

            # Convert to percent
            percent = risk_prob * 100

            # ระดับความเสี่ยง
            if percent >= 50:
                level = "เสี่ยงปานกลาง - เสี่ยงสูง"
                color = "red"

            elif risk_prob >= threshold:
                level = "เสี่ยงต่ำ - เสี่ยงปานกลาง"
                color = "orange"

            else:
                level = "เสี่ยงต่ำ"
                color = "green"

            # Result HTML
            result = f"""
            <div style='line-height:1.8;'>

            <h3 style='color:#0a66c2;'>
            ผลการวิเคราะห์จาก AI
            </h3>

            มีความเสี่ยงเป็น
            <b>Melanoma</b>

            <br>

            <span style='
                font-size:32px;
                font-weight:bold;
                color:{color};
            '>
                {percent:.2f}%
            </span>

            <br><br>

            ระดับการประเมิน:
            <b style='color:{color};'>
                {level}
            </b>

            </div>
            """

    return render_template(
        "questionnaire.html",
        result=result
    )


@app.route("/hospital")
def hospital():

    with open(
        "hospitals.json",
        "r",
        encoding="utf-8"
    ) as f:

        hospitals = json.load(f)

    return render_template(
        "hospital.html",
        hospitals=hospitals
    )


@app.route("/info")
def info():
    return render_template("info.html")


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)