from flask import Flask, render_template, request, make_response
from werkzeug.utils import secure_filename
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
        non_melanoma_prob = probs[0][1].item()

        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    return melanoma_prob, non_melanoma_prob, confidence

# =========================================
# Landing Page
# =========================================

@app.route("/")
def landing():

    disclaimer = """
    This system is intended to support early screening
    and encourage users to seek medical advice from specialists.
    """

    response = make_response(
        render_template(
            "landing.html",
            disclaimer=disclaimer
        )
    )

    if not request.cookies.get("cookie_consent"):
        response.set_cookie(
            "cookie_consent",
            "accepted",
            max_age=60 * 60 * 24 * 365,
            httponly=True,
            samesite="Lax"
        )

    return response

# =========================================
# Questionnaire
# =========================================

@app.route("/questionnaire", methods=["GET", "POST"])
def questionnaire():

    result = None

    disclaimer = """
    ระบบนี้ใช้สำหรับการคัดกรองเบื้องต้นเท่านั้น
    This system is intended for preliminary screening only.
    """

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename != "":

            filename = secure_filename(file.filename)

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(filepath)

            melanoma_prob, non_melanoma_prob, confidence = predict_image(filepath)

            try:
                os.remove(filepath)
            except:
                pass

            # =========================================
            # Risk Assessment
            # =========================================

            if melanoma_prob >= 0.70:

                risk_level = "ความเสี่ยงสูง (High Risk)"

                recommendation = """
• ควรเข้าพบแพทย์ผิวหนังหรือผู้เชี่ยวชาญโดยเร็ว

• หากรอยโรคมีการเปลี่ยนแปลงอย่างรวดเร็ว ควรเข้ารับการตรวจเพิ่มเติม

• Consult a dermatologist or medical specialist promptly.

• Seek medical evaluation if the lesion changes rapidly.
"""

                color = "#d32f2f"

            elif melanoma_prob >= 0.40:

                risk_level = "ความเสี่ยงปานกลาง (Moderate Risk)"

                recommendation = """
• ควรเฝ้าสังเกตการเปลี่ยนแปลงของรอยโรค

• ถ่ายภาพเก็บไว้เพื่อติดตามอาการ

• หากมีอาการผิดปกติควรปรึกษาแพทย์

• Monitor the lesion regularly.

• Consult a healthcare professional if abnormalities occur.
"""

                color = "#f57c00"

            else:

                risk_level = "ความเสี่ยงต่ำ (Low Risk)"

                recommendation = """
• ยังไม่พบความเสี่ยงเด่นชัดจากการประเมินของ AI

• ควรตรวจผิวหนังด้วยตนเองเป็นประจำ

• ใช้ครีมกันแดดและหลีกเลี่ยงแสงแดดจัด

• No significant risk was identified by the AI assessment.

• Continue regular skin self-examinations.
"""

                color = "#2e7d32"

            # =========================================
            # Invalid Image Detection
            # =========================================

            if confidence < 0.60:

                risk_level = "ไม่สามารถประเมินได้ (Unable to Assess)"

                recommendation = """
• ภาพอาจไม่ชัดเจนเพียงพอสำหรับการวิเคราะห์

• กรุณาถ่ายภาพใหม่ในที่มีแสงสว่างเพียงพอ

• ให้รอยโรคอยู่กึ่งกลางภาพ

• The image quality may be insufficient for analysis.

• Please retake the image under adequate lighting conditions.
"""

                color = "#616161"

            result = f"""
            <div style='line-height:1.8;'>

                <h2 style='color:#0a66c2;'>
                    การประเมินความเสี่ยงมะเร็งผิวหนังด้วย AI
                    <br>
                    (AI Skin Cancer Risk Assessment)
                </h2>

                <br>

                ระดับความเสี่ยง (Risk Level)

                <br><br>

                <span style='
                    font-size:42px;
                    font-weight:bold;
                    color:{color};
                '>
                    {risk_level}
                </span>

                <br><br>

                คำแนะนำ (Recommendations)

                <br><br>

                <div style='
                    background:#fafafa;
                    padding:15px;
                    border-radius:12px;
                    border-left:5px solid {color};
                    white-space:pre-line;
                '>
                    {recommendation}
                </div>

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

                    <b>ข้อจำกัดการใช้งาน (Disclaimer)</b>

                    <br><br>

                    ระบบ AI นี้ถูกพัฒนาขึ้นเพื่อใช้ในการคัดกรองเบื้องต้นเท่านั้น
                    ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์
                    หรือการตรวจโดยแพทย์ผู้เชี่ยวชาญได้

                    <br><br>

                    ผู้ใช้งานควรปรึกษาแพทย์หรือบุคลากรทางการแพทย์
                    เพื่อรับการประเมินและวินิจฉัยที่ถูกต้อง

                    <hr>

                    This AI system is intended for preliminary screening only
                    and should not be used as a final medical diagnosis.

                    <br><br>

                    The assessment result does not replace professional
                    medical examination, diagnosis, or treatment.

                    <br><br>

                    Users are strongly encouraged to consult qualified
                    healthcare professionals for further evaluation.

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

# =========================================
# Info Page
# =========================================

@app.route("/info")
def info():

    return render_template("info.html")

# =========================================
# Run Flask
# =========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
