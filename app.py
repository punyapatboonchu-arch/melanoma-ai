# Modified app.py
# Result shows only:
# - Melanoma Detected
# - Melanoma Not Detected

from flask import Flask, render_template, request, make_response
from werkzeug.utils import secure_filename
import os
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cpu")

model = timm.create_model(
    "efficientnet_b4",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(
    torch.load("model.pth", map_location=device)
)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

TEMPERATURE = 8.0


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


@app.route("/")
def landing():

    disclaimer = '''
    This system is intended to support early screening
    and encourage users to seek medical advice from specialists.
    '''

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


@app.route("/questionnaire", methods=["GET", "POST"])
def questionnaire():

    result = None

    disclaimer = '''
    ระบบนี้ใช้สำหรับการคัดกรองเบื้องต้นเท่านั้น
    This system is intended for preliminary screening only.
    '''

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

            if melanoma_prob >= 0.50:

                risk_level = '''
                พบความเสี่ยงมะเร็งผิวหนังเมลาโนมา
                <br>
                (Melanoma Detected)
                '''

                recommendation = '''
• ควรเข้าพบแพทย์ผิวหนังหรือผู้เชี่ยวชาญเพื่อรับการประเมินเพิ่มเติม

• ผลลัพธ์นี้เป็นเพียงการคัดกรองเบื้องต้นจาก AI

• Consult a dermatologist for further evaluation.

• This result is intended for screening purposes only.
'''

                color = "#d32f2f"

            else:

                risk_level = '''
                ไม่พบความเสี่ยงมะเร็งผิวหนังเมลาโนมา
                <br>
                (Melanoma Not Detected)
                '''

                recommendation = '''
• ยังไม่พบความเสี่ยงเด่นชัดจากการประเมินของ AI

• ควรตรวจผิวหนังด้วยตนเองอย่างสม่ำเสมอ

• No significant melanoma risk was identified by the AI assessment.

• Continue regular skin self-examinations.
'''

                color = "#2e7d32"

            result = f'''
            <div style="line-height:1.8;">

                <h2 style="color:#0a66c2;">
                    การประเมินมะเร็งผิวหนังด้วย AI
                    <br>
                    (AI Skin Cancer Assessment)
                </h2>

                <br>

                ผลการประเมิน (Assessment Result)

                <br><br>

                <span style="
                    font-size:38px;
                    font-weight:bold;
                    color:{color};
                ">
                    {risk_level}
                </span>

                <br><br>

                คำแนะนำ (Recommendations)

                <br><br>

                <div style="
                    background:#fafafa;
                    padding:15px;
                    border-radius:12px;
                    border-left:5px solid {color};
                    white-space:pre-line;
                ">
                    {recommendation}
                </div>

                <br><br>

                <div style="
                    margin-top:20px;
                    padding:14px;
                    border-radius:12px;
                    background:#f4f8ff;
                    border-left:5px solid #0a66c2;
                    font-size:14px;
                    color:#333;
                ">

                    <b>ข้อจำกัดการใช้งาน (Disclaimer)</b>

                    <br><br>

                    ระบบ AI นี้ถูกพัฒนาขึ้นเพื่อใช้ในการคัดกรองเบื้องต้นเท่านั้น
                    ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์
                    หรือการตรวจโดยแพทย์ผู้เชี่ยวชาญได้

                    <hr>

                    This AI system is intended for preliminary screening only
                    and should not be used as a final medical diagnosis.

                    <br><br>

                    Please consult qualified healthcare professionals
                    for proper diagnosis and treatment.

                </div>

            </div>
            '''

    return render_template(
        "questionnaire.html",
        result=result,
        disclaimer=disclaimer
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


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
