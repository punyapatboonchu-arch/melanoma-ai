from flask import Flask, render_template, request, make_response, redirect, url_for, session, g
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import os
import json
import uuid
import torch
import torchvision.transforms as transforms
import timm

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-this-secret-key")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

SUPPORTED_LANGUAGES = ("th", "en", "ja", "zh", "ko")
DEFAULT_LANGUAGE = "th"

TRANSLATIONS = {
    "th": {
        "site_title": "Skin Lesion Detection Platform for Melanoma",
        "home": "หน้าแรก", "ai_analysis": "วิเคราะห์ด้วย AI", "hospital": "ค้นหาโรงพยาบาล", "info": "ความรู้เกี่ยวกับ Melanoma",
        "upload_title": "วิเคราะห์ภาพผิวหนังด้วย AI", "upload_label": "อัปโหลดภาพผิวหนัง", "choose_file": "เลือกภาพ", "analyze": "วิเคราะห์ด้วย AI",
        "no_file": "กรุณาเลือกไฟล์ภาพก่อนส่งวิเคราะห์", "invalid_image": "ไฟล์ที่เลือกไม่ใช่ภาพที่รองรับ กรุณาเลือกไฟล์ภาพใหม่",
        "result_title": "ผลการประเมินจาก AI", "risk_level": "ระดับคะแนนจากโมเดล",
        "melanoma_score": "คะแนนจากโมเดลสำหรับ Melanoma", "non_melanoma_score": "คะแนนจากโมเดลสำหรับ Non-melanoma",
        "high": "ระดับคะแนนจากโมเดลสูง", "moderate": "ระดับคะแนนจากโมเดลปานกลาง", "low": "ระดับคะแนนจากโมเดลต่ำ",
        "high_recommendation": "ควรปรึกษาแพทย์หรือแพทย์ผิวหนังโดยเร็วเพื่อรับการประเมินเพิ่มเติม",
        "moderate_recommendation": "ควรปรึกษาบุคลากรทางการแพทย์หากรอยโรคเปลี่ยนแปลงหรือมีข้อกังวล",
        "low_recommendation": "คะแนนจากโมเดลต่ำไม่ได้ตัดความเป็นไปได้ของโรค หากกังวลหรือรอยโรคเปลี่ยนแปลง ควรปรึกษาแพทย์",
        "recommendations": "คำแนะนำ", "disclaimer_title": "ข้อจำกัดการใช้งาน",
        "disclaimer": "เครื่องมือนี้ใช้เพื่อให้ข้อมูลและคัดกรองเบื้องต้นเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์ และไม่ควรใช้แทนการตรวจจากบุคลากรทางการแพทย์",
        "upload_error": "ไม่สามารถประมวลผลไฟล์นี้ได้ กรุณาเลือกภาพ JPG, PNG หรือ WEBP ที่ถูกต้อง",
        "language": "ภาษา", "country_question": "ขณะนี้คุณอยู่ในประเทศไทยหรือไม่?", "yes_thailand": "ใช่ ฉันอยู่ในประเทศไทย",
        "outside_thailand": "ไม่ ฉันอยู่นอกประเทศไทย", "continue": "ดำเนินการต่อ", "language_saved": "บันทึกการตั้งค่าแล้ว"
    },
    "en": {
        "site_title": "Skin Lesion Detection Platform for Melanoma",
        "home": "Home", "ai_analysis": "AI Analysis", "hospital": "Find a Hospital", "info": "Melanoma Information",
        "upload_title": "AI Skin Lesion Analysis", "upload_label": "Upload a skin image", "choose_file": "Choose image", "analyze": "Analyze with AI",
        "no_file": "Please choose an image before submitting.", "invalid_image": "Unsupported image file. Please choose another image.",
        "result_title": "AI Assessment Result", "risk_level": "Model score category",
        "melanoma_score": "Model score for melanoma", "non_melanoma_score": "Model score for non-melanoma",
        "high": "Higher model score", "moderate": "Moderate model score", "low": "Lower model score",
        "high_recommendation": "Please consult a doctor or dermatologist promptly for further evaluation.",
        "moderate_recommendation": "Consult a healthcare professional if the lesion changes or you are concerned.",
        "low_recommendation": "A low model score does not rule out disease. Seek medical advice if the lesion changes or concerns you.",
        "recommendations": "Recommendations", "disclaimer_title": "Important limitation",
        "disclaimer": "This tool provides educational information and preliminary screening support only. It is not a medical diagnosis and does not replace examination by a healthcare professional.",
        "upload_error": "The file could not be processed. Please use a valid JPG, PNG, or WEBP image.",
        "language": "Language", "country_question": "Are you currently in Thailand?", "yes_thailand": "Yes, I am in Thailand",
        "outside_thailand": "No, I am outside Thailand", "continue": "Continue", "language_saved": "Your preferences have been saved."
    },
    "ja": {
        "site_title": "メラノーマ向け皮膚病変検出プラットフォーム",
        "home": "ホーム", "ai_analysis": "AI画像解析", "hospital": "病院を探す", "info": "メラノーマについて",
        "upload_title": "AIによる皮膚画像解析", "upload_label": "皮膚画像をアップロード", "choose_file": "画像を選択", "analyze": "AIで解析",
        "no_file": "解析する画像を選択してください。", "invalid_image": "対応していない画像形式です。別の画像を選択してください。",
        "result_title": "AI評価結果", "risk_level": "モデルスコアの区分",
        "melanoma_score": "メラノーマのモデルスコア", "non_melanoma_score": "非メラノーマのモデルスコア",
        "high": "モデルスコア高", "moderate": "モデルスコア中", "low": "モデルスコア低",
        "high_recommendation": "詳しい評価のため、速やかに医師または皮膚科医に相談してください。",
        "moderate_recommendation": "病変に変化がある場合や心配な場合は、医療従事者に相談してください。",
        "low_recommendation": "スコアが低くても病気を否定できません。変化や心配があれば医師に相談してください。",
        "recommendations": "推奨事項", "disclaimer_title": "重要な注意事項",
        "disclaimer": "このツールは情報提供と初期スクリーニングの補助を目的としており、医学的診断の代わりにはなりません。",
        "upload_error": "画像を処理できませんでした。有効なJPG、PNG、WEBP画像を選択してください。",
        "language": "言語", "country_question": "現在タイにいますか？", "yes_thailand": "はい、タイにいます",
        "outside_thailand": "いいえ、タイ国外にいます", "continue": "続行", "language_saved": "設定を保存しました。"
    },
    "zh": {
        "site_title": "黑色素瘤皮肤病变检测平台",
        "home": "首页", "ai_analysis": "AI 分析", "hospital": "查找医院", "info": "黑色素瘤知识",
        "upload_title": "AI 皮肤病变分析", "upload_label": "上传皮肤图像", "choose_file": "选择图片", "analyze": "使用 AI 分析",
        "no_file": "请先选择图片。", "invalid_image": "不支持此图片格式，请选择其他图片。",
        "result_title": "AI 评估结果", "risk_level": "模型分数类别",
        "melanoma_score": "黑色素瘤模型分数", "non_melanoma_score": "非黑色素瘤模型分数",
        "high": "模型分数较高", "moderate": "模型分数中等", "low": "模型分数较低",
        "high_recommendation": "请尽快咨询医生或皮肤科医生以进行进一步评估。",
        "moderate_recommendation": "如果病变发生变化或您感到担忧，请咨询医疗专业人员。",
        "low_recommendation": "分数较低并不能排除疾病。如有变化或疑虑，请咨询医生。",
        "recommendations": "建议", "disclaimer_title": "重要提示",
        "disclaimer": "本工具仅用于健康信息和初步筛查辅助，不能替代医疗专业人员的检查或诊断。",
        "upload_error": "无法处理该文件。请选择有效的 JPG、PNG 或 WEBP 图片。",
        "language": "语言", "country_question": "您目前在泰国吗？", "yes_thailand": "是，我在泰国",
        "outside_thailand": "不，我在泰国以外", "continue": "继续", "language_saved": "已保存您的设置。"
    },
    "ko": {
        "site_title": "흑색종 피부 병변 감지 플랫폼",
        "home": "홈", "ai_analysis": "AI 분석", "hospital": "병원 찾기", "info": "흑색종 정보",
        "upload_title": "AI 피부 병변 분석", "upload_label": "피부 이미지 업로드", "choose_file": "이미지 선택", "analyze": "AI로 분석",
        "no_file": "분석할 이미지를 선택해 주세요.", "invalid_image": "지원하지 않는 이미지 형식입니다. 다른 이미지를 선택해 주세요.",
        "result_title": "AI 평가 결과", "risk_level": "모델 점수 범주",
        "melanoma_score": "흑색종 모델 점수", "non_melanoma_score": "비흑색종 모델 점수",
        "high": "모델 점수 높음", "moderate": "모델 점수 중간", "low": "모델 점수 낮음",
        "high_recommendation": "추가 평가를 위해 의사 또는 피부과 전문의와 신속히 상담하세요.",
        "moderate_recommendation": "병변이 변하거나 걱정되는 경우 의료 전문가와 상담하세요.",
        "low_recommendation": "점수가 낮아도 질병을 배제할 수 없습니다. 변화나 우려가 있으면 의사와 상담하세요.",
        "recommendations": "권장 사항", "disclaimer_title": "중요 안내",
        "disclaimer": "이 도구는 정보 제공과 초기 선별 지원을 위한 것이며 의료 진단을 대체하지 않습니다.",
        "upload_error": "파일을 처리할 수 없습니다. 올바른 JPG, PNG 또는 WEBP 이미지를 선택하세요.",
        "language": "언어", "country_question": "현재 태국에 있습니까?", "yes_thailand": "네, 태국에 있습니다",
        "outside_thailand": "아니요, 태국 밖에 있습니다", "continue": "계속", "language_saved": "설정이 저장되었습니다."
    }
}

@app.before_request
def load_preferences():
    lang = session.get("language", DEFAULT_LANGUAGE)
    if lang not in SUPPORTED_LANGUAGES:
        lang = DEFAULT_LANGUAGE
    g.lang = lang
    g.t = TRANSLATIONS[lang]
    g.in_thailand = session.get("in_thailand")

@app.context_processor
def inject_template_globals():
    return {
        "t": getattr(g, "t", TRANSLATIONS[DEFAULT_LANGUAGE]),
        "current_language": getattr(g, "lang", DEFAULT_LANGUAGE),
        "in_thailand": getattr(g, "in_thailand", None),
        "supported_languages": SUPPORTED_LANGUAGES
    }

@app.route("/language", methods=["GET", "POST"])
def language():
    if request.method == "POST":
        lang = request.form.get("language", DEFAULT_LANGUAGE)
        country = request.form.get("in_thailand", "yes")
        session["language"] = lang if lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
        session["in_thailand"] = (country == "yes")
        session.permanent = True
        return redirect(url_for("landing"))
    return render_template("language.html", translations=TRANSLATIONS)

@app.route("/set-language/<lang>")
def set_language(lang):
    if lang in SUPPORTED_LANGUAGES:
        session["language"] = lang
    return redirect(request.referrer or url_for("landing"))

@app.route("/")
def landing():
    response = make_response(render_template("landing.html", disclaimer=g.t["disclaimer"]))
    if not request.cookies.get("cookie_consent"):
        response.set_cookie("cookie_consent", "accepted", max_age=60 * 60 * 24 * 365,
                            httponly=True, samesite="Lax")
    return response

@app.route("/questionnaire", methods=["GET", "POST"])
def questionnaire():
    result = None
    message = None
    if request.method == "POST":
        file = request.files.get("image")
        if not file or not file.filename:
            message = g.t["no_file"]
        else:
            filename = secure_filename(file.filename)
            extension = os.path.splitext(filename)[1].lower()
            if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
                message = g.t["invalid_image"]
            else:
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}{extension}")
                try:
                    file.save(filepath)
                    with Image.open(filepath) as check:
                        check.verify()
                    melanoma_prob, non_melanoma_prob, melanoma_percent, non_melanoma_percent = predict_image(filepath)
                    if melanoma_prob >= 0.70:
                        level_key, color = "high", "#c43d42"
                    elif melanoma_prob >= 0.40:
                        level_key, color = "moderate", "#b66b13"
                    else:
                        level_key, color = "low", "#2d8062"
                    recommendation = g.t[f"{level_key}_recommendation"]
                    result = f"""
                    <section class="ai-result" style="line-height:1.8">
                      <h2>{g.t['result_title']}</h2>
                      <p><strong>{g.t['risk_level']}:</strong> <span style="color:{color};font-weight:700">{g.t[level_key]}</span></p>
                      <div class="score-panel">
                        <p><strong>{g.t['melanoma_score']}:</strong> {melanoma_percent:.2f}%</p>
                        <div class="score-bar"><div style="width:{max(0,min(100,melanoma_percent)):.2f}%;background:#c43d42"></div></div>
                        <p><strong>{g.t['non_melanoma_score']}:</strong> {non_melanoma_percent:.2f}%</p>
                        <div class="score-bar"><div style="width:{max(0,min(100,non_melanoma_percent)):.2f}%;background:#2d8062"></div></div>
                      </div>
                      <h3>{g.t['recommendations']}</h3>
                      <p>{recommendation}</p>
                      <aside class="medical-disclaimer"><strong>{g.t['disclaimer_title']}</strong><br>{g.t['disclaimer']}</aside>
                    </section>"""
                except (UnidentifiedImageError, OSError, ValueError):
                    message = g.t["upload_error"]
                finally:
                    try:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                    except OSError:
                        app.logger.warning("Could not remove temporary upload file")
    return render_template("questionnaire.html", result=result, message=message)

@app.route("/hospital")
def hospital():
    if session.get("in_thailand") is not True:
        return redirect(url_for("language"))
    with open("hospitals.json", "r", encoding="utf-8") as f:
        hospitals = json.load(f)
    return render_template("hospital.html", hospitals=hospitals)

@app.route("/info")
def info():
    return render_template("info.html")

@app.route("/health")
def health():
    return {"status": "ok"}

# Keep the existing EfficientNet-B4 inference pipeline and preprocessing.
device = torch.device("cpu")
model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
TEMPERATURE = 8.0

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output / TEMPERATURE, dim=1)
        melanoma_prob = probs[0][0].item()
        non_melanoma_prob = probs[0][1].item()
        return melanoma_prob, non_melanoma_prob, melanoma_prob * 100, non_melanoma_prob * 100

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
