from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.core.config import settings


def build_pdf_report(prediction: dict) -> str:
    report_path = settings.upload_path / "reports" / f"{prediction['id']}.pdf"
    c = canvas.Canvas(str(report_path), pagesize=A4)
    c.setTitle("Deepfake Detection Report")
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, 800, "AI-Based Deepfake Detection Report")
    c.setFont("Helvetica", 11)
    lines = [
        f"Filename: {prediction['filename']}",
        f"Prediction: {prediction['result']}",
        f"Confidence: {prediction['confidence']:.2f}%",
        f"Fake Probability: {prediction['fakeProbability']:.4f}",
        f"Real Probability: {prediction['realProbability']:.4f}",
        f"Created At: {prediction['createdAt']}",
    ]
    y = 760
    for line in lines:
        c.drawString(40, y, line)
        y -= 22
    c.save()
    return f"/static/reports/{Path(report_path).name}"
