from __future__ import annotations

import io

from flask import Flask, jsonify, render_template, request, send_file
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from fraud_model import ensure_model_exists, predict_fraud

app = Flask(__name__)


@app.route('/')
def home():
    ensure_model_exists()
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')

    form_data = request.form

    transaction = {
        'step': int(form_data.get('step', 1)),
        'type': form_data.get('type', 'PAYMENT'),
        'amount': float(form_data.get('amount', 0)),
        'oldbalanceOrg': float(form_data.get('oldbalanceOrg', 0)),
        'newbalanceOrig': float(form_data.get('newbalanceOrig', 0)),
        'oldbalanceDest': float(form_data.get('oldbalanceDest', 0)),
        'newbalanceDest': float(form_data.get('newbalanceDest', 0)),
    }

    prediction, probability = predict_fraud(transaction)
    result = 'Fraudulent Transaction' if prediction == 1 else 'Legitimate Transaction'
    percent = round(probability * 100, 2)

    return render_template(
        'result.html',
        result=result,
        probability=percent,
        transaction=transaction,
        prediction=prediction,
    )


@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    form_data = request.form
    transaction = {
        'step': int(form_data.get('step', 1)),
        'type': form_data.get('type', 'PAYMENT'),
        'amount': float(form_data.get('amount', 0)),
        'oldbalanceOrg': float(form_data.get('oldbalanceOrg', 0)),
        'newbalanceOrig': float(form_data.get('newbalanceOrig', 0)),
        'oldbalanceDest': float(form_data.get('oldbalanceDest', 0)),
        'newbalanceDest': float(form_data.get('newbalanceDest', 0)),
    }

    prediction, probability = predict_fraud(transaction)
    result = 'Fraudulent Transaction' if prediction == 1 else 'Legitimate Transaction'
    percent = round(probability * 100, 2)

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle('Fraud Detection Report')
    pdf.setAuthor('Fraud Detection App')

    pdf.setFillColor(HexColor('#1d4ed8'))
    pdf.setFont('Helvetica-Bold', 20)
    pdf.drawString(60, 740, 'Fraud Detection Report')

    pdf.setFillColor(HexColor('#111827'))
    pdf.setFont('Helvetica-Bold', 12)
    pdf.drawString(60, 690, f'Result: {result}')
    pdf.drawString(60, 670, f'Probability: {percent}%')

    lines = [
        ('Step', str(transaction['step'])),
        ('Transaction Type', transaction['type']),
        ('Amount', str(transaction['amount'])),
        ('Old Balance (Origin)', str(transaction['oldbalanceOrg'])),
        ('New Balance (Origin)', str(transaction['newbalanceOrig'])),
        ('Old Balance (Destination)', str(transaction['oldbalanceDest'])),
        ('New Balance (Destination)', str(transaction['newbalanceDest'])),
    ]

    y = 620
    for label, value in lines:
        pdf.setFont('Helvetica', 11)
        pdf.drawString(60, y, f'{label}: {value}')
        y -= 24

    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='fraud_detection_report.pdf',
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json(silent=True) or {}
    transaction = {
        'step': int(payload.get('step', 1)),
        'type': payload.get('type', 'PAYMENT'),
        'amount': float(payload.get('amount', 0)),
        'oldbalanceOrg': float(payload.get('oldbalanceOrg', 0)),
        'newbalanceOrig': float(payload.get('newbalanceOrig', 0)),
        'oldbalanceDest': float(payload.get('oldbalanceDest', 0)),
        'newbalanceDest': float(payload.get('newbalanceDest', 0)),
    }

    prediction, probability = predict_fraud(transaction)
    return jsonify({
        'is_fraud': bool(prediction),
        'prediction': 'fraudulent' if prediction else 'legitimate',
        'probability': round(probability * 100, 2),
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
