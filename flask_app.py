from flask import Flask, render_template, request, redirect, url_for
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load model and tokenizer
model_path = "model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Function to predict if text is AI-generated
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    return probabilities

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['GET', 'POST'])
def check():
    if request.method == 'POST':
        text = request.form['text']
        return redirect(url_for('result', text=text))
    return render_template('check.html')

@app.route('/result')
def result():
    text = request.args.get('text', '')
    probabilities = predict(text)
    probabilities_percentage = [prob * 100 for prob in probabilities]
    label = 'Yes' if probabilities[1] > probabilities[0] else 'No'
    result = {
        'text': text,
        'label': label,
        'probabilities': {
            'Human-Written': probabilities_percentage[0],
            'AI-Generated': probabilities_percentage[1]
        }
    }
    return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
