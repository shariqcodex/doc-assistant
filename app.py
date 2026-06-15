from flask import Flask, request, jsonify, render_template_string
from groq import Groq
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer, util

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

document_chunks = []
chunk_embeddings = None

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Document Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #f3f3ef; font-family: 'Segoe UI', sans-serif; height: 100vh; display: flex; flex-direction: column; }
        #header { padding: 16px 24px; background: #f3f3ef; border-bottom: 1px solid #e0e0d8; font-size: 17px; font-weight: 600; }
        #main { flex: 1; max-width: 780px; width: 100%; margin: 0 auto; padding: 30px 20px; display: flex; flex-direction: column; gap: 20px; }
        #upload-box { background: white; border: 2px dashed #e0e0d8; border-radius: 16px; padding: 30px; text-align: center; }
        #upload-box p { color: #aaa; margin-bottom: 12px; }
        #file-input { display: none; }
        .btn { background: #1a1a1a; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-size: 14px; }
        .btn:hover { background: #333; }
        #status { font-size: 13px; color: #888; margin-top: 10px; }
        #chat { flex: 1; display: flex; flex-direction: column; gap: 16px; }
        .message { max-width: 80%; padding: 12px 16px; border-radius: 16px; font-size: 15px; line-height: 1.6; }
        .user { background: #e8e8e2; align-self: flex-end; border-radius: 16px 16px 4px 16px; }
        .ai { background: white; align-self: flex-start; border-radius: 16px 16px 16px 4px; border: 1px solid #e0e0d8; max-width: 85%; }
        #input-wrapper { padding: 16px 20px 24px; background: #f3f3ef; }
        #input-box { max-width: 780px; margin: 0 auto; background: white; border: 1px solid #e0e0d8; border-radius: 16px; padding: 12px 16px; display: flex; gap: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
        #prompt { flex: 1; background: transparent; border: none; color: #1a1a1a; font-size: 15px; outline: none; font-family: 'Segoe UI', sans-serif; }
        #prompt::placeholder { color: #aaa; }
        #send { background: #1a1a1a; color: white; border: none; width: 34px; height: 34px; border-radius: 8px; cursor: pointer; font-size: 16px; }
    </style>
</head>
<body>
    <div id="header">📄 Document Assistant</div>
    <div id="main">
        <div id="upload-box">
            <p>Upload a document to get started</p>
            <button class="btn" onclick="document.getElementById('file-input').click()">Choose File (PDF or TXT)</button>
            <input type="file" id="file-input" accept=".txt,.pdf" onchange="uploadFile(event)">
            <div id="status"></div>
        </div>
        <div id="chat"></div>
    </div>
    <div id="input-wrapper">
        <div id="input-box">
            <input id="prompt" type="text" placeholder="Ask a question about your document...">
            <button id="send" onclick="askQuestion()">↑</button>
        </div>
    </div>

    <script>
        async function uploadFile(event) {
            const file = event.target.files[0];
            if (!file) return;
            document.getElementById('status').innerText = 'Uploading...';
            const formData = new FormData();
            formData.append('file', file);
            const res = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();
            document.getElementById('status').innerText = '✅ ' + file.name + ' uploaded (' + data.characters + ' characters)';
        }

        async function askQuestion() {
            const prompt = document.getElementById('prompt');
            const text = prompt.value.trim();
            if (!text) return;
            addMessage(text, 'user');
            prompt.value = '';
            const res = await fetch('/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: text})
            });
            const data = await res.json();
            addMessage(data.answer, 'ai');
        }

        function addMessage(text, role) {
            const div = document.createElement('div');
            div.className = 'message ' + role;
            div.innerText = text;
            document.getElementById('chat').appendChild(div);
            div.scrollIntoView();
        }

        document.getElementById('prompt').addEventListener('keydown', e => {
            if (e.key === 'Enter') askQuestion();
        });
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(HTML)


@app.route('/upload', methods=['POST'])
def upload():
    global document_chunks, chunk_embeddings
    file = request.files['file']
    filename = file.filename

    if filename.endswith('.txt'):
        text = file.read().decode('utf-8')
    elif filename.endswith('.pdf'):
        import PyPDF2
        import io
        reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    document_chunks = [text[i:i+500] for i in range(0,len(text),500)]
    chunk_embeddings = model.encode(document_chunks, convert_to_tensor=True)

    return jsonify({"status":"uploaded", "chunks": len(document_chunks)})



@app.route('/ask',methods=['POST'])
def ask():
   global document_chunks, chunk_embeddings
   data = request.json
   question = data['question']

   question_emedding = model.encode(question, convert_to_tensor=True)
   scores = util.cos_sim(question_emedding, chunk_embeddings)[0]
   best_chunk = document_chunks[scores.argmax()]

   response = client.chat.completions.create(
       model = "llama-3.3-70b-versatile",
       messages=[ 
           {"role": "system", "content": f"Answer based on this excerpt:\n\n{best_chunk}"},
           {"role": "user", "content":question}


       ]
   )


   answer = response.choices[0].message.content
   return jsonify({"answer":answer})



if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))



    