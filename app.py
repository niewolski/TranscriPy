import os
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
from transcripy import Transcriber

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# use absolute paths for better windows compatibility
base_dir = Path(__file__).parent.absolute()
app.config['UPLOAD_FOLDER'] = str(base_dir / 'uploads')
app.config['OUTPUT_FOLDER'] = str(base_dir / 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500mb max file size

# allowed file extensions
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'wav', 'm4a'}

def allowed_file(filename):
    # check if file extension is allowed
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# create upload and output directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    # main page with upload form
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # handle file upload and transcription
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'file type not allowed. use: mp3, mp4, wav, m4a'}), 400
    
    upload_path = None
    try:
        # save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # save file
        file.save(upload_path)
        
        # verify file was saved
        if not os.path.exists(upload_path):
            return jsonify({'error': 'failed to save uploaded file'}), 500
        
        # convert upload path to absolute path
        upload_path = os.path.abspath(upload_path)
        
        # verify file exists before transcription
        if not os.path.exists(upload_path):
            return jsonify({'error': 'uploaded file does not exist'}), 500
        
        # transcribe file
        transcriber = Transcriber(model_size="base")
        output_filename = Path(filename).stem + '.txt'
        output_path = os.path.abspath(os.path.join(app.config['OUTPUT_FOLDER'], output_filename))
        
        # ensure output directory exists
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
        
        text = transcriber.transcribe(upload_path, output_file=output_path)
        
        # cleanup uploaded file
        if os.path.exists(upload_path):
            os.remove(upload_path)
        
        return jsonify({
            'success': True,
            'filename': output_filename,
            'text_length': len(text)
        }), 200
        
    except Exception as e:
        # cleanup on error
        if upload_path and os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except:
                pass
        
        # log full error for debugging
        import traceback
        error_details = traceback.format_exc()
        print(f"transcription error: {str(e)}")
        print(f"traceback: {error_details}")
        
        return jsonify({'error': f'transcription failed: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    # download transcribed text file
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(filename))
        if not os.path.exists(file_path):
            return jsonify({'error': 'file not found'}), 404
        
        return send_file(file_path, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

