from flask import Flask, request, render_template, redirect, send_from_directory, url_for
import os
from subprocess import CalledProcessError, run

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    shift_factors = request.form.get('shift_factors', '')  # Get shift factors from form
    use_equation = request.form.get('use_equation', 'false').lower() == 'true'  # Get boolean for using equation

    if file.filename == '':
        return redirect(request.url)
    
    if file and file.filename.endswith('.csv'):
        uploaded_filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(uploaded_filepath)

        # Handle CSV files
        app.logger.info(f'Uploaded CSV file: {uploaded_filepath}')

        # Check if the Python script exists in the same directory as app.py
        fixed_python_script_filename = 'converted_script.py'
        fixed_python_script_path = os.path.join(os.path.dirname(__file__), fixed_python_script_filename)

        if os.path.exists(fixed_python_script_path):
            try:
                # Run the Python Script with the uploaded CSV file path and shift factors
                run_script(fixed_python_script_path, uploaded_filepath, shift_factors, use_equation)
                
                # Redirect to view graphs or another page
                return redirect(url_for('show_graphs'))
            except Exception as e:
                app.logger.error(f'Error during script execution: {e}')
                return 'Error executing script', 500
    
    return redirect(request.url)

def run_script(script_path, csv_file_path, shift_factors, use_equation):
    try:
        app.logger.info(f'Running script {script_path} with file {csv_file_path}')
        # Execute the Python script with the CSV file path and shift factors as arguments
        result = run(['python', script_path, csv_file_path, shift_factors, str(use_equation)], check=True, text=True, capture_output=True)
        app.logger.info(f'Script output: {result.stdout}')
    except CalledProcessError as e:
        app.logger.error(f'Error executing script: {e.stderr}')
        raise

@app.route('/graphs')
def show_graphs():
    # List all graph images in the upload folder
    graphs = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.png')]
    return render_template('graphs.html', graphs=graphs)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=8000)