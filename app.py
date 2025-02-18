from flask import Flask, render_template, request, redirect, flash

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Route to display the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the file upload and redirect to results
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect('/')
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect('/')
    
    # Check if the file is a CSV
    if file and file.filename.endswith('.csv'):
        flash('File uploaded successfully!')
        # For testing, just send a test message to results.html
        test_data = "Test results: File uploaded and received!"
        return render_template('results.html', test_data=test_data)
    else:
        flash('Invalid file format. Please upload a CSV file.')
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)