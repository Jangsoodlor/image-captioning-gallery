from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import torch
import logging

from image_captioning import allowed_file, load_model, generate_caption, image_to_base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size


# Create upload directory if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global variables for model and processor
processor = None
model = None


@app.route("/")
def index():
    """Main page"""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and generate caption"""
    if "file" not in request.files:
        return jsonify({"error": "No file selected"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Generate caption
            caption = generate_caption(filepath)

            # Convert image to base64 for display
            image_base64 = image_to_base64(filepath)

            # Clean up uploaded file
            os.remove(filepath)

            return jsonify({"caption": caption, "image": image_base64, "success": True})

        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    return jsonify({"error": "Invalid file type. Please upload an image file."}), 400


@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "processor_loaded": processor is not None,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    )


if __name__ == "__main__":
    # Load model on startup
    try:
        load_model()
        print("🚀 Starting Flask application...")
        print("📝 Model loaded successfully!")
        print("🌐 Open http://localhost:5000 in your browser")
        app.run(debug=True, host="0.0.0.0", port=5000)
    except Exception as e:
        print(f"❌ Failed to start application: {str(e)}")
        print("Make sure you have the required dependencies installed:")
        print("pip install flask torch transformers pillow")
