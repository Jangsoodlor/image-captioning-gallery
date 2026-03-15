from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import torch
import logging
import json

from image_captioning import allowed_file, load_model, generate_caption

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

# Path to captions metadata file
CAPTIONS_FILE = os.path.join(app.config["UPLOAD_FOLDER"], "captions.json")


def load_captions():
    """Load captions from JSON metadata file"""
    if os.path.exists(CAPTIONS_FILE):
        try:
            with open(CAPTIONS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading captions file: {str(e)}")
            return {}
    return {}


def save_captions(captions):
    """Save captions to JSON metadata file"""
    try:
        with open(CAPTIONS_FILE, "w") as f:
            json.dump(captions, f, indent=2)
    except IOError as e:
        logger.error(f"Error saving captions file: {str(e)}")
        raise


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    """Serve uploaded image files"""
    try:
        filename = secure_filename(filename)
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    except Exception as e:
        logger.error(f"Error serving file: {str(e)}")
        return jsonify({"error": "File not found"}), 404


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
            # Save uploaded file with secure filename
            filename = secure_filename(file.filename)
            # Handle filename collisions by adding timestamp if needed
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            if os.path.exists(filepath):
                name, ext = os.path.splitext(filename)
                import time

                filename = f"{name}_{int(time.time())}{ext}"
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(filepath)
            logger.info(f"File saved: {filepath}")

            # Generate caption
            caption = generate_caption(filepath)
            logger.info(f"Caption generated: {caption}")

            # Load existing captions and add new one
            captions = load_captions()
            captions[filename] = caption
            save_captions(captions)
            logger.info("Caption saved to metadata file")

            return jsonify({"success": True, "filename": filename, "caption": caption})

        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    return jsonify({"error": "Invalid file type. Please upload an image file."}), 400


@app.route("/gallery-data", methods=["GET"])
def get_gallery_data():
    """Get all images and their captions"""
    try:
        captions = load_captions()

        # Get list of actual image files in uploads directory
        images = []
        if os.path.exists(app.config["UPLOAD_FOLDER"]):
            for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
                # Skip captions.json file
                if filename == "captions.json":
                    continue
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                if os.path.isfile(filepath) and allowed_file(filename):
                    caption = captions.get(filename, "No caption available")
                    images.append({"filename": filename, "caption": caption})

        return jsonify({"images": images, "success": True})
    except Exception as e:
        logger.error(f"Error retrieving gallery data: {str(e)}")
        return jsonify({"error": f"Error retrieving gallery data: {str(e)}"}), 500


@app.route("/delete", methods=["POST"])
def delete_image():
    """Delete an image and its caption"""
    try:
        data = request.get_json()
        if not data or "filename" not in data:
            return jsonify({"error": "No filename provided"}), 400

        filename = secure_filename(data["filename"])
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({"error": "Image file not found"}), 404

        # Delete the image file
        os.remove(filepath)
        logger.info(f"Image deleted: {filepath}")

        # Remove caption from metadata
        captions = load_captions()
        if filename in captions:
            del captions[filename]
            save_captions(captions)
            logger.info(f"Caption removed from metadata: {filename}")

        return jsonify({"success": True, "message": "Image deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting image: {str(e)}")
        return jsonify({"error": f"Error deleting image: {str(e)}"}), 500


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

else:
    load_model()
