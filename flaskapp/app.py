import torch
from flask import Flask, request, abort, Response, render_template
import cv2
import io
from PIL import Image
from azure.storage.blob import BlobServiceClient  # upload image and video to azure blob
import secrets


# curl -X POST -F "image=@Desktop/yolo_data/test/images/240_F_1192541_cGmwbG3iNBTHZjOCznx78nxbG4jNgL_jpg.rf.ffaa1d90739b21198c04e86e3c4f5968.jpg" http://localhost:5000/object_detection/yolov5m
model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", force_reload=True)

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route("/object_detection/yolov5m", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "image" not in request.files:
            return "No image uploaded.", 400
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        results = model(img, size=640)
        result_array = results.render()[0]

        result_array = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite("yolo.jpg", result_array)

        # Generate a unique filename for the uploaded image
        # filename = secrets.token_hex(8)+".jpg"
        #container_client.upload_blob(name="yolo.jpg", data=result_array.tobytes(), overwrite=True)

        return render_template("upload.html", image_path="yolo.jpg")
    
    return render_template("upload.html")


@app.route("/", methods=["GET"])
def get():
    return {"message": "Hello World!"}      

if __name__ == "__main__":
    app.run()
