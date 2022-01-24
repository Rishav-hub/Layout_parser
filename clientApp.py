from main import Detector
from flask import Flask, render_template, request, Response, jsonify
import os
from flask_cors import CORS, cross_origin
from com_ineuron_apparel.com_ineuron_utils.utils import decodeImage

app = Flask(__name__)

detector = Detector(filename="InputImage.JPG")

# RENDER_FACTOR = 35

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

CORS(app)


# @cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "InputImage.JPG"
        # modelPath = 'research/ssd_mobilenet_v1_coco_2017_11_17'
        self.label_parser = Detector(self.filename)

@app.route("/")
def home():
    # return "Landing Page"
    return render_template("index.html")


@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():

    image = request.json['image']
    decodeImage(image, clApp.filename)
    _, result = clApp.label_parser.detect_layout()
    # print(type(result))
    clApp.label_parser.extract_text()
    return jsonify(result)


# port = int(os.getenv("PORT"))
if __name__ == "__main__":
    clApp = ClientApp()
    port = 5000
    app.run(host='127.0.0.1', port=port)
