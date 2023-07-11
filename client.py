import os
import json
from PIL import Image
import argparse
import requests
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", help="filepath of MNIST image", type=str, required=True)
    args = parser.parse_args()

    img = Image.open(args.filepath).convert("L").resize((28, 28))
    img = np.array(img).tolist()

    
    data = json.dumps({"image": img})
    response = requests.post("http://127.0.0.1:5000/predict", data)
    print(response.text)