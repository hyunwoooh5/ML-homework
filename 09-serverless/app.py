from io import BytesIO
from urllib import request

from PIL import Image

import numpy as np
import torch
from torchvision import transforms

import onnx
import onnxruntime as ort


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def predict(img):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor_img = preprocess(img)

    onnx_model = onnx.load('hair_classifier_empty.onnx')

    input_names = [node.name for node in onnx_model.graph.input]
    output_names = [node.name for node in onnx_model.graph.output]

    input_name = input_names[0]

    input_data = {input_name : np.expand_dims(tensor_img.numpy(), axis=0)}

    sess = ort.InferenceSession('hair_classifier_empty.onnx')

    results = sess.run(output_names, input_data)

    return results

if __name__ == "__main__":
    img = prepare_image(download_image("https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"), (200, 200))
    
    output = predict(img)
    print(f"Model output: {output}")