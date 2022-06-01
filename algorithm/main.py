import base64
import io
import re

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


def string_to_PIL(json):
    base = base64.b64decode(json['data']['src'])
    return Image.open(io.BytesIO(base)).convert('RGB')

def main(model, json):
    clss = get_clss()
    pil_image = string_to_PIL(json)
    cls_index = count_people(model, pil_image)
    cls = clss[cls_index]
    return cls, cls_index

def get_clss():
    clss = []
    with open('algorithm/classes.txt') as f:
        for line in f:
            clss.append(line.replace('\n', ''))
    return clss

def load_model():
    model = MTCNN()
    return model


def normalize(cls):
    if cls >= 1 and cls < 4:
        return 1
    elif cls > 4:
        return 2
    else:
        return cls

def count_people(model, img):
    sum = 0
    for i in (0, 90, 180, 270):
        img = img.rotate(i, Image.NEAREST, expand = 1)
        _, probs = model.detect(img, landmarks=False)
        new_probs = []
        for prob in probs:
            try:
                if float(prob) > 0.89:
                    new_probs.append(float(prob))
            except:
                break
        sum += len(new_probs)
    cls = round(sum/4)
    print(sum)
    return normalize(cls)

if __name__ == "__main__":
    print(get_clss())