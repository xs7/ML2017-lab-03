import os
import pickle
import numpy as np
from PIL import Image
from feature import NPDFeature


def dealimage():
    print("dealimage")
    img_path = './datasets/original/nonface'
    image_list = [os.path.join(img_path, f) for f in os.listdir(img_path)]

    img_face = []
    for i in range(len(image_list)):
        img_temp = Image.open(image_list[i]).convert('L')
        img_temp.thumbnail((24,24))
        img_face.append(np.array(img_temp))
    return img_face


def extractfeature(img_face):
    for i in range(len(img_face)):
        f = NPDFeature(img_face[i])
        f.extract()
        feature_list.append(f.features)
    print(len(feature_list))
    output = open('nonface_feature','wb')
    pickle.dump(feature_list, output)
    print("dump over")


if __name__ == "__main__":
    img_face = dealimage()
    feature_list = []
    input = open('nonface_feature', 'rb')
    data1 = pickle.load(input)

    print(type(data1[0]))



