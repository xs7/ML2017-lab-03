import os
import pickle
import numpy as np
from PIL import Image
from feature import NPDFeature
from sklearn.tree import DecisionTreeClassifier
from ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#将图片转化为像素大小为24*24的灰度图
#img_path为相对路径
def dealimage(img_path):
    print("dealimage")
    #img_path = './datasets/original/nonface'
    image_list = [os.path.join(img_path, f) for f in os.listdir(img_path)]
    img = []
    for i in range(len(image_list)):
        img_temp = Image.open(image_list[i]).convert('L')
        img_temp.thumbnail((24, 24))
        img.append(np.array(img_temp))
    return img


#提取图片中的特征
#img_face为图片集，filename为导出特征的文件名
def extractfeature(img_face,filename):
    feature_list = []
    for i in range(len(img_face)):
        f = NPDFeature(img_face[i])
        f.extract()
        feature_list.append(f.features)
    #output = open('nonface_feature','wb')
    output = open(filename, 'wb')
    pickle.dump(feature_list, output)
    print("dump over")


# 打乱人脸和非人脸数据
def shuffle_array(X, y):
    randomlist = np.arange(X.shape[0])
    np.random.shuffle(randomlist)
    X_random = X[randomlist]
    y_random = y[randomlist]
    return X_random, y_random

if __name__ == "__main__":
    #如果特征文件不存在，则重新计算特征得到特征文件
    if os.path.exists('face_feature'):
        print("get face_feature")
    else:
        print("don't found face_feature")
        img_face = dealimage('./datasets/original/face')
        extractfeature(img_face,'face_feature')

    if os.path.exists('nonface_feature'):
        print("get nonface_feature")
    else:
        print("don't found nonface_feature")
        img_nonface = dealimage('./datasets/original/nonface')
        extractfeature(img_nonface,'nonface_feature')

    input = open('nonface_feature', 'rb')
    nonface_data = pickle.load(input)
    X = nonface_data
    y = np.ones(len(X))
    y = y * (-1)
    input = open('face_feature', 'rb')
    face_data = pickle.load(input)
    face_y = np.ones(len(face_data))
    X.extend(face_data)
    y = np.append(y, face_y)
    X = np.array(X)

    X, y = shuffle_array(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    abc = AdaBoostClassifier(DecisionTreeClassifier(), 20)
    abc.fit(X_train, y_train)

    y_test_predict = abc.predict(X_test, 0)

    target_names = ['face', 'nonface']
    print(classification_report(y_test, y_test_predict, target_names=target_names))
    with open('report.txt', 'w') as f:
        f.write(classification_report(y_test, y_test_predict, target_names=target_names))


