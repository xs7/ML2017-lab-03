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

    img_face = []
    print(type(img_face))
    print(type(np.reshape(img_face,len(img_face))))
    for i in range(len(image_list)):
        img_temp = Image.open(image_list[i]).convert('L')
        img_temp.thumbnail((24, 24))
        img_face.append(np.array(img_temp))
    return img_face


#提取图片中的特征
#img_face为图片集，filename为导出特征的文件名
def extractfeature(img_face,filename):
    feature_list = []
    for i in range(len(img_face)):
        f = NPDFeature(img_face[i])
        f.extract()
        feature_list.append(f.features)
    print(len(feature_list))
    #output = open('nonface_feature','wb')
    output = open(filename,'wb')
    pickle.dump(feature_list, output)
    print("dump over")


# 打乱人脸和非人脸数据
def shuffle_array(X,y):
    randomlist = np.arange(X.shape[0])
    np.random.shuffle(randomlist)
    X_random = X[randomlist]
    y_random = y[randomlist]
    return X_random, y_random

if __name__ == "__main__":

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
    result = np.zeros(X_test.shape[0])
    result[y_test_predict == y_test] = 1
    print(sum(result)/result.shape[0])
    target_names = ['face', 'nonface']
    print(classification_report(y_test, y_test_predict, target_names=target_names))
    with open('report.txt', 'w') as f:
        f.write(classification_report(y_test, y_test_predict, target_names=target_names))


