from deepface import DeepFace
import pandas as pd
import pickle

##  人脸比对
# result  = DeepFace.verify("an1.jpg", "anzhijie1.jpg")
# results = DeepFace.verify([['img1.jpg', 'img2.jpg'], ['img1.jpg', 'img3.jpg']])
# print("Is verified: ", result)
# print("Is verified: ", result["verified"])


## 检测目标路径下的图片是否和给定的图片人脸相符
# df = DeepFace.find(img_path ="my_db/an1.jpg", db_path ="my_db")
# dfs = DeepFace.find(img_path = ["img1.jpg", "img2.jpg"], db_path = "C:/workspace/my_db")

## 显示人脸匹配的结果
# path = 'my_db/representations_vgg_face.pkl'
# f = open(path, 'rb')
# data = pickle.load(f)
# print(data)

## 表情识别
# obj = DeepFace.analyze(img_path = "my_db/an1.jpg", actions = ['age', 'gender', 'race', 'emotion'])
# print(obj["age"]," years old ",obj["dominant_race"]," ",obj["dominant_emotion"]," ", obj["gender"])
objs = DeepFace.analyze(["my_db/an1.jpg", "my_db/an2.jpg", "my_db/gao1.jpg", "my_db/anzhijie1.jpg"])  # analyzing multiple faces same time
for i in range(len(objs)):
    j = i + 1
    key = "instance_" + str(j)
    print(objs.get(key)["age"], " years old ", objs.get(key)["dominant_race"], " ",
          objs.get(key)["dominant_emotion"], " ", objs.get(key)["gender"])
