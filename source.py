import time
import os

class data:
    name = ""
    address = ""
    status = ""
    result = ""
    def __init__(self, name, address):
        self.name = name
        self.address = address
        self.status = "undiagnosed"
        self.result = "unknow"
    def show(self):
        print("Name: "+self.name +" address: " +self.address + " status: "+self.status+" result: " + self.result)
    def diagnos(self):
        img = image.load_img(self.address, target_size=(512, 512))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        predict = model.predict(img_tensor)
        if predict >= 0.5:
            print("Dữ liệu của bệnh nhân này: \33[92m không có bệnh \33[0m ")
            self.status = "diagnosed"
            self.result = "\33[92m normal \33[0m"
        else:
            print("Dữ liệu của bệnh nhân này: \33[91m có bệnh 'Inf' ở phổ \33[0m ")
            self.status = "diagnosed"
            self.result = "\33[91m Infiltrate \33[0m"

mData = []
mData.append(data("Nguyen Thanh Test", "test/test.png"))
os.system("cls")
stringOpen = "Hello, welcome"
for char in stringOpen:
    print(char, end="")
    time.sleep(1)

stringOpen = "=================================================="
for char in stringOpen:
    print(char, end="")
    time.sleep(1)
print(" ")
print(" <>Initialization system: ")
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
import keras.utils as image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt
import zipfile
import random
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
print("1. Complete Initialization import libary")
print("2. Initialization model....")
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
print("3. Complete Initialization model.")
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])
print("4. Complete compile model")
model.load_weights('model.h5')
print("5. Complete load weights from the last update")
stringOpen = "=================================================="
for char in stringOpen:
    print(char, end="")
    time.sleep(0.4)
print(" ")
print("Initialization system successfull, with summary on below")
model.summary()
print("Screen will be cleared in 5 second")
time.sleep(5)

while 1:
    os.system("cls")
    print("             Main Menu           ")
    print("\n \n \n \n")
    print("<> Nhập command '/list' để in danh sách các ảnh có sẵn.")
    print("<> Nhập command '/predict' để chẩn đoán bệnh INF.")
    print("<> Nhập command: ", end="")
    command = input()
    if command == "/list":
        os.system("cls")
        while 1:
            counter = 0
            print("=============Danh sách ảnh==============")
            for num in mData:
                counter += 1
                print("<> "+ str(counter) + " ",end="")
                num.show()
            print("Nhập lệnh '/add' để thêm bệnh nhân mới")
            print("Nhập lệnh '/edit' để thay đổi thông tin")
            print("Nhập lênh '/rm' để xóa bỏ dữ liệu của bệnh nhân")
            print("\n \n \n")
            print("Nhập lệnh: ", end= "")
            command = input()
            if command == "/add":
                os.system("cls")
                while 1:
                    counter = 0
                    os.system("cls")
                    print("=============Thêm dữ liệu bệnh nhân mới==============")
                    for num in mData:
                        counter += 1
                        print("<> "+ str(counter) + " ",end="")
                        num.show()
                    print("\n \n \n")
                    print("Nếu muốn thoát xin vui lòng nhập lệnh '/exit' vào ô tên của bệnh nhân")
                    print("Nhập tên bệnh nhân mới: ",end="")
                    name = input()
                    if name == "/exit":
                        break
                    print("Nhập địa chỉ ảnh của bệnh nhân mới", end="")
                    address = input()
                    mData.append(data(name,address))
                    print("<> dữ liệu của bệnh nhân "+name+ " đã được thêm thành công")
            elif command == "/rm":
                print("Nhập số thứ tự của bệnh nhân cần xóa: ",end="")
                number = input()
                mData.pop(number-1)
                print("\nThông tin của bệnh nhân đã được xóa thành công")
                print("Nhấn bấc kỳ phím nào để tiếp tục")
                number = input()
            elif command == "/exit":
                break
    if command == "/predict":
        while 1:
            os.system("cls")
            print("=============Chẩn đoán bệnh==============")
            counter = 0
            print("\n \n \n")
            for num in mData:
                    counter += 1
                    print("<> "+ str(counter) + " ",end="")
                    num.show()
            print("\nNhập số thứ tự của bệnh nhân cần predict: ",end="")
            name = input()
            if name == "/exit":
                break
            else:
                try:
                    name = int(name)
                except:
                    print("Nhập mã số của bệnh nhân không đúng định dạng")
                    print("Nhấn bấc kỳ phiếm nào để tiếp tục")
                    temp = input()
                    continue
                name -= 1
                print("Kết quả chẩn đoán bệnh của bệnh nhân: "+mData[name].name)
                mData[name].diagnos()
                print("Nhấn bấc kỳ phím nào để tiếp tục.")
                img = cv2.imread(mData[name].address)
                cv2.imshow("image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                temp = input()





