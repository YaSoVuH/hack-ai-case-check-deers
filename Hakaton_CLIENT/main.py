import flet as ft
import os
import glob
import time
import config as cnf
import shutil

import keras
from matplotlib import pyplot as plt
from keras import backend as K
from keras.layers import Dense, Activation, Dropout, Reshape, Permute, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils, MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image

assets_dir=os.getcwd()

DIRECTORY_WITH_PHOTO_TO_CLASSIFICATE = f"{assets_dir}\\{cnf.FOLDER_WITH_PHOTO_TO_CLASSIFICATE}"
DIRECTORY_WITH_PHOTO_CLASSIFICATED = f"{assets_dir}\\{cnf.FOLDER_WITH_PHOTO_CLASSIFICATED}"

if not os.path.exists(DIRECTORY_WITH_PHOTO_TO_CLASSIFICATE):
    os.makedirs(DIRECTORY_WITH_PHOTO_TO_CLASSIFICATE)

if not os.path.exists(DIRECTORY_WITH_PHOTO_CLASSIFICATED):
    os.makedirs(DIRECTORY_WITH_PHOTO_CLASSIFICATED)

mobile = keras.applications.mobilenet.MobileNet()

base_model = MobileNet(weights='imagenet', include_top=False)

classes_count = 3
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
preds = Dense(classes_count, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

class_names = cnf.CLASS_NAMES

def load_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = preprocess_input(img_tensor)
    return img_tensor

def main(page: ft.Page):
    page.title = "FIND DEERS"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.adaptive = True
    page.theme_mode = "LIGHT"

    def processing_photos(e):
        RowArray[0] = loading

        jpg_files = glob.glob(DIRECTORY_WITH_PHOTO_TO_CLASSIFICATE)

        for i in range(1, len(jpg_files) + 1):
            RowArray.append(ft.Text(f"Ожидайте, идёт обработка фотографий {i}/{len(jpg_files)}"))
            RowArray.append(ft.Text(f"Сейчас обрабатывается фотография: {jpg_files[i-1].split("\\")[-1]}"))

            new_image = load_image(jpg_files[i-1])
            pred = model.predict(new_image)
            class_idx = np.argmax(pred, axis=1)[0]
            class_name = class_names[class_idx]
            class_dir = os.path.join(DIRECTORY_WITH_PHOTO_CLASSIFICATED, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            shutil.move(jpg_files[i-1], class_dir)
            pred = model.predict(new_image)
            print("File:", jpg_files[i-1].split("\\")[-1], "Prediction:", pred)

            page.update()
            time.sleep(1)
            del RowArray[1:3]
        
        RowArray[0] = buttonStart
        page.update()
    
    buttonStart = ft.TextButton('ОБНАРУЖИТЬ ОЛЕНЕЙ', on_click=processing_photos, icon=ft.icons.SEARCH, icon_color=ft.colors.RED)
    loading = ft.ProgressRing()

    RowArray = [buttonStart]

    page.add(
        ft.Row(
            [
            ft.Column(
                RowArray,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER
            )
        ],
        alignment=ft.MainAxisAlignment.CENTER
        )
    )

ft.app(main, assets_dir=assets_dir)