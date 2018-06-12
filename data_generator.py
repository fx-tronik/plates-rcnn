# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tablice_model as modellib
import cv2
import numpy as np
import time
import glob
import detekcja
import tensorflow as tf


INPUT_FOLDER = 'tablice_DE/'
INPUT_DIR = 'data_generator/' + INPUT_FOLDER + '/'
CSV_FILE = 'data_generator/tablice_oznaczone.csv'


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True

session1 = tf.Session(config=config)
with session1.as_default():
    model_tablice = modellib.lokalizacja_tablic('mask_rcnn_plates_0100.h5')

session2 = tf.Session(config=config)
with session2.as_default():
    model_ocr = detekcja.predict(session2)

plate_size = (520, 120)
pts_doc = np.array([[0,plate_size[1]], [0,0], [plate_size[0],0], [plate_size[0], plate_size[1]]], dtype = "float32")

plik = open(CSV_FILE, 'w+', encoding='utf-8-sig')
image_path = glob.glob(INPUT_DIR + '*')
liczba_zdjec = len(image_path)
if(liczba_zdjec == 0):
    print("Brak zdjęć w folderze ", INPUT_FOLDER)
    os.sys.exit(0)

print("Liczba zdjęć w zbiorze ", INPUT_FOLDER, ': ', liczba_zdjec)
czas = time.time()
for indeks in range(liczba_zdjec):
    print("\rindeks: ", indeks, end='')
    file_name = image_path[indeks][len(INPUT_DIR)-1:]
    str_nazwa = INPUT_FOLDER + file_name
    str_tablice = ''
    str_lokalizacja = ''

    image = cv2.imread(image_path[indeks], 1)

    with session1.as_default():
        model_pts, r = model_tablice.detect(image)

    #dst = image.copy()
    #for i, p in enumerate(model_pts):
    #    cv2.drawContours(dst,[p],0,(0,255,0),2)   # rysowanie konturu tablicy wykrytej przez model
    #    cv2.putText(dst, str(i), (p[0][0], p[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    #cv2.imwrite('test/' + str(indeks) + '.jpg', dst)

    for i, pts in enumerate(model_pts):
        M = cv2.getPerspectiveTransform(pts.astype(np.float32), pts_doc)
        tablica = cv2.warpPerspective(image, M, plate_size)

        model_znaki = ''
        with session2.as_default():
            model_znaki = model_ocr.ocr(tablica)

        cv2.imwrite('data_generator/tablice_oznaczone/' + model_znaki + '.jpg', tablica)
        str_tablice += model_znaki + ';'

        for p in pts:
            str_lokalizacja += str(p[0]) + "," + str(p[1]) + ";"
        str_lokalizacja = str_lokalizacja[:-1] + '\t'

    plik.write(str_nazwa + '\t' + str_tablice[:-1] + '\t' + str_lokalizacja[:-1] + '\n')

print('\nPrzeanalizowanych zdjęć: ', liczba_zdjec)
print('fps: %.2f' % (liczba_zdjec/(time.time() - czas)))
