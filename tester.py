import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tablice_model as modellib
import cv2
import numpy as np
import time
#import glob
#import csv
import detekcja
import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True

#session2 = tf.Session(config=config)
#with session2.as_default():
#    model_ocr = detekcja.predict(session2)
#session1 = tf.Session(config=config)
#with session1.as_default():
#    model_tablice = modellib.lokalizacja_tablic('logs/plates20180607T1021/mask_rcnn_plates_0100.h5')



config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
#g1 = tf.Graph()

#with g1.as_default():
session1 = tf.Session(config=config)
with session1.as_default():
    model_tablice = modellib.lokalizacja_tablic('mask_rcnn_plates_0100.h5')
#g2 = tf.Graph()
#with g2.as_default():
session2 = tf.Session(config=config)
with session2.as_default():
    model_ocr = detekcja.predict(session2)


#model = modellib.lokalizacja_tablic('logs/plates20180607T1021/mask_rcnn_plates_0100.h5')
#test = detekcja.predict()
#logs/plates20180607T0945/mask_rcnn_plates_0002.h5'
#'logs/plates20180423T0856/mask_rcnn_plates_0131.h5' najlepszy model

liczba_zdjec = 5
import csv
plik = open('train/tablice_oznaczone.csv', 'r', encoding='utf-8-sig')
CSVreader = csv.reader(plik, delimiter='\t', quotechar='|')
wiersz = list(CSVreader)

plate_size = (520, 120)

TP = 0
FP = 0
FN = 0
czas = time.time()
for indeks in range(liczba_zdjec):
    col = wiersz[indeks]
    image = cv2.imread('train/' + col[0], 1)

    with session1.as_default():
        model_pts, r = model_tablice.detect(image)
    print("\rindeks: ", indeks)

    val_znaki = col[1].split(';')

    val_pts = []
    for y in range(2, len(col)):
        if(len(col[y]) <= 0):
            continue
        p = col[y].split(';')
        val_pts.append(np.array([x.split(',') for x in p if not x == ''], dtype=np.int32))

    model_znaki = []
    for i, p in enumerate(model_pts):
        pts_doc = np.array([[0,plate_size[1]], [0,0], [plate_size[0],0], [plate_size[0], plate_size[1]]])

        M = cv2.getPerspectiveTransform(p.astype(np.float32), pts_doc.astype(np.float32))
        tablica = cv2.warpPerspective(image, M, plate_size)
        cv2.imwrite('test/' + str(indeks) + "---" + str(i) + '.jpg', tablica)

        with session2.as_default():
            model_znaki.append(model_ocr.ocr(tablica))
        cv2.drawContours(image,[p],0,(0,255,0),2)   # rysowanie konturu tablicy wykrytej przez model
        cv2.putText(image, str(i), (p[0][0], p[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    val_masks = []
    for i,p in enumerate(val_pts):
        cv2.drawContours(image,[p],0,(0,0,255),2)   # rysowanie konturu tablicy ze zbioru walidacyjnego
        cv2.putText(image, str(i), (p[0][0], p[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        mask = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
        cv2.drawContours(mask,[p],0,(1,1,1),-1)   # rysowanie konturu tablicy ze zbioru walidacyjnego na masce
        val_masks.append(mask)

    val_masks = np.array(val_masks)
    val_masks = val_masks[:,:,:,0]
    model_masks = r['masks']
    model_masks = model_masks.swapaxes(0, 2).swapaxes(1, 2)
    if(model_masks.shape[-2:] != val_masks.shape[-2:]):
        model_masks = np.zeros((0, val_masks.shape[1], val_masks.shape[2]))

    for i, v_mask in enumerate(val_masks):
        for i, m_mask in enumerate(model_masks):
            bw_and = np.bitwise_and(m_mask, v_mask)
            bw_or = np.bitwise_or(m_mask, v_mask)
            if(cv2.countNonZero(bw_and) / cv2.countNonZero(bw_or) > 0.5):
                print("val znaki: ", val_znaki[i])
                print("model_znaki: ", model_znaki[i])
                if(val_znaki[i] == model_znaki[i]):
                    TP += 1
                    break
        else:
            FN += 1
    for m_mask in model_masks:
        for v_mask in val_masks:
            bw_and = np.bitwise_and(m_mask, v_mask)
            bw_or = np.bitwise_or(m_mask, v_mask)
            if(cv2.countNonZero(bw_and) / cv2.countNonZero(bw_or) > 0.5):
                break
        else:
            FP += 1

    cv2.imwrite('test/' + str(indeks) + '.jpg', image)

R = TP/(TP+FN)
P = TP/(TP+FP)
Fscore = 2/(1/P + 1/R)
print("TP: ", TP)
print("FP: ", FP)
print("FN: ", FN)
print("Fscore: ", Fscore)
print('Przeanalizowanych zdjęć: ', liczba_zdjec)
print('fps: %.2f' % (liczba_zdjec/(time.time() - czas)))
