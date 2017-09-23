import cv2
import os
import numpy as np

"Конвертирует все .bmp файлы в .png в некой директории."
"Convert and replace every *.bmp with *.png image in some directory."

path_from = 'path_to_bmp_images'

x = [i[2] for i in os.walk(path_from)]
y = []
for t in x:
    for f in t:
        if f.endswith(".bmp"):
            img = cv2.imread(path_from + '/' + f)
            os.remove(path_from + '/' + f)
            fn_ext = os.path.splitext(f)
            cv2.imwrite(path_from + '/' + fn_ext[0] + '.png', img)
