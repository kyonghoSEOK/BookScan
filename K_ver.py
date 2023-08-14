#!/usr/bin/env python
#
# Scanning script for the Noisebridge book scanner.

#!/usr/bin/env python
#
# Scanning script for the Noisebridge book scanner.

import sys
import subprocess
import re
import time
import os
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PyPDF2 import PdfFileReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
pdfmetrics.registerFont(UnicodeCIDFont('HYGothic-Medium'))

# GPHOTO = 'gphoto2'
# VIEW_FILE = 'view.html'
# TMP_FILE = 'view_tmp.html'
# IMG_FORMAT = 'img%05d.jpg'
# TMP_FORMAT = 'tmp%05d.jpg'

import cv2
import numpy as np
import math

import common
import point

from pytesseract import Output
from pytesseract import *
from PIL import Image

pytesseract.tesseract_cmd = R'C:\Program Files\Tesseract-OCR\tesseract'

from pathlib import Path
from typing import Union, Literal, List

from PyPDF2 import PdfWriter, PdfReader
# "img/image2.png", this fucking image deosn't work 
scans = ["img/test1.jpg","img/test2.jpg","img/img_1.jpg","img/test3.jpg","img/image1.jpg","img/image3.jpg","img/image4.jpg","img/image5.jpg"]
k = 0
for img in scans:
    k += 1
    image=cv2.imread(img,cv2.IMREAD_GRAYSCALE) 
    image=cv2.resize(image,(1200,1200),Image.ANTIALIAS) #resizing


    # find contours
    lines = common.findCont(image) 

    # choose 4 lines
    if len(lines) == 4:
        # find 4 points
        points = point.get_points(lines, image)    
    elif len(lines) > 4:
        lines = common.minimizing(lines)
        points = point.get_points(lines, image)
    else:
        print("lack of lines")

    # transfer image
    L_B, L_T, R_B, R_T = points

    src_pts = np.array([L_B, L_T, R_B, R_T], dtype=np.float32)
    dst_pts = np.array([[0, 1000], [0, 0], [800, 1000], [800, 0]], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    transfered_img = cv2.warpPerspective(image, M, (800, 1000))

    #### transfered image ####
    common.show('transfered',transfered_img)
   
    #find images and table
    # only_text_image, objects, x_y_minmax
    only_text, images_in_page, min_max_point = common.find_image(transfered_img)
    for i in range(len(images_in_page)):
        L_B, L_T, R_B, R_T = images_in_page[i]
        min_x, max_x, min_y, max_y = min_max_point[i]

        src_pts = np.array([L_B, L_T, R_B, R_T], dtype=np.float32)
        dst_pts = np.array([[min_x, max_y], [min_x, min_y], [max_x, max_y], [max_x, min_y]], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        part_img = cv2.warpPerspective(transfered_img[min_y:max_y, min_x:max_x], M, (max_x-min_x, max_y-min_y))

        print(part_img.dtype)
        common.show('part_img', part_img)
        image_name =  str(i) + "_result.png"
        print("/partImages/" + image_name)
        cv2.imwrite("partImages/" + image_name, part_img)

    #### masked image ####
    common.show('masked_img',only_text)

    # text extraction
    text = pytesseract.image_to_data(only_text, lang = 'kor+eng',output_type = Output.DICT)
    text_real = pytesseract.image_to_string(only_text, lang = 'kor+eng')
    print( "text" )
    print(text["text"],text["line_num"])

    # print("text_real")
    # print(text_real)

    # write text on img
    for i in range(0, len(text["text"])):
        x = text["left"][i]
        y = text["top"][i]
        w = text["width"][i]
        h = text["height"][i]
        text_value = text["text"][i]
        conf = int(text["conf"][i])
        if conf > 5:
            text_value = "".join([c if ord(c) < 128 else "" for c in text_value]).strip()
            cv2.rectangle(transfered_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(transfered_img, text_value, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

    cv2.imshow('custom window name', transfered_img)
    cv2.waitKey(0)
    
    # make pdf file
    save_path = './maked/terreract'+str(k)+'.pdf'
    pdf_writer = PdfWriter() # empty pdf

    ## make page
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=(800, 1000))
    can.setFont('HYGothic-Medium', 20)
    before_L =0
    tmp_arr =[]
    font_size=[]
    mean_y =[]
    for i in range(0, len(text['text'])):
        if text['conf'][i] > 5:
            
            x = text['left'][i]
            y = 1000 - text['top'][i]
            
            # w = text['width'][i]
            h = text['height'][i]
            line_num = text['line_num'][i]
            
            ## different line
            if x - before_L < 0 :
                mean = sum(font_size,0.0)/len(font_size)
                y_m = sum(mean_y,0.0)/len(mean_y)
                can.setFont('HYGothic-Medium', mean // 5 * 5 + 1)
                for xx, tt in tmp_arr:
                    can.drawString(x=xx, y=y_m, text=tt)
                    
                
                before_L = x
                
                tmp_arr = [[x,text['text'][i]]]
                font_size =[h]
                mean_y = [y]
            ## same line
            else:
                tmp_arr.append([x,text['text'][i]])
                font_size.append(h)
                mean_y.append(y)
                # can.drawString(x=x, y=line_y, text=text['text'][i])
                before_L = x
    for i in range(len(images_in_page)):
        min_x, max_x, min_y, max_y = min_max_point[i]
        can.drawImage("partImages/"+str(i)+"_result.png",min_x, 1000-max_y, max_x-min_x, max_y-min_y )

    can.save()

    # add page to pdf
    packet.seek(0)
    new_pdf = PdfReader(packet)
    page = new_pdf.pages[0]
    pdf_writer.add_page(page)

    ## save pdf
    with open(save_path, 'wb') as f:
        pdf_writer.write(f)
    ### 