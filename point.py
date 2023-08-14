#!/usr/bin/env python
#
# Scanning script for the Noisebridge book scanner.

import sys
import subprocess
import re
import time
import os
import io
import random
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

from pytesseract import Output
from pytesseract import *
from PIL import Image

pytesseract.tesseract_cmd = R'C:\Program Files\Tesseract-OCR\tesseract'

from pathlib import Path
from typing import Union, Literal, List

from PyPDF2 import PdfWriter, PdfReader

################################################################################################################
def get_points(lines, image):
    point = set()
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i<=j:
                continue
            if abs(abs(lines[i][1]) - abs(lines[j][1])) > 25:
                point.add(find_intersection(line1=lines[i][0],line2=lines[j][0]))
    
    point = list(point)
    point = sorted(point)
    # ���� �Ʒ�, ���� �� ��ǥ ����
    left_bottom = max(point[:2], key=lambda p: p[1])
    left_top = min(point[:2], key=lambda p: p[1])

    # ������ �Ʒ�, ������ �� ��ǥ ����
    right_bottom = max(point[2:], key=lambda p: p[1])
    right_top = min(point[2:], key=lambda p: p[1])
    sorted_points = [left_bottom,left_top,right_bottom,right_top]

    print("sorted_points")
    print(sorted_points)   
    
    copy_img = image.copy()
    copy_img = cv2.cvtColor(copy_img,cv2.COLOR_GRAY2BGR)
    ### ������ִ� �ڵ�
    for i in range(len(sorted_points)):
        x, y = map(int, sorted_points[i])
        cv2.circle(copy_img, (x, y), 3, (0, 0, 255), -1)
    
    # common.show('dots_img',copy_img)
    ##Ʃ���� ����Ʈ�� ��ȯ�ؼ� ����
    point_list = []
    for sorted_point in sorted_points:
        point_list.append(list(sorted_point))
    return(point_list)
################################################################################################################

def find_intersection(line1, line2):
    # ���� 1�� ������ �Ķ���� ����
    x1, y1, x2, y2 = line1
    if x2 - x1 != 0:
        m1 = (y2 - y1) / (x2 - x1)
        c1 = y1 - m1 * x1
        # ����� ������ ������
        extended_line1 = lambda x: m1 * x + c1
    else:
        # x1�� x2�� ���� ��� ������ �����̹Ƿ� x = c1�� ǥ��
        extended_line1 = lambda x: x1
    
    # ���� 2�� ������ �Ķ���� ����
    x3, y3, x4, y4 = line2
    if x4 - x3 != 0:
        m2 = (y4 - y3) / (x4 - x3)
        c2 = y3 - m2 * x3
        # ����� ������ ������
        extended_line2 = lambda x: m2 * x + c2
    else:
        # x3�� x4�� ���� ��� ������ �����̹Ƿ� x = c2�� ǥ��
        extended_line2 = lambda x: x3
    
    # ����� ������ ���� ���
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1

    return (x, y)

################################################################################################################