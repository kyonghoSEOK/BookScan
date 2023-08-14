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

import point

from pytesseract import Output
from pytesseract import *
from PIL import Image

pytesseract.tesseract_cmd = R'C:\Program Files\Tesseract-OCR\tesseract'

from pathlib import Path
from typing import Union, Literal, List

from PyPDF2 import PdfWriter, PdfReader

################################################################################################################

def show(text,img):
    cv2.imshow(text,img)
    cv2.waitKey()
    # cv2.destroyAllWindows()

################################################################################################################

def findCont(image):
    # print(image.shape)
    # image = cv2.GaussianBlur(image,(9,9),0)
    copy_img = image.copy()
    # ��ü �ν��Ϸ��� ��ü�� ����̾�� �ϹǷ� inv�� ����ȭ �Ѵ�.
    bin_img = cv2.adaptiveThreshold(copy_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,41,5)
    
    copy_img = cv2.cvtColor(copy_img,cv2.COLOR_GRAY2BGR)
    # ��ü ����
    # cv2.imshow('bin_img',bin_img)
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    #################### �ܰ��� ���� #########################
    for i in range(len(contours)):
        # �ּ� ���� ȸ���� ���簢�� ���
        retval = cv2.minAreaRect(contours[i])
        # �ּ� ���� ���簢���� �ʺ�� ����
        width, height = retval[1]
        if width * height < 7000:
            # cv2.drawContours(copy_img, contours, i, (0,0,255), 1)
            cv2.drawContours(bin_img, contours, i, 0, 3)

    # cv2.imshow('copy_img_changed',copy_img)   
    # cv2.imshow('bin_img_changed',bin_img)

    ############## ����ȭ �̹��� �������� ���� ################
    kernel_size = (5, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    bin_img = cv2.morphologyEx(bin_img,cv2.MORPH_OPEN ,kernel)
    bin_img = cv2.morphologyEx(bin_img,cv2.MORPH_ERODE ,kernel)    
    # cv2.imshow('dst_copied_2',bin_img)
    
    # ���� ã�Ƴ���
    edge = cv2.Canny(bin_img, 80, 150)
    
    # show(edge)
    # ���� �̹��� / �ȼ� / ��ȯ ���� / �Ӱ谪 / �ּ� ���� / ���а��� (ũ�� �����Ѱ��� �ν�)
    lines = cv2.HoughLinesP(edge, 1, math.pi / 1800, 210, minLineLength = 400 , maxLineGap= 900)

    # ���� �׸���
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(copy_img, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
    # show('lines',copy_img)

    ############### Lines �׷�ȭ ###############
    if lines is not None:
        # 1.1������ ���⸦ ����
        # print(lines)

        final_lines = []
        check = [ 0 for _ in range(len(lines))]
        ###### Lines �� ������ �ܰ����� ����� ������ tmp�� �ְ� ����� 
        ###### final_lines�� �ִ´�
        for i in range(len(lines)):
            if check[i] != 0:
                continue
            check[i] = 1

            x1, y1, x2, y2 = lines[i][0]
            rad = math.atan2(y2 - y1, x2 - x1)
            rad = math.degrees(rad)
            
            if x1 == x2:
                m = 1000
                c = x1
            else: 
                m = (y2 - y1) / (x2 - x1 + 1e-6)  # ���� ���� �и� ���Ͽ� 0���� ������ ���� ����
                c = y1 - m * x1
            
            # print(rad)

            tmp = [lines[i][0]] 

            for j in range(len(lines)):
                if check[j] != 0:
                    continue

                x3, y3, x4, y4 = lines[j][0]
                rad2 = math.atan2(y4 - y3, x4 - x3)
                rad2 = math.degrees(rad2)

                distances = []
                distances.append( get_distance(x3,y3,m,c) if m!=1000 else c-x3 )
                distances.append( get_distance((x3+x4)/2,(y3+y4)/2,m,c) if m!=1000 else c-(x3+x4)/2 )
                distances.append( get_distance(x4,y4,m,c) if m!=1000 else c-x4 )

                # ��ġ�� ���� ��� ������ �������� Ȯ��
                if all(distance < 50 for distance in distances) and abs(abs(rad) - abs(rad2)) < 5:
                    tmp.append(lines[j][0])                
                    check[j] = 1

            # ������ ������ �����ϴ� ���, ��� ����Ͽ� final_lines�� �߰�
            avg_line = np.mean(tmp, axis=0, dtype=np.int32)
            x5, y5, x6, y6 = avg_line
            avg_rad = math.atan2(y6 - y5, x6 - x5)
            avg_rad = math.degrees(avg_rad)
            final_lines.append([avg_line,avg_rad])

        # ���� �׸���
        for line in final_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(copy_img, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
        
        # print(*final_lines, sep='\n')
        # show('final_lines',copy_img)

        return final_lines
    
    else:
        return print('err no lines are found')

################################################################################################################

def get_distance(x,y,m,c):
    return abs(y - m * x - c) / math.sqrt(1 + m**2)

################################################################################################################

def minimizing(lines):
    check = [0 for _ in range(len(lines))]
    minimized = []
    cluster = []
    tmp =[]
    for i in range(len(lines)):
        if check[i] != 0:
            continue
        tmp.clear()
        check[i] = 1
        tmp.append(lines[i])
        for j in range(len(lines)):
            if check[j] != 0:
                continue
            if abs(abs(lines[i][1])-abs(lines[j][1])) < 10:
                check[j] = 1
                tmp.append(lines[j])
        cluster.append(tmp.copy())
    # print("buf cluster")
    # print(*cluster,sep='\n')

    for i in range(len(cluster)):
        further_line = []
        dist = 0
        if(len(cluster[i])==2):
            continue
        else:
        #################################################
            for j in range(len(cluster[i])):
                x1, y1, x2, y2 = cluster[i][j][0]
                if x1 == x2:
                    m = 1000
                    c = x1
                else: 
                    m = (y2 - y1) / (x2 - x1 + 1e-6)  # ���� ���� �и� ���Ͽ� 0���� ������ ���� ����
                    c = y1 - m * x1
                #################################################
                for k in range((len(cluster[i]))):
                    if j <= k:
                        continue
                    tmp_dist = get_distance(x=(cluster[i][k][0][0]+cluster[i][k][0][2])/2,\
                                            y=(cluster[i][k][0][1]+cluster[i][k][0][3])/2,\
                                            m=m,c=c)
                    
                    if dist < tmp_dist and (abs(abs(cluster[i][k][1])-abs(cluster[i][j][1])) < 4):
                        dist = tmp_dist
                        further_line.clear()
                        further_line.append(cluster[i][j])
                        further_line.append(cluster[i][k])
            
            cluster[i] = further_line.copy()
            further_line.clear()
    # print("last cluster")
    # print(*cluster,sep='\n')
    final_lines = []
    for i in range(len(cluster)):
        for j in range(len(cluster[i])):
                final_lines.append(cluster[i][j])

    # print(*final_lines, sep='\n')
    return final_lines

################################################################################################################
## ��ǥ : ǥ�� �̹����� ��� ����ŷ�Ͽ� �ٽ� �����ϰ� ��ȯ�Ͽ� ��ġ������ �Բ� �����ϰ�
## �۾��� �ٽ� ���� �ش� �̹����� ��ġ�� ���ؼ� �����ְ� �̹��� ���� �����Ѵ�.
## ������ : å�� �ִ� ��ü�� �ƴҶ� ���� ���־�� �Ѵ�.
## ����� ����ŷ�ϴ� ��ɸ� ����
def find_image(image):
    copy_img = image.copy()
    # ��ü �ν��Ϸ��� ��ü�� ����̾�� �ϹǷ� inv�� ����ȭ �Ѵ�.
    gmin, gmax, _, _ = cv2.minMaxLoc(copy_img)
    copy_img = cv2.convertScaleAbs(copy_img, alpha=255.0/(gmax - gmin), beta =  -gmin*255.0/(gmax-gmin))
    show('hist',copy_img)
    bin_img = cv2.adaptiveThreshold(copy_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,9)
    
    show('oribin',bin_img)
    # cv2.imshow('bin_img',bin_img)
    sorted_points = []
    x_y_point = []
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #################### �ܰ��� ���� #########################
    for i in range(len(contours)):
        color = 100
        if cv2.contourArea(contours[i]) < 160:
            continue
        else:
            approx = cv2.approxPolyDP(contours[i], cv2.arcLength(contours[i],True)*0.03, True)

            vtc = len(approx)

            if vtc == 4:
                print("approx")
                print(approx)
                point_data = np.argsort(approx[:, 0, 0])
                point = approx[point_data].squeeze().tolist()
                print("point")
                print(point)

                buf_x_y=[]
                # ���� �Ʒ�, ���� �� ��ǥ ����
                left_bottom = max(point[:2], key=lambda p: p[1])
                left_top = min(point[:2], key=lambda p: p[1])

                buf_x_y.append(min(point[:2], key=lambda p: p[0])[0])

                # ������ �Ʒ�, ������ �� ��ǥ ����
                right_bottom = max(point[2:], key=lambda p: p[1])
                right_top = min(point[2:], key=lambda p: p[1])

                buf_x_y.append(max(point[2:], key=lambda p: p[0])[0])

                buf_x_y.append(min(point[:], key=lambda p: p[1])[1])
                buf_x_y.append(max(point[:], key=lambda p: p[1])[1])

                sorted_points.append([ left_bottom, left_top, right_bottom, right_top ])
                x_y_point.append(buf_x_y)

                print(x_y_point)

                print(sorted_points[0])

                for point in approx:
                    x, y = point[0]
                    cv2.circle(bin_img, (x, y), 5, color + int((y//x)*30), -1)
                    cv2.drawContours(bin_img, contours, i, 0, cv2.FILLED)
            else:
                cv2.drawContours(bin_img, contours, i, 0, cv2.FILLED)

    # for i in range(len(contours)):
    #     # �ּ� ���� ȸ���� ���簢�� ���
    #     retval = cv2.minAreaRect(contours[i])
    #     # �ּ� ���� ���簢���� �ʺ�� ����
    #     width, height = retval[1]
    #     if (width * height > 500):
    #         cv2.drawContours(bin_img, contours, i, 0, cv2.FILLED)
    
    # copy_img = cv2.bitwise_not(copy_img)
    return bin_img, sorted_points, x_y_point

#not use
################################################################################################################

def findLine(image):
    edge = cv2.Canny(image,80,200)
    lines = cv2.HoughLinesP(edge, 1, math.pi / 180, 100, minLineLength =10 , maxLineGap= 20)

    dst = cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1])
            pt2 = (lines[i][0][2], lines[i][0][3])
            cv2.line(dst, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)

    return dst