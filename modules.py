import random
import cv2 as cv
from tkinter import filedialog
from matplotlib import pyplot as plt
import numpy as np
import math


class MyOperations:
    def open_file1(self):
        filetypes = (
            ('png files', '*.png'),
            ('jpg files', '*.jpg'),
            ('all files', '*.*')
        )
        filename = filedialog.askopenfilename(
            title='open file',
            initialdir='/Downloads',
            filetypes=filetypes
        )
        img = cv.imread(filename)
        img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
        return img

    def save_file(self):
        pass

    def open_file(self):
        img = self.open_file1()
        cv.imshow('img', img)
        cv.waitKey()

    def gray_image(self):  # 灰階影像
        src = self.open_file1()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        cv.imshow('gray', gray)
        cv.waitKey()

    def img_histogram(self):  # 直方圖
        src = self.open_file1()
        cv.imshow('original image', src)
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv.calcHist([src], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.show()

    def hist_equa(self):  # 直方圖等化
        src = self.open_file1()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        img_eq = cv.equalizeHist(gray)
        cv.imshow('equalized image', img_eq)
        plt.hist(img_eq.ravel(), 256, [0, 256])
        plt.show()

    def roi_cut(self):
        src = self.open_file1()
        roi = cv.selectROI(src)   #一個funtion 自動幫你創視窗
        print(roi)                      #看選的部分
        img_cropped = src[int(roi[1]):int(roi[1] + roi[3]),   #y座標
                     int(roi[0]):int(roi[0] + roi[2])]        #x座標
        cv.imshow("Cropped Image", img_cropped)
        cv.waitKey()

    def canny_detector(self):
        max_lowThreshold = 100
        window_name = 'Edge Map'
        title_trackbar = 'Min Threshold:'
        ratio = 3
        kernel_size = 3

        def canny_value_change(val):  # callback
            low_threshold = val
            img_blur = cv.blur(gray, (3, 3))
            detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
            mask = detected_edges != 0  # 是True
            dst = src * (mask[:, :, None].astype(src.dtype))  # src * mask所有數值
            cv.imshow(window_name, dst)

        src = self.open_file1()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        cv.namedWindow(window_name)
        cv.createTrackbar(title_trackbar, window_name, 50, max_lowThreshold, canny_value_change)
        canny_value_change(0)  # 先給數值
        cv.waitKey()

    def thresholding(self):
        max_value = 255
        max_type = 4
        max_binary_value = 255
        trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'  # 種類
        trackbar_value = 'Value'  # trackbar的值
        window_name = 'Threshold Demo'

        def Threshold_Demo(val):
            # 0: 二直化
            # 1: 二直化(相反)
            # 2: 閾值截短
            # 3: 閾值為零
            # 4: 閾值為零(相反)
            threshold_type = cv.getTrackbarPos(trackbar_type, window_name)
            threshold_value = cv.getTrackbarPos(trackbar_value, window_name)
            _, dst = cv.threshold(gray, threshold_value, max_binary_value, threshold_type)  # _不會被呼叫
            cv.imshow(window_name, dst)

        src = self.open_file1()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        cv.namedWindow(window_name)
        cv.createTrackbar(trackbar_type, window_name, 3, max_type, Threshold_Demo)
        cv.createTrackbar(trackbar_value, window_name, 0, max_value, Threshold_Demo)  # 可以選要的數值的trackbar
        Threshold_Demo(0)
        cv.waitKey()

    def hough_transform(self):
        src = cv.cvtColor(self.open_file1(), cv.COLOR_BGR2GRAY)  # 先灰階
        dst = cv.Canny(src, 50, 200, None, 3)  # 零界點,sobel孔大小

        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)  # 轉BGR
        cdstP = np.copy(cdst)  # 複製

        lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)  # 1距離 150門檻值 0預設值

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]  # 算每條線的ab值
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
        linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

        cv.imshow("Source", src)
        cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)  # 標準霍夫線變換
        cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)  # 概率線變換

    def filtering(self):
        window_name = 'filter'
        src = self.open_file1()
        ddepth = -1
        ind = 0
        while True:
            kernel_size = 3 + 2 * (ind % 5)      #卷積核大小
            kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)     #矩陣都設為1
            kernel /= (kernel_size * kernel_size)

            dst = cv.filter2D(src, ddepth, kernel)

            cv.imshow(window_name, dst)
            c = cv.waitKey(1000)         #每一秒卷稽核大小會改變
            if c == 27:                  #按下ESC中止
                break
            ind += 1
        return 0

    def affine(self):
        src = self.open_file1()

        #兩組 3 個點來導出仿射變換關係
        srcTri = np.array([[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]]).astype(np.float32)
        dstTri = np.array([[0, src.shape[1] * 0.33], [src.shape[1] * 0.85, src.shape[0] * 0.25],
                           [src.shape[1] * 0.15, src.shape[0] * 0.7]]).astype(np.float32)
        warp_mat = cv.getAffineTransform(srcTri, dstTri)   #計算仿射變換
        warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))   #計算完的仿射變換應用在圖片上
        center = (warp_dst.shape[1] // 2, warp_dst.shape[0] // 2)  #旋轉的中點
        angle = -50                                                 #旋轉的角度
        scale = 0.6                                                 #旋轉的比例
        rot_mat = cv.getRotationMatrix2D(center, angle, scale)      #選轉矩陣
        warp_rotate_dst = cv.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))    #旋轉應用於我們之前的轉換的輸出
        cv.imshow('Source image', src)
        cv.imshow('平移', warp_dst)
        cv.imshow('旋轉', warp_rotate_dst)
        cv.waitKey()

    def perspective(self):
        src = self.open_file1()
        cv.circle(src, (180, 120), 5, (255, 128, 128), -1)         #(圖片, 圓的位子, 圓大小,顏色, 實心 )
        cv.circle(src, (480, 120), 5, (255, 128, 128), -1)
        cv.circle(src, (120, 580), 5, (255, 128, 128), -1)
        cv.circle(src, (540, 580), 5, (255, 128, 128), -1)

        pts1 = np.float32([[180, 120], [480, 120], [120, 475], [540, 475]])     #第二圖對應的點
        pts2 = np.float32([[0,0], [400, 0], [0, 600], [400, 600]])             #第二圖的大小
        matrix = cv.getPerspectiveTransform(pts1, pts2)                       #轉換

        result = cv.warpPerspective(src, matrix, (400, 600))                  #結果

        cv.imshow('src', src)
        cv.imshow('perspective transform', result)

    def simple_contour(self):
        src = self.open_file1()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        ret, src_thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY) #return value
        contours, hierarchy = cv.findContours(src_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #image,mode,method
        contour_all = cv.drawContours(image=src, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
        cv.imshow('contours', contour_all)                                            #chain code紀錄contour編碼方式
        cv.waitKey()

    def find_contour(self):
        def contour_threshold_callback(val):
            threshold = val
            canny_output = cv.Canny(gray, threshold, threshold * 2)
            contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            #畫contours
            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
            for i in range(len(contours)):
                color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
            cv.imshow('contours', drawing)

        src = self.open_file1()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        gray = cv.blur(gray, (3, 3))

        source_window = 'source image'
        cv.namedWindow(source_window)
        cv.imshow(source_window, src)
        cv.createTrackbar('threshold: ', source_window, 100, 255, contour_threshold_callback)
        contour_threshold_callback(100)
        cv.waitKey()

    def convex_hull(self):
        def thresh_callback(val):
            threshold = val
            # 偵測線
            canny_output = cv.Canny(gray, threshold, threshold * 2)
            contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # 尋找convex hull
            hull_list = []
            for i in range(len(contours)):
                hull = cv.convexHull(contours[i])
                hull_list.append(hull)
            # 畫contours + hull
            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
            for i in range(len(contours)):
                color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                cv.drawContours(drawing, contours, i, color)
                cv.drawContours(drawing, hull_list, i, color)
            cv.imshow('Contours', drawing)

        src = self.open_file1()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        gray = cv.blur(gray, (3, 3))
        source_window = 'source image'
        cv.namedWindow(source_window)
        cv.imshow(source_window, src)
        cv.createTrackbar('threshold: ', source_window, 100, 255, thresh_callback)
        thresh_callback(100)
        cv.waitKey()

    def bounding_box(self):
        def thresh_callback(val):
            threshold = val
            canny_output = cv.Canny(gray, threshold, threshold * 2)

            contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            contours_poly = [None] * len(contours)
            boundRect = [None] * len(contours)
            centers = [None] * len(contours)
            radius = [None] * len(contours)
            for i, c in enumerate(contours):
                contours_poly[i] = cv.approxPolyDP(c, 3, True)
                boundRect[i] = cv.boundingRect(contours_poly[i])
                centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

            for i in range(len(contours)):
                color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                cv.drawContours(drawing, contours_poly, i, color)
                cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
                             (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
                cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

            cv.imshow('Contours', drawing)

        src = self.open_file1()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        gray = cv.blur(gray, (3, 3))
        source_window = 'source image'
        cv.namedWindow(source_window)
        cv.imshow(source_window, src)
        cv.createTrackbar('threshold: ', source_window, 100, 255, thresh_callback)
        thresh_callback(100)
        cv.waitKey()

    def basic_morphology(self):
        src = self.open_file1()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 127, 255, 0)
        erosion_size = 1
        erosion_element = cv.getStructuringElement(cv.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size +1),
                                                   (erosion_size, erosion_size))
        erosion = cv.erode(thresh, erosion_element)
        cv.imshow('erosion', erosion)

        dilation_size = 3
        dilation_element = cv.getStructuringElement(cv.MORPH_RECT, (2 * dilation_size + 1, 2 * dilation_size +1),
                                                   (dilation_size, dilation_size))
        dilation = cv.dilate(erosion, dilation_element)
        cv.imshow('dilation', dilation)

        opening_size = 3
        opening_element = cv.getStructuringElement(cv.MORPH_RECT, (2 * opening_size + 1, 2 * opening_size + 1),
                                                    (opening_size, opening_size))
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, opening_element)
        cv.imshow('opening', opening)

        cv.waitKey()

    def advanced_morphology(self):
        def morph_shape(val):
            if val == 0:
                return cv.MORPH_RECT
            elif val == 1:
                return cv.MORPH_CROSS
            elif val == 2:
                return cv.MORPH_ELLIPSE

        def erosion_callback(val):
            erosion_size = val
            erosion_shape = morph_shape(cv.getTrackbarPos(element_shape, erosion_window))
            element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                                    (erosion_size, erosion_size))
            erosion_result = cv.erode(src, element)
            cv.imshow(erosion_window, erosion_result)

        def dilation_callback(val):
            dilation_size = val
            dilation_shape = morph_shape(cv.getTrackbarPos(element_shape, dilation_window))
            element = cv.getStructuringElement(dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                                    (dilation_size, dilation_size))
            dilation_result = cv.dilate(src, element)
            cv.imshow(dilation_window, dilation_result)

        max_element_size = 3
        max_kernel_size = 21
        element_shape = 'element:\n 0:rectangle\n 1:cross\n 2:ellipse'
        kernel_size = 'kernel size:\n 2n+1'
        erosion_window = 'erosion'
        dilation_window = 'dilation'

        src = self.open_file1()

        cv.namedWindow(erosion_window)
        cv.createTrackbar(element_shape, erosion_window, 0, max_element_size, erosion_callback)
        cv.createTrackbar(kernel_size, erosion_window, 1, max_kernel_size, erosion_callback)

        cv.namedWindow(dilation_window)
        cv.createTrackbar(element_shape, dilation_window, 0, max_element_size, dilation_callback)
        cv.createTrackbar(kernel_size, dilation_window, 1, max_kernel_size, dilation_callback)

        erosion_callback(0)
