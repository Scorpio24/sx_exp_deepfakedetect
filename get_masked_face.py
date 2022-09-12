import math
import random

import cv2
import face_alignment
import numpy as np


#输入图片，得到对应的掩码处理后的结果。
#图片格式为cv2读取后的格式，shape为（H，W，C），颜色通道顺序已从BGR转换为RGB。
def get_masked_face(input_img):
    #默认使用GPU，如果要在CPU上运行，则需要添加参数device='cpu'。
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)
    #并获取人脸关键点。
    preds = fa.get_landmarks(input_img)

    #计算各区域边界。
    left_eye_left = math.ceil(preds[0][36, 0] - (preds[0][39, 0] - preds[0][36, 0])*0.1)
    left_eye_top = math.ceil(min(preds[0][37, 1], preds[0][38, 1]) - (max(preds[0][41, 1], preds[0][40, 1]) - min(preds[0][37, 1], preds[0][38, 1]))*0.1)
    left_eye_bottom = math.ceil(max(preds[0][41, 1], preds[0][40, 1]) + (max(preds[0][41, 1], preds[0][40, 1]) - min(preds[0][37, 1], preds[0][38, 1]))*0.1)
    right_eye_top = math.ceil(min(preds[0][43, 1], preds[0][44, 1]) - (max(preds[0][47, 1], preds[0][46, 1]) - min(preds[0][43, 1], preds[0][44, 1]))*0.1)
    right_eye_right = math.ceil(preds[0][45, 0] + (preds[0][45, 0] - preds[0][42, 0])*0.1)
    right_eye_bottom = math.ceil(max(preds[0][47, 1], preds[0][46, 1]) + (max(preds[0][47, 1], preds[0][46, 1]) - min(preds[0][43, 1], preds[0][44, 1]))*0.1)
    mouth_left = math.ceil(preds[0][48, 0] - (preds[0][54, 0] - preds[0][48, 0])*0.1)
    mouth_right = math.ceil(preds[0][54, 0] + (preds[0][54, 0] - preds[0][48, 0])*0.1)
    mouth_bottom = math.ceil(preds[0][57, 1] + (preds[0][57, 1] - min(preds[0][50, 1], preds[0][52, 1]))*0.1)

    #计算各个区域的掩码区域。
    mask_area1 = np.array([[[left_eye_left, left_eye_bottom], [0, left_eye_bottom], [0, 0], [left_eye_left, 0]]], dtype = np.int32)
    mask_area2 = np.array([[[right_eye_right, min(left_eye_top, right_eye_top)], [left_eye_left, min(left_eye_top, right_eye_top)], [left_eye_left, 0], [right_eye_right, 0]]], dtype = np.int32)
    mask_area3 = np.array([[[input_img.shape[:2][1], right_eye_bottom], [right_eye_right, right_eye_bottom], [right_eye_right, 0], [input_img.shape[:2][1], 0]]], dtype = np.int32)
    mask_area4 = np.array([[[mouth_left, mouth_bottom], [0, mouth_bottom], [0, left_eye_bottom], [mouth_left, left_eye_bottom]]], dtype = np.int32)
    mask_area5 = np.array([[[input_img.shape[:2][1], mouth_bottom], [mouth_right, mouth_bottom], [mouth_right, right_eye_bottom], [input_img.shape[:2][1], right_eye_bottom]]], dtype = np.int32)
    mask_area6 = np.array([[[mouth_left, input_img.shape[:2][0]], [0, input_img.shape[:2][0]], [0, mouth_bottom], [mouth_left, mouth_bottom]]], dtype = np.int32)
    mask_area7 = np.array([[[mouth_right, input_img.shape[:2][0]], [mouth_left, input_img.shape[:2][0]], [mouth_left, mouth_bottom], [mouth_right, mouth_bottom]]], dtype = np.int32)
    mask_area8 = np.array([[[input_img.shape[:2][1], input_img.shape[:2][0]], [mouth_right, input_img.shape[:2][0]], [mouth_right, mouth_bottom], [input_img.shape[:2][1], mouth_bottom]]], dtype = np.int32)
    mask_list = [mask_area1, mask_area2, mask_area3, mask_area4, mask_area5, mask_area6, mask_area7, mask_area8]

    #随机选择6个区域，使用黑色填充。
    random.shuffle(mask_list)
    masked = input_img
    for mask_area in mask_list[0:6]:#超参数：6
        mask = np.full(input_img.shape[:2], 255, dtype = 'uint8')
        cv2.polylines(mask, mask_area, 1, 255)
        cv2.fillPoly(mask, mask_area, 0)
        masked = cv2.bitwise_and(masked, masked, mask=mask)
    #cv2.imshow(masked)

    #随机选择6个区域，添加高斯噪声。
    random.shuffle(mask_list)
    masked = np.array(input_img).copy()
    masked = masked / 255.0
    for mask_area in mask_list[0:6]:#超参数：6
        mask_shape = (mask_area[0][0][1] - mask_area[0][2][1], mask_area[0][0][0] - mask_area[0][2][0], 3)
        # 产生高斯 noise
        noise = np.random.normal(0, 1, mask_shape)
        # 将噪声和图片叠加
        masked[mask_area[0][2][1]:mask_area[0][0][1], mask_area[0][2][0]:mask_area[0][0][0]] = masked[mask_area[0][2][1]:mask_area[0][0][1], mask_area[0][2][0]:mask_area[0][0][0]] + noise
        # 将超过 1 的置 1，低于 0 的置 0
        masked = np.clip(masked, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    masked = np.uint8(masked*255)
    #cv2.imshow(masked)

if __name__ == 'main':
    #读取图片。
    img_path = "test.jpg"
    input_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    get_masked_face(input_img)