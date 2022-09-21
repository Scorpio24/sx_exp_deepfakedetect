import math
import random

import cv2
from facenet_pytorch.models.mtcnn import MTCNN
import numpy as np


# 输入图片，得到对应的掩码处理后的结果。
# 图片格式为cv2读取后的格式，shape为（H，W，C），颜色通道顺序为CV2的BGR（这个没什么关系）。
# mask_method='black' or 'noise'，前者为黑色填充，后者为高斯噪声填充。
def get_masked_face_simple(input_img, random_list, mask_method, mask_number):

    if mask_method not in ['black', 'noise']:
        raise Exception("Mask_method should be 'black' or 'noise'. \
                        Now mask_method is {}".format(mask_method))
    if mask_number not in range(0, 9):
        raise Exception("mask_number should be in range(0, 9). \
                        Now mask_number is {}".format(mask_number))

    detector = MTCNN(margin=0, thresholds=[0.65, 0.75, 0.75], device="cpu")
    _, _, landmarks = detector.detect(input_img, landmarks=True)
    if landmarks is None:
        return input_img
    hight = input_img.shape[0]
    wight = input_img.shape[1]
    landmarks = landmarks[0]

    #计算各区域边界。
    eyes_wight = wight * 0.2
    eyes_hight = hight * 0.16
    mouth_wight = landmarks[4][0] - landmarks[3][0]
    mouth_hight = hight * 0.16
    left_eye_left = max(math.ceil(landmarks[0][0] - eyes_wight / 2), 0)
    left_eye_top = max(math.ceil(landmarks[0][1] - eyes_hight / 2), 0)
    left_eye_bottom = min(math.ceil(landmarks[0][1] + eyes_hight / 2), hight)
    right_eye_top = max(math.ceil(landmarks[1][1] - eyes_hight / 2), 0)
    right_eye_right = min(math.ceil(landmarks[1][0] + eyes_wight / 2), wight)
    right_eye_bottom = min(math.ceil(landmarks[1][1] + eyes_hight / 2), hight)
    mouth_left = max(math.ceil(landmarks[3][0] - mouth_wight / 10), 0)
    mouth_right = min(math.ceil(landmarks[4][0] + mouth_wight / 10), wight)
    mouth_bottom = min(math.ceil(landmarks[3][1] + mouth_hight / 2), hight)

    #计算各个区域的掩码区域。
    mask_area1 = np.array([[[left_eye_left, left_eye_bottom], [0, left_eye_bottom], [0, 0], [left_eye_left, 0]]], dtype = np.int32)
    mask_area2 = np.array([[[right_eye_right, min(left_eye_top, right_eye_top)], [left_eye_left, min(left_eye_top, right_eye_top)], [left_eye_left, 0], [right_eye_right, 0]]], dtype = np.int32)
    mask_area3 = np.array([[[wight, right_eye_bottom], [right_eye_right, right_eye_bottom], [right_eye_right, 0], [wight, 0]]], dtype = np.int32)
    mask_area4 = np.array([[[mouth_left, mouth_bottom], [0, mouth_bottom], [0, left_eye_bottom], [mouth_left, left_eye_bottom]]], dtype = np.int32)
    mask_area5 = np.array([[[wight, mouth_bottom], [mouth_right, mouth_bottom], [mouth_right, right_eye_bottom], [wight, right_eye_bottom]]], dtype = np.int32)
    mask_area6 = np.array([[[mouth_left, hight], [0, hight], [0, mouth_bottom], [mouth_left, mouth_bottom]]], dtype = np.int32)
    mask_area7 = np.array([[[mouth_right, hight], [mouth_left, hight], [mouth_left, mouth_bottom], [mouth_right, mouth_bottom]]], dtype = np.int32)
    mask_area8 = np.array([[[wight, hight], [mouth_right, hight], [mouth_right, mouth_bottom], [wight, mouth_bottom]]], dtype = np.int32)
    mask_list = [mask_area1, mask_area2, mask_area3, mask_area4, mask_area5, mask_area6, mask_area7, mask_area8]

    if mask_method == 'black': #随机选择6个区域，使用黑色填充。
        masked = input_img
        for i in random_list[:mask_number]:#超参数：6
            try:
                mask_area = mask_list[i]
                mask = np.full(input_img.shape[:2], 255, dtype = 'uint8')
                cv2.polylines(mask, mask_area, 1, 255)
                cv2.fillPoly(mask, mask_area, 0)
                masked = cv2.bitwise_and(masked, masked, mask=mask)
            except Exception as e:
                print("Mask Part Error: ", e)
    elif mask_method == 'noise': #随机选择6个区域，添加高斯噪声。
        masked = np.array(input_img).copy()
        masked = masked / 255.0
        for i in random_list[:mask_number]:#超参数：6
            try:
                mask_area = mask_list[i]
                mask_shape = (mask_area[0][0][1] - mask_area[0][2][1], mask_area[0][0][0] - mask_area[0][2][0], 3)
                # 产生高斯 noise
                noise = np.random.normal(0, 1, mask_shape)
                # 将噪声和图片叠加
                masked[mask_area[0][2][1]:mask_area[0][0][1], mask_area[0][2][0]:mask_area[0][0][0]] = masked[mask_area[0][2][1]:mask_area[0][0][1], mask_area[0][2][0]:mask_area[0][0][0]] + noise
                # 将超过 1 的置 1，低于 0 的置 0
                masked = np.clip(masked, 0, 1)
            except Exception as e:
                print("Mask Part Error: ", e)
        # 将图片灰度范围的恢复为 0-255
        masked = np.uint8(masked*255)

    return masked

if __name__ == '__main__':
    import yaml
    # 读取配置文件。
    with open("S3D/configs/architecture.yaml", 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    #读取图片。
    img_path = "aug_frames/0c93d0e8-0e55-4a16-acfb-a96cf93b6f68_452.png"
    input_img = cv2.imread(img_path)
    random_list = [i for i in range(0, 8)]
    random.shuffle(random_list)
    masked = get_masked_face_simple(input_img, random_list, config['training']['mask-method'], config['training']['mask-number'])
    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    cv2.imshow('masked', masked)
    cv2.waitKey(0)