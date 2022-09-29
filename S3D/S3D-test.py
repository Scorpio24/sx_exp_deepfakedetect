import argparse
from cProfile import label
import glob
import json
import os
import tempfile
import random
from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool
from os import cpu_count

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from albumentations import (Compose, FancyPCA, GaussianBlur, GaussNoise,
                            HorizontalFlip, HueSaturationValue,
                            ImageCompression, OneOf, PadIfNeeded,
                            RandomBrightnessContrast, Rotate, ShiftScaleRotate,
                            ToGray)
from progress.bar import Bar
from sklearn import metrics
from sklearn.metrics import accuracy_score, auc, f1_score
from torch import einsum, nn
from tqdm import tqdm
from get_masked_face_simple import get_masked_face_simple

from model import S3D
from transforms.albu import IsotropicResize
from utils import (check_correct, custom_round, custom_video_round, get_method,
                   get_n_params, resize, shuffle_dataset, transform_frame)

MODELS_DIR = "models"
BASE_DIR = "./data"
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TEST_DIR = os.path.join(DATA_DIR, "test_set")
OUTPUT_DIR = os.path.join(MODELS_DIR, "tests")

TEST_LABELS_PATH = os.path.join(BASE_DIR, "dataset/dfdc_test_labels.csv")



if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def create_base_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

def snippet_transform(videos, name, config):
    video = list(map(np.asarray, videos))

    if config['training']['mask-method'] != 'none':
        # 目前的方法是对一个视频内的所有人脸图像采取同样的掩码区域。
        random_list = [i for i in range(0, 8)]
        random.shuffle(random_list)
        for i in range(0, len(videos)):
            videos[i] = get_masked_face_simple(
                            videos[i],
                            tempdir,
                            name + "_" + str(i),
                            random_list=random_list, 
                            mask_method=config['training']['mask-method'], 
                            mask_number=config['training']['mask-number'])

            
    #unique = uuid.uuid4()
    #cv2.imwrite("data/dataset/aug_frames/"+str(unique)+"_"+str(index)+".png", video[0])
    
    video = np.concatenate(video, axis=-1)
    video = torch.from_numpy(video).permute(2, 0, 1).contiguous().float()
    video = video.view(1,-1,3,video.size(1),video.size(2)).permute(0,2,1,3,4)

    return video

def save_roc_curves(correct_labels, preds, model_name, accuracy, loss, f1):
  plt.figure(1)
  plt.plot([0, 1], [0, 1], 'k--')

  fpr, tpr, th = metrics.roc_curve(correct_labels, preds)

  model_auc = auc(fpr, tpr)


  plt.plot(fpr, tpr, label="Model_"+ model_name + ' (area = {:.3f})'.format(model_auc))

  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.savefig(os.path.join(OUTPUT_DIR, model_name +  "_" + opt.dataset + "_acc" + str(accuracy*100) + "_loss"+str(loss)+"_f1"+str(f1)+".jpg"))
  plt.clf()

def read_frames(video_path, videos, opt, config):
    
    # Get the video label based on dataset selected
    method = get_method(video_path, DATA_DIR)
    if "Original" in video_path:
        label = 0.
    elif method == "DFDC":
        test_df = pd.DataFrame(pd.read_csv(TEST_LABELS_PATH))
        video_folder_name = os.path.basename(video_path)
        video_key = video_folder_name + ".mp4"
        label = test_df.loc[test_df['filename'] == video_key]['label'].values[0]
    else:
        label = 1.
    

    # Calculate the interval to extract the frames
    frames_number = len(os.listdir(video_path))
    frames_interval = int(frames_number / opt.frames_per_video)
    frames_paths = os.listdir(video_path)
    frames_paths.sort(key=lambda x:int(x.split('_')[0]))
    frames_paths_dict = {}

    # Group the faces with the same index, reduce probabiity to skip some faces in the same video
    for path in frames_paths:
        for i in range(0,1):
            if "_" + str(i) in path:
                if i not in frames_paths_dict.keys():
                    frames_paths_dict[i] = [path]
                else:
                    frames_paths_dict[i].append(path)

    # Select only the frames at a certain interval
    # if frames_interval > 0:
    #     for key in frames_paths_dict.keys():
    #         if len(frames_paths_dict) > frames_interval:
    #             frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]
            
    #         frames_paths_dict[key] = frames_paths_dict[key][:opt.frames_per_video]

    # Select N frames from the collected ones
    video = {}
    snippet = []
    transform = create_base_transform(config['model']['image-size'])
    for key in frames_paths_dict.keys():
        if len(frames_paths_dict[key]) < 20:
            continue
        frames_paths_dict[key] = frames_paths_dict[key][:20]
        for index, frame_image in enumerate(frames_paths_dict[key]):
            #image = np.asarray(resize(cv2.imread(os.path.join(video_path, frame_image)), IMAGE_SIZE))
            image = transform(image=cv2.imread(os.path.join(video_path, frame_image)))['image']
            snippet.append(image)
    if len(snippet) > 0:
        # if key in video:
        #     video[key].extend(snippet)
        # else:
        #     video[key] = [snippet]
        videos.append((snippet, label, video_path))

# Main body
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--model_path', default="S3D_final_DFDC_plan1", type=str, metavar='PATH',
                        help='Path to model checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='DFDC', 
                        help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|DFDC)")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, default="plan3",
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--frames_per_video', type=int, default=30, 
                        help="How many equidistant frames for each video (default: 30)")
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Batch size (default: 32)")
    
    opt = parser.parse_args()
    print(opt)

    if not opt.config:
        raise Exception("please input name of config file by '--config' .")

    # 读取配置文件。
    config_path = os.path.join("S3D/configs", opt.config+".yaml")
    with open(config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    print(config)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    opt.model_path = os.path.join("models/final_models", opt.model_path)
    if os.path.exists(opt.model_path):
        state_dict = torch.load(opt.model_path, map_location=dev)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.lstrip("module.") # remove `module.`
            new_state_dict[name] = v
    else:
        print("No model found.")
        exit()

    num_class = 1
    model = S3D(num_class, config['model']['SRM-net'])
    model.load_state_dict(new_state_dict)
    model.eval()
    model = model.to(dev)

    model_name = os.path.basename(opt.model_path)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    
    preds = []
    mgr = Manager()
    paths = []
    videos = mgr.list()

    if opt.dataset != "DFDC":
        folders = ["Original", opt.dataset]
    else:
        folders = [opt.dataset]

    for folder in folders:
        method_folder = os.path.join(TEST_DIR, folder)  
        for index, video_folder in enumerate(os.listdir(method_folder)):
            paths.append(os.path.join(method_folder, video_folder))

    #for path in paths:
    #    read_frames(path, videos)
      
    with Pool(processes=opt.workers) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(read_frames, videos=videos, opt=opt, config=config),paths):
                pbar.update()

    video_names = np.asarray([row[2] for row in videos])
    correct_test_labels = np.asarray([row[1] for row in videos])
    videos = [row[0] for row in videos]
    preds = []

    tempdir = tempfile.gettempdir()

    bar = Bar('Predicting', max=len(videos))

    f = open(opt.dataset + "_" + model_name + "_labels.txt", "w+")
    for index, video in enumerate(videos):
        video_faces_preds = []
        video_name = video_names[index]
        f.write(video_name)
        #for key in video:
        faces_preds = []
        video_faces = snippet_transform(video, os.path.basename(video_name), config)
        video_faces = video_faces.to(dev).float()
        
        pred = model(video_faces)
        
        scaled_pred = []
        for idx, p in enumerate(pred):
            scaled_pred.append(torch.sigmoid(p))
        faces_preds.extend(scaled_pred)
            
        current_faces_pred = sum(faces_preds)/len(faces_preds)
        face_pred = current_faces_pred.cpu().detach().numpy()[0]
        f.write(" " + str(face_pred))
        video_faces_preds.append(face_pred)
        
        bar.next()
        if len(video_faces_preds) > 1:
            video_pred = custom_video_round(video_faces_preds)
        else:
            video_pred = video_faces_preds[0]
        preds.append([video_pred])
        
        f.write(" --> " + str(video_pred) + "(CORRECT: " + str(correct_test_labels[index]) + ")" +"\n")
        
    f.close()
    bar.finish()

    loss_fn = torch.nn.BCEWithLogitsLoss()
    tensor_labels = torch.tensor([[float(label)] for label in correct_test_labels])
    tensor_preds = torch.tensor(preds)


    loss = loss_fn(tensor_preds, tensor_labels).numpy()

    #accuracy = accuracy_score(np.asarray(preds).round(), correct_test_labels)
    accuracy = accuracy_score(custom_round(np.asarray(preds)), correct_test_labels)

    f1 = f1_score(correct_test_labels, custom_round(np.asarray(preds)))
    print(model_name, "Test Accuracy:", accuracy, "Loss:", loss, "F1", f1)
    save_roc_curves(correct_test_labels, preds, model_name, accuracy, loss, f1)

    for video_name in video_names:
        for i in range(21):
            tempfilename = os.path.join(tempdir, os.path.basename(video_name)+"_"+str(i)+".npy")
            if os.path.exists(tempfilename) is True:
                os.remove(tempfilename)