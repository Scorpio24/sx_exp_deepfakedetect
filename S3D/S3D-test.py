import argparse
import glob
import json
import os
import tempfile
import random
from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from albumentations import (Compose, PadIfNeeded)
from progress.bar import Bar
from sklearn import metrics
from sklearn.metrics import accuracy_score, auc, f1_score
from tqdm import tqdm
from get_masked_face_simple import get_masked_face_simple

from model import S3D
from CA_S3D import CA_S3D
from msca_S3D import msca_S3D
from msca_S3D import msca_S3D_SRM
from transforms.albu import IsotropicResize
from utils import (custom_round, custom_video_round, get_method)

MODELS_DIR = "models"
BASE_DIR = "./data"
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TEST_DIR = os.path.join(DATA_DIR, "test_set")
OUTPUT_DIR = os.path.join(MODELS_DIR, "tests")
METADATA_PATH = os.path.join(BASE_DIR, "metadata") # Folder containing all training metadata for DFDC dataset
TEST_LABELS_PATH = os.path.join(BASE_DIR, "dataset/dfdc_test_labels.csv")

modelname = {
    "S3D_final_DFDC_plan1":"S3D",
    "S3D_final_DFDC_plan5":"S3D+SRM",
    "S3D_final_DFDC_plan9":"S3D+mask6",
    "S3D_final_DFDC_plan9_2":"S3D+mask8",
    "S3D_final_DFDC_plan9_3":"S3D+mask4",
    "S3D_final_DFDC_plan11":"S3D+SRM+mask6",
    "msca_S3D_final_DFDC_mplan1":"msca_S3D",
    "msca_S3D_final_DFDC_mplan5":"msca_S3D+SRM",
    "msca_S3D_final_DFDC_mplan9":"msca_S3D+mask6",
    "msca_S3D_final_DFDC_mplan9_3":"msca_S3D+mask4",
}

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

    # if config['training']['mask-method'] != 'none':
    #     random_list = [i for i in range(0, 8)]
    #     random.shuffle(random_list)
    #     for i in range(0, len(videos)):
    #         videos[i] = get_masked_face_simple(
    #                         videos[i],
    #                         tempdir,
    #                         name + "_" + str(i),
    #                         random_list=random_list, 
    #                         mask_method=config['training']['mask-method'], 
    #                         mask_number=config['training']['mask-number'])

            
    #unique = uuid.uuid4()
    #cv2.imwrite("data/dataset/aug_frames/"+str(unique)+"_"+str(index)+".png", video[0])
    
    video = np.concatenate(video, axis=-1)
    video = torch.from_numpy(video).permute(2, 0, 1).contiguous().float()
    video = video.view(1,-1,3,video.size(1),video.size(2)).permute(0,2,1,3,4)

    return video

def save_roc_curves(dataset, correct_labels, preds, model_name, accuracy, loss, f1):
  plt.figure(1)
  plt.plot([0, 1], [0, 1], 'k--')

  fpr, tpr, th = metrics.roc_curve(correct_labels, preds)

  model_auc = auc(fpr, tpr)


  plt.plot(fpr, tpr, label=modelname[model_name] + ' (area = {:.3f})'.format(model_auc))

  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')

  output_dir = os.path.join(OUTPUT_DIR, model_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  plt.savefig(os.path.join(output_dir, dataset + "_acc" + str(accuracy*100) + "_loss"+str(loss)+"_f1"+str(f1)+".jpg"))
  plt.clf()

def read_frames(video_path, videos, opt, config):
    
    # Get the video label based on dataset selected
    #method = get_method(video_path, DATA_DIR)
    if "Original" in video_path:
        label = 0.
    #elif method == "DFDC":
    elif "real" in video_path:
        label = 0.
    elif "DFDC" in video_path:
        for json_path in glob.glob(os.path.join(METADATA_PATH, "*.json")):
            with open(json_path, "r") as f:
                metadata = json.load(f)
            video_folder_name = os.path.basename(video_path)
            video_key = video_folder_name + ".mp4"
            if video_key in metadata.keys():
                item = metadata[video_key]
                label = item.get("label", None)
                if label == "FAKE":
                    label = 1.0   
                else:
                    label = 0.0
                break
            else:
                label = None
                print("NOT FOUND", video_path)
        # test_df = pd.DataFrame(pd.read_csv(TEST_LABELS_PATH))
        # video_folder_name = os.path.basename(video_path)
        # video_key = video_folder_name + ".mp4"
        # label = test_df.loc[test_df['filename'] == video_key]['label'].values[0]
    elif "synthesis" in video_path:
        label = 1.
    else:
        label = 1.
    
    # 得到视频的人脸帧图像，并且按照视频顺序排序。
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

    # Select N frames from the collected ones
    snippet = []
    transform = create_base_transform(config['model']['image-size'])
    for key in frames_paths_dict.keys():
        if len(frames_paths_dict[key]) < 200:
            continue
        frames_paths_dict[key] = frames_paths_dict[key][:200]
        for index, frame_image in enumerate(frames_paths_dict[key]):
            if index % 10 == 0:
                image = transform(image=cv2.imread(os.path.join(video_path, frame_image)))['image']
                snippet.append(image)
    if len(snippet) > 0:
        videos.append((snippet, label, video_path))

def modeleval(opt, dataset, config):
    dev = torch.device("cpu")
    # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_path = os.path.join("models/final_models", opt.model_path)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=dev)
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
    if opt.model_type == 0:
        model = S3D(num_class, config['model']['SRM-net'])
    elif opt.model_type == 1:
        model = msca_S3D(num_class, config['model']['SRM-net'])
    elif opt.model_type == 2:
        model = msca_S3D_SRM(num_class, config['model']['SRM-net'])
    elif opt.model_type == 3:
        model = CA_S3D(num_class, config['model']['SRM-net'])
        model_name = "CA_S3D"
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

    if dataset == "Celeb_DF":
        folders = ["Celeb-real", "Celeb-synthesis"]
    elif dataset != "DFDC":
        folders = ["Original", dataset]
    else:
        folders = [dataset]

    for folder in folders:
        method_folder = os.path.join(TEST_DIR, folder)  
        for _, video_folder in enumerate(os.listdir(method_folder)):
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

    bar = Bar('Predicting', max=len(videos))

    #f = open(opt.dataset + "_" + model_name + "_labels.txt", "w+")
    for index, video in enumerate(videos):
        video_faces_preds = []
        video_name = video_names[index]
        #f.write(video_name)
        
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
        #f.write(" " + str(face_pred))
        video_faces_preds.append(face_pred)
        
        bar.next()
        if len(video_faces_preds) > 1:
            video_pred = custom_video_round(video_faces_preds)
        else:
            video_pred = video_faces_preds[0]
        preds.append(video_pred)
        
        #f.write(" --> " + str(video_pred) + "(CORRECT: " + str(correct_test_labels[index]) + ")" +"\n")
        
    #f.close()
    bar.finish()

    loss_fn = torch.nn.BCEWithLogitsLoss()
    tensor_labels = torch.tensor([[float(label)] for label in correct_test_labels])
    tensor_preds = torch.tensor([[float(label)] for label in preds])

    loss = loss_fn(tensor_preds, tensor_labels).numpy()

    accuracy = accuracy_score(correct_test_labels, custom_round(np.asarray(preds)))
    f1 = f1_score(correct_test_labels, custom_round(np.asarray(preds)))
    print(model_name, " ", dataset, " Test Accuracy:", accuracy, "Loss:", loss, "F1", f1)

    save_roc_curves(dataset, correct_test_labels, preds, model_name, accuracy, loss, f1)

    for video_name in video_names:
        for i in range(21):
            tempfilename = os.path.join(tempdir, os.path.basename(video_name)+"_"+str(i)+".npy")
            if os.path.exists(tempfilename) is True:
                os.remove(tempfilename)

# Main body
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--model_path', default="S3D_final_DFDC_plan11", type=str, metavar='PATH',
                        help='Path to model checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default="DFDC",
                        help="Which dataset to use (Deepfakes|Face2Face|FaceSwap|NeuralTextures|DFDC|Celeb_DF)")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str,
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--model_type', type=int, default=0, 
                        help="Which Net to use (0,1,2, default: 0)")
    
    opt = parser.parse_args()
    print(opt)

    opt.config = "plan" + opt.model_path.split("plan")[1]
    if not opt.config:
        raise Exception("please input name of config file by '--config' .")

    # 读取配置文件。
    config_path = os.path.join("S3D/configs", opt.config+".yaml")
    with open(config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    print(config)

    tempdir = tempfile.gettempdir()

    if not opt.dataset:
        datasets = ['Deepfakes','Face2Face','FaceSwap','NeuralTextures','DFDC', 'Celeb_DF']
    else:
        datasets = [opt.dataset]
    for dataset in datasets:
        modeleval(opt, dataset, config)