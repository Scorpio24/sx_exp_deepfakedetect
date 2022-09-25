import argparse
import collections
import glob
import json
import math
import tempfile
import os
from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool
from operator import mod

import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import torch.nn.functional as F
import yaml
from progress.bar import ChargingBar
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from deepfakes_dataset import DeepFakesDataset
from model import S3D
from utils import (check_correct, get_method, get_n_params, resize,
                   shuffle_dataset)

BASE_DIR = './data'
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAINING_DIR = os.path.join(DATA_DIR, "training_set")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation_set")
TEST_DIR = os.path.join(DATA_DIR, "test_set")
MODELS_PATH = "models"
METADATA_PATH = os.path.join(BASE_DIR, "metadata") # Folder containing all training metadata for DFDC dataset
VALIDATION_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_val_labels.csv")

# 初始化分布式环境
def init_distributed_mode(args):

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

# 将多GPU上的结果取平均值
def reduce_value(value, average=True):

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value

# 读取视频帧的方法。
def read_frames(video_path, train_dataset, validation_dataset, config):
    
    # Get the video label based on dataset selected
    # 0 = true, 1 = fake
    if TRAINING_DIR in video_path:# 训练集视频。
        if "Original" in video_path: # 如果是原视频路径，则label直接为0（真）。
            label = 0.0
        elif "DFDC" in video_path: # 如果是DFDC，则根据metadata.json中的数据进行处理。
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
        else: # 如果是其他视频路径（这里指FF++中的伪造视频路径），则直接为1（假）。
            label = 1.0
        if label == None:
            print("NOT FOUND", video_path)
    else: # 验证集数据。
        if "Original" in video_path:
            label = 0.
        elif "DFDC" in video_path:
            # val_df = pd.DataFrame(pd.read_csv(VALIDATION_LABELS_PATH))
            # video_folder_name = os.path.basename(video_path)
            # video_key = video_folder_name + ".mp4"
            # label = float(val_df.loc[val_df['filename'] == video_key]['label'].values[0])
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
        else:
            label = 1.0

    # 参考项目中的代码，本实验中并没有用。
    # Calculate the interval to extract the frames
    frames_number = len(os.listdir(video_path))
    if label == 0:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing_real']),1) # Compensate unbalancing
    else:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing_fake']),1)

    
    if VALIDATION_DIR in video_path:
        min_video_frames = int(max(min_video_frames/8, 2))
    frames_interval = int(frames_number / min_video_frames)
    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}

    # 当视频中有多个人脸时，处理得到的视频帧会以"*_0.png"和"*_1.png"的形式出现。
    # 这里在参考项目中将所有人脸都包括了进来，但在本实验中目前只读取单人脸。
    # Group the faces with the same index, reduce probabiity to skip some faces in the same video
    for path in frames_paths:
        for i in range(0,1):
            if "_" + str(i) in path:
                if i not in frames_paths_dict.keys():
                    frames_paths_dict[i] = [path]
                else:
                    frames_paths_dict[i].append(path)

    # Select only the frames at a certain interval
    #if frames_interval > 0:
    #    for key in frames_paths_dict.keys():
    #        if len(frames_paths_dict) > frames_interval:
    #            frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]
    #        
    #        frames_paths_dict[key] = frames_paths_dict[key][:min_video_frames]


    # 读取一个视频中的前20帧。
    # Select N frames from the collected ones
    snippet = []
    for key in frames_paths_dict.keys():
        if len(frames_paths_dict[key]) < 20:
            continue
        frames_paths_dict[key] = frames_paths_dict[key][:20]
        for _, frame_image in enumerate(frames_paths_dict[key]):
            image=cv2.imread(os.path.join(video_path, frame_image))
            snippet.append(image)
    if len(snippet) != 0:
        if TRAINING_DIR in video_path:
            train_dataset.append((snippet, label))
        else:
            validation_dataset.append((snippet, label))


def fit(rank, world_size, opt, config, train_dataset, validation_dataset):

    # 初始化各进程环境 start
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    opt.rank = rank
    opt.world_size = world_size
    opt.gpu = rank

    opt.distributed = True

    torch.cuda.set_device(opt.gpu)
    opt.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        opt.rank, opt.dist_url), flush=True)
    dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                            world_size=opt.world_size, rank=opt.rank)
    dist.barrier()
    # 初始化各进程环境 end

    rank = opt.rank
    dev = torch.device(opt.device)
    batch_size = config['training']['bs']
    lr = config['training']['lr'] * opt.world_size  # 学习率要根据并行GPU的数量进行倍增

    if rank == 0:
        tb_writer = SummaryWriter(log_dir="runs/" + opt.config)

    # 得到一些数据集的数据，并将数据集打乱。
    train_samples = len(train_dataset)
    train_dataset = shuffle_dataset(train_dataset)
    validation_samples = len(validation_dataset)
    validation_dataset = shuffle_dataset(validation_dataset)

    # Print some useful statistics
    train_counters = collections.Counter(image[1] for image in train_dataset)
    class_weights = train_counters[0] / train_counters[1]
    val_counters = collections.Counter(image[1] for image in validation_dataset)
    if rank == 0:
        print("Train videos:", len(train_dataset), "Validation videos:", len(validation_dataset))
        print("__TRAINING STATS__")
        print(train_counters)
        print("Weights", class_weights)
        print("__VALIDATION STATS__")
        print(val_counters)
        print("___________________")

    # 设置损失函数。
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]).to(dev))

    # Create the data loaders
    validation_labels = np.asarray([row[1] for row in validation_dataset])
    labels = np.asarray([row[1] for row in train_dataset])

    # 创建对应的dataset和dataloader。
    train_dataset = DeepFakesDataset(
        np.asarray([row[0] for row in train_dataset]), 
        labels, 
        config['model']['image-size'], 
        config['training']['mask-method'], 
        config['training']['mask-number'],
        config['training']['picture-color'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    dl = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, 
                                 num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)
    del train_dataset

    validation_dataset = DeepFakesDataset(
        np.asarray([row[0] for row in validation_dataset]), 
        validation_labels, 
        config['model']['image-size'], 
        config['training']['mask-method'], 
        config['training']['mask-number'],
        config['training']['picture-color'],
        mode='validation')
    val_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, sampler=val_sampler,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)
    del validation_dataset

    # 获得模型。
    num_class = 1
    model = S3D(num_class, config['model']['SRM-net'])
    model.to(dev)
    
    # 如果之前训练在某个点中断，可以从最近的检查点恢复。
    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume, map_location=dev))
        starting_epoch = int(opt.resume.split("checkpoint")[1].split("_")[0]) + 1 # The checkpoint's file name format should be "checkpoint_EPOCH"
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            print("No checkpoint loaded.")
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=dev))

    # 在tensorboard中保存网络模型。
    if rank == 0:
        init_img = torch.zeros((1, 3, 20, 224, 224), device=dev)
        tb_writer.add_graph(model, init_img)

    # 转为DDP模型
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(dev)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config['training']['weight-decay'])
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    lf = lambda x: ((1 + math.cos(x * math.pi / opt.num_epochs)) / 2) * (1 - opt.lrf) + opt.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 开始训练。
    not_improved_loss = torch.zeros(1).to(dev)
    previous_loss = torch.tensor([math.inf]).to(dev)
    for t in range(starting_epoch, opt.num_epochs + 1):
        train_sampler.set_epoch(t)

        # 如果验证集的loss没有提升的轮次到达设置的值，就提前退出训练。
        if not_improved_loss.item() == opt.patience:
            break
        counter = torch.zeros(1).to(dev)

        total_loss = torch.zeros(1).to(dev)
        total_val_loss = torch.zeros(1).to(dev)
        
        # 设置进度条。
        if rank == 0:
            bar = ChargingBar('EPOCH #' + str(t), max=(len(dl)*config['training']['bs']+len(val_dl)))
        train_correct = torch.zeros(1).to(dev)
        positive = torch.zeros(1).to(dev)
        negative = torch.zeros(1).to(dev)
        # 训练集训练。
        model.train()
        for index, (images, labels) in enumerate(dl):
            labels = labels.unsqueeze(1).to(dev)
            images = images.to(dev)
            
            y_pred = model(images)
            
            loss = loss_func(y_pred, labels)
            loss.backward()
            loss = reduce_value(loss, average=True)

            # 计算正确率和统计训练集数据中的预测（preds）数据。
            # positive_class为预测为1的数量，negative_class为预测为0的数量。
            corrects, positive_class, negative_class = check_correct(y_pred, labels)  
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            
            optimizer.step()
            optimizer.zero_grad()

            counter += 1
            total_loss += round(loss.item(), 2)
            
            #if index % 1200 == 0 and rank == 0: # Intermediate metrics print
            #    print("\nLoss: ", total_loss/counter, "Accuracy: ",train_correct/(counter*batch_size) ,"Train 0s: ", negative, "Train 1s:", positive)

            # 更新进度条。
            if rank == 0:
                for i in range(config['training']['bs']):
                    bar.next()

        # 等待所有进程计算完毕
        if dev != torch.device("cpu"):
            torch.cuda.synchronize(dev)

        dist.all_reduce(train_correct)
        train_correct /= train_samples
        dist.all_reduce(total_loss)
        dist.all_reduce(counter)
        total_loss /= counter

        # 更新优化器的调度器。    
        scheduler.step()

        # 设置验证集统计数据。
        val_correct = torch.zeros(1).to(dev)
        val_positive = torch.zeros(1).to(dev)
        val_negative = torch.zeros(1).to(dev)
        val_counter = torch.zeros(1).to(dev)
        
        # 验证集预测。
        model.eval()
        with torch.no_grad():
            for index, (val_images, val_labels) in enumerate(val_dl):
                val_images = val_images.to(dev)
                val_labels = val_labels.unsqueeze(1).to(dev)
                val_pred = model(val_images)

                # 统计验证集对应数据。
                val_loss = loss_func(val_pred, val_labels)
                val_loss = reduce_value(val_loss, average=True)
                total_val_loss += round(val_loss.item(), 2)
                corrects, positive_class, negative_class = check_correct(val_pred, val_labels)
                val_correct += corrects
                val_positive += positive_class
                val_counter += 1
                val_negative += negative_class
                if rank == 0:
                    bar.next()

        # 等待所有进程计算完毕
        if dev != torch.device("cpu"):
            torch.cuda.synchronize(dev)

        # 完结进度条
        if rank == 0:
            bar.finish()

        # 计算本次epoch中的验证集损失和正确率。
        # 如果验证集的loss没有提升则对应计数变量+1.   
        dist.all_reduce(total_val_loss)
        dist.all_reduce(val_counter) 
        total_val_loss /= val_counter
        dist.all_reduce(val_correct)
        val_correct /= validation_samples
        if previous_loss <= total_val_loss:
            if rank == 0:
                print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            not_improved_loss.zero_()
        
        # 打印本epoch的训练数据信息。
        previous_loss = total_val_loss
        dist.all_reduce(val_negative)
        dist.all_reduce(val_positive)
        if rank == 0:
            print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" +str(total_loss.item()) + 
                " accuracy:" + str(train_correct.item()) +" val_loss:" + str(total_val_loss.item()) + 
                " val_accuracy:" + str(val_correct.item()) + 
                " val_0s:" + str(val_negative.item()) + "/" + str(np.count_nonzero(validation_labels == 0)) + 
                " val_1s:" + str(val_positive.item()) + "/" + str(np.count_nonzero(validation_labels == 1)))
        
            # 在tensorboard中保存本epoch的训练信息
            tb_writer.add_scalar("train loss", total_loss.item(), t)
            tb_writer.add_scalar("train acc", train_correct.item(), t)
            tb_writer.add_scalar("val loss", total_val_loss.item(), t)
            tb_writer.add_scalar("val acc", val_correct.item(), t)
            tb_writer.add_scalar("learning rate", optimizer.param_groups[0]["lr"], t)

            # 把每10个epoch当做一个检查点，保存对应的模型数据，以便后面继续训练。
            if not os.path.exists(MODELS_PATH):
                os.makedirs(MODELS_PATH)
            if t % 10 == 0:
                torch.save(model.state_dict(), 
                    os.path.join(MODELS_PATH,  
                    "S3D_checkpoint" + str(t) + "_" + opt.dataset + "_" + opt.config))
        #exit()
    
    # 删除临时缓存文件
    if rank == 0:
        tb_writer.close()
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
        
        # 保存最终模型。
        torch.save(model.state_dict(), 
            os.path.join(MODELS_PATH, "final_models",  "S3D_final_" + opt.dataset + "_" + opt.config))

    # 结束分布式环境。
    dist.destroy_process_group()


if __name__ == '__main__':

    # 读取命令行的参数。
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=300, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=1, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='DFDC', 
                        help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|All)")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str,
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--patience', type=int, default=5, 
                        help="How many epochs wait before stopping for validation loss not improving.")
    parser.add_argument('--lrf', type=float, default=0.1)
    # 以下是多GPU的参数
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),在单机中指使用GPU的数量
    parser.add_argument('--world-size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    
    opt = parser.parse_args()
    print(opt)

    if not opt.config:
        raise Exception("please input name of config file by '--config' .")

    # 读取配置文件。
    config_path = os.path.join("S3D/configs", opt.config+".yaml")
    with open(config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    print(config)

    #READ DATASET
    folders = ["DFDC"]
    sets = [TRAINING_DIR, VALIDATION_DIR]

    # 遍历目录和数据集，得到视频的路径，精确到视频目录，具体格式类似为“data\dataset\training_set\DFDC\aagfhgtpmv”。
    paths = []
    for dataset in sets:
        for folder in folders:
            subfolder = os.path.join(dataset, folder)
            for _, video_folder_name in enumerate(os.listdir(subfolder)):
                if os.path.isdir(os.path.join(subfolder, video_folder_name)):
                    paths.append(os.path.join(subfolder, video_folder_name))

    #for path in paths:
    #    read_frames(path, [], [], config)
    
    # 多进程变量的设置。
    mgr = Manager()
    train_dataset = mgr.list()
    validation_dataset = mgr.list()

    # 使用多进程的方式读取视频帧。
    #with Pool(processes=10) as p:
    p = Pool(processes=10)
    with tqdm(total=len(paths)) as pbar:
        for v in p.imap_unordered(partial(read_frames, 
            train_dataset=train_dataset, 
            validation_dataset=validation_dataset,
            config=config),
            paths):
            pbar.update()
    p.close()
    p.join()

    world_size = opt.world_size
    processes = []
    for rank in range(world_size):
        p = Process(target=fit, args=(rank, world_size, opt, config, train_dataset, validation_dataset))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
