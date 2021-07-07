'''
@File    :   train.py
@Time    :   2021/06/07 16:06:36
@Author  :   Tan Wenhao
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
'''

# 库函数
import argparse
import torch
import os
import socket
from torch.utils.data import DataLoader
from datetime import datetime
from tensorboardX import SummaryWriter
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from PIL import Image
import torch.backends.cudnn as cudnn
# 内部函数
import utils.progress_bar as pb
import utils.utils as u
import utils.loss as LS
from config import cnv_single_config
from dataset.CNV import CNV
from dataset.CNV_2d5 import CNV_2d5
from models.net_builder import net_builder


def val(args, model, dataloader):
    with torch.no_grad():
        model.eval()
        val_progressor = pb.Val_ProgressBar(save_model_path=args.save_model_path,total=len(dataloader)) # 验证进度条，用于显示指标

        total_Dice = []
        total_Acc = []
        total_jaccard = []
        total_Sensitivity = []
        total_Specificity = []

        for i, (data, label) in enumerate(dataloader):
            val_progressor.current = i
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # get RGB predict image
            predict = model(data)
            Dice, Acc, jaccard, Sensitivity, Specificity = u.eval_single_seg(predict, label) # 每个指标返回的是batchsize长的列表
            # 将每个batch的列表相加，在迭代中动态显示指标的平均值（最终值）
            total_Dice += Dice
            total_Acc += Acc
            total_jaccard += jaccard
            total_Sensitivity += Sensitivity
            total_Specificity += Specificity
            # 表示对batchsize个值取平均，len=batchsize
            dice = sum(total_Dice) / len(total_Dice)
            acc = sum(total_Acc) / len(total_Acc)
            jac = sum(total_jaccard) / len(total_jaccard)
            sen = sum(total_Sensitivity) / len(total_Sensitivity)
            spe = sum(total_Specificity) / len(total_Specificity)
            val_progressor.val=[dice,acc,jac,sen,spe]
            val_progressor()    
        val_progressor.done()
            
        return dice, acc, jac, sen, spe


def train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer):
    best_pred = 0.0
    best_epoch = 0
    end_epoch = None # 可以设为1，用于直接进入test过程，检查bug
    step = 0         # tensorboard相关
    end_index = None # 可以设为1，用于直接进入val过程，检查bug
    train_start_time = datetime.now().strftime('%b%d_%H-%M-%S')
    with open("./logs/%s.txt" % args.save_model_path, "a") as f:
        print(train_start_time, file=f)
    for epoch in range(args.num_epochs):
        if(epoch==end_epoch):
            break
        train_loss = u.AverageMeter() # 滑动平均
        train_progressor = pb.Train_ProgressBar(mode='train', epoch=epoch, total_epoch=args.num_epochs, 
            save_model_path=args.save_model_path, total=len(dataloader_train)*args.batch_size) # train进度条，用于显示loss和lr
        lr = u.adjust_learning_rate(args, optimizer, epoch) # 自动调节学习率
        model.train()

        for i, (data, label) in enumerate(dataloader_train):
            if i==end_index:
                break
            train_progressor.current = i*args.batch_size

            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            output = model(data)

            output = torch.sigmoid(output) # 将输出结果映射到0:1,满足BCELoss要求

            loss_aux = criterion[0](output, label) # criterion[0]=BCELoss
            loss_main = criterion[1](output, label) # criterion[1]=DiceLoss
            loss = loss_main + loss_aux
            train_loss.update(loss.item(), data.size(0)) # loss.item()表示去除张量的元素值，data.size()表示batchsize
            train_progressor.current_loss = train_loss.avg
            train_progressor.current_lr = lr
            optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播
            optimizer.step() # 梯度更新
            train_progressor() # 显示进度条
            step += 1
            if step % 10 == 0:
                writer.add_scalar('Train/loss_step', loss, step)
        train_progressor.done() # 输出logs
        writer.add_scalar('Train/loss_epoch', float(train_loss.avg), epoch)
        Dice, Acc, jaccard, Sensitivity, Specificity = val(args, model, dataloader_val)
        writer.add_scalar('Valid/Dice_val', Dice, epoch)
        writer.add_scalar('Valid/Acc_val', Acc, epoch)
        writer.add_scalar('Valid/Jac_val', jaccard, epoch)
        writer.add_scalar('Valid/Sen_val', Sensitivity, epoch)
        writer.add_scalar('Valid/Spe_val', Specificity, epoch)

        is_best = Dice > best_pred
        if is_best:
            best_pred = Dice
            best_jac = jaccard
            best_acc = Acc
            best_sen = Sensitivity
            best_spe = Specificity
            best_epoch = epoch + 1
        checkpoint_dir = os.path.join('./checkpoints',args.save_model_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_latest_name = os.path.join(checkpoint_dir, 'checkpoint_latest.path.tar') # 保存最新的一个checkpoint用于中继训练
        u.save_checkpoint({
            'epoch': best_epoch,
            'state_dict': model.state_dict(),
            'best_dice': best_pred
        }, best_pred, epoch, is_best, checkpoint_dir, filename=checkpoint_latest_name) # 保存最好的
    # 记录该折分割效果最好一次epoch的所有参数
    best_indicator_message = "best pred in Epoch:{}\nVal Result:\nMetric: Dice Precision jaccard Sensitivity Specificity\n{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
        best_epoch, best_pred, best_acc, best_jac, best_sen, best_spe)
    test_start_time = datetime.now().strftime('%b%d %H:%M:%S')
    with open("./logs/%s_test_indicator.txt" % args.save_model_path, mode='a') as f:
        print("Test time: "+test_start_time, file=f)
        print(best_indicator_message, file=f)


def eval(args, model, dataloader):
    print('\nStart Test!')
    model_path = list()
    # num_checkpoint = len(os.listdir(os.path.join("./checkpoints",args.net_work)))
    for c in os.listdir(os.path.join('./checkpoints',args.save_model_path)):
        model_path.append(os.path.join('./checkpoints',args.save_model_path,c))
    assert len(model_path)>0,print("No checkpoint detected! Please train first or change the 'save_model_path'!")
    pretrained_model_path = model_path[0] # 最后一个模型(最好的)
    print("Load best model "+'\"'+os.path.abspath(pretrained_model_path)+'\"')
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    with torch.no_grad():
        model.eval()
        test_progressor = pb.Test_ProgressBar(total=len(dataloader),save_model_path=args.save_model_path)

        total_Dice = []
        total_Acc = []
        total_jaccard = []
        total_Sensitivity = []
        total_Specificity = []

        for i, (data, [label,label_path]) in enumerate(dataloader):
            test_progressor.current = i
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # get RGB predict image
            predict = model(data)
            Dice, Acc, jaccard, Sensitivity, Specificity = u.eval_single_seg(predict, label)
            predict = torch.round(torch.sigmoid(predict)).byte()
            predict_seg = predict.data.cpu().numpy().squeeze()*255
            img = Image.fromarray(predict_seg,mode='L')
            label_name = label_path[0].split(os.sep)[-1].split('.')[0]+'.png'
            img.save(args.result_path+label_name)
            total_Dice += Dice
            total_Acc += Acc
            total_jaccard += jaccard
            total_Sensitivity += Sensitivity
            total_Specificity += Specificity
        
            dice = sum(total_Dice) / len(total_Dice)
            acc = sum(total_Acc) / len(total_Acc)
            jac = sum(total_jaccard) / len(total_jaccard)
            sen = sum(total_Sensitivity) / len(total_Sensitivity)
            spe = sum(total_Specificity) / len(total_Specificity)
            test_progressor.val=[dice,acc,jac,sen,spe]
            test_progressor()    
        test_progressor.done()


def main(mode='train', args=None, writer=None):
    # create dataset and dataloader
    dataset_path = os.path.join(args.data, args.dataset)
    assert args.img == '2d' or args.img == '2d5'
    if args.img == '2d':
        dataset_train = CNV(dataset_path, scale=(args.crop_height, args.crop_width), mode='train')
        dataloader_train = DataLoader(
            dataset_train, 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=True, 
            drop_last=False)
        dataset_val = CNV(dataset_path, scale=(args.crop_height, args.crop_width), mode='val')
        dataloader_val = DataLoader(
            dataset_val, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False)
        dataset_test = CNV(dataset_path, scale=(args.crop_height, args.crop_width), mode='test')
        dataloader_test = DataLoader(
            dataset_test, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False)
        in_channels = 1

    elif args.img == '2d5':
        dataset_train = CNV_2d5(dataset_path, scale=(args.crop_height, args.crop_width), mode='train')
        dataloader_train = DataLoader(
            dataset_train, 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=True, 
            drop_last=True)
        dataset_val = CNV_2d5(dataset_path, scale=(args.crop_height, args.crop_width), mode='val')
        dataloader_val = DataLoader(
            dataset_val, 
            batch_size=1, 
            shuffle=True,
            num_workers=args.num_workers, 
            pin_memory=True, 
            drop_last=False)
        dataset_test = CNV_2d5(dataset_path, scale=(args.crop_height, args.crop_width), mode='test')
        dataloader_test = DataLoader(
            dataset_test, 
            batch_size=1, 
            shuffle=False,
            num_workers=args.num_workers, 
            pin_memory=True, 
            drop_last=False)
        in_channels = 3
    
    # 模型
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = net_builder(name=args.net_work, pretrained=True, in_channels=in_channels, n_class=args.num_classes).cuda()
    cudnn.benchmark = True
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # 损失函数
    criterion_aux = nn.BCELoss()
    criterion_main = LS.DiceLoss()
    criterion = [criterion_aux, criterion_main]

    if mode == 'train':  # 训练
        train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer)
    if mode == 'test':  # 测试
        eval(args,model, dataloader_test)
    if mode == 'train_test': # 训练的结果用于测试
        train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer)
        eval(args, model, dataloader_test)


if __name__ == "__main__":
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = cnv_single_config()
    modes = args.mode

    if modes == 'train':
        comments = os.getcwd().split(os.sep)[-1]
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(args.log_dirs, args.net_work + '_' + current_time + '_' + socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir)
        main(mode='train', args=args, writer=writer)

    elif modes == 'train_test':
        comments = os.getcwd().split(os.sep)[-1]
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(args.log_dirs, args.net_work + '_' + current_time + '_' + socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir)
        main(mode='train_test', args=args, writer=writer)

    elif modes == 'test':
        main(mode='test', args=args, writer=None)
