'''
@File    :   train.py
@Time    :   2021/08/12 10:23:52
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
# from tensorboardX import SummaryWriter
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from PIL import Image
import torch.backends.cudnn as cudnn
# 内部函数
from dataset import CNV
import utils.progress_bar as pb
import utils.utils as u
import utils.loss as LS
from config import cnv_single_config
from models.net_builder import net_builder


def val(args, model, dataloader, fold):
    result_path = args.result_path
    if not os.path.exists(result_path+os.sep+'f1'):
        os.makedirs(result_path+os.sep+'f1')
    if not os.path.exists(result_path+os.sep+'f2'):
        os.makedirs(result_path+os.sep+'f2')
    if not os.path.exists(result_path+os.sep+'f3'):
        os.makedirs(result_path+os.sep+'f3')
    if not os.path.exists(result_path+os.sep+'f4'):
        os.makedirs(result_path+os.sep+'f4')

    with torch.no_grad():
        model.eval()
        val_progressor = pb.Val_ProgressBar(save_model_path=args.save_model_path,total=len(dataloader)) # 验证进度条，用于显示指标

        total_Dice = []
        total_jaccard = []
        total_Sensitivity = []
        total_Specificity = []

        for i, (data, (label,label_path)) in enumerate(dataloader):
            val_progressor.current = i
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # get RGB predict image
            predict = model(data)

            # save segment result
            predict_seg = torch.round(predict).byte()
            predict_seg = predict_seg.data.cpu().numpy().squeeze()*255
            img = Image.fromarray(predict_seg,mode='L')
            label_name = label_path[0].split(os.sep)[-1]
            img.save(os.path.join(args.result_path,'f'+str(fold),label_name))

            # verify the model
            Dice, jaccard, Sensitivity, Specificity = u.eval_single_seg(predict, label) # 每个指标返回的是batchsize长的列表
            # 将每个batch的列表相加，在迭代中动态显示指标的平均值
            total_Dice += Dice
            total_jaccard += jaccard
            total_Sensitivity += Sensitivity
            total_Specificity += Specificity
            # 表示对batchsize个值取平均，len=batchsize
            dice = sum(total_Dice) / len(total_Dice)
            jac = sum(total_jaccard) / len(total_jaccard)
            sen = sum(total_Sensitivity) / len(total_Sensitivity)
            spe = sum(total_Specificity) / len(total_Specificity)
            val_progressor.val=[dice,jac,sen,spe]
            val_progressor()    
        val_progressor.done()
        return dice, jac, sen, spe


def train(args, model, optimizer, criterion, train_dataloader, val_dataloader,  k_fold):
    best_pred = 0.0
    best_epoch = 0
    end_epoch = None # 可以设为1，用于直接进入test过程，检查bug
    step = 0         # tensorboard相关
    end_index = None # 可以设为1，用于直接进入val过程，检查bug
    train_start_time = datetime.now().strftime('%b%d_%H-%M-%S')
    with open("%s.txt" % args.save_model_path, "a") as f:
        print(train_start_time, file=f)
    for epoch in range(args.num_epochs):
        if(epoch==end_epoch):
            break
        loss_record = []
        train_loss = u.AverageMeter() # 滑动平均
        train_progressor = pb.Train_ProgressBar(mode='train', epoch=epoch, total_epoch=args.num_epochs, fold=k_fold,
            save_model_path=args.save_model_path, total=len(train_dataloader)*args.batch_size) # train进度条，用于显示loss和lr
        lr = u.adjust_learning_rate(args, optimizer, epoch) # 自动调节学习率
        model.train()

        for i, (data, label) in enumerate(train_dataloader):
            if i==end_index:
                break
            train_progressor.current = i*args.batch_size

            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            output = model(data)

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
            # if step % 10 == 0:
                # writer.add_scalar('Train/loss_step_{}'.format(int(k_fold)), loss, step)
            loss_record.append(loss.item())

        train_progressor.done() # 输出logs
        # writer.add_scalar('Train/loss_epoch_{}'.format(int(k_fold)), float(train_loss.avg), epoch)
        Dice, jaccard, Sensitivity, Specificity = val(args, model, val_dataloader,k_fold)
        # writer.add_scalar('Valid/Dice_val_{}'.format(int(k_fold)), Dice, epoch)
        # writer.add_scalar('Valid/Jac_val_{}'.format(int(k_fold)), jaccard, epoch)
        # writer.add_scalar('Valid/Sen_val_{}'.format(int(k_fold)), Sensitivity, epoch)
        # writer.add_scalar('Valid/Spe_val_{}'.format(int(k_fold)), Specificity, epoch)

        is_best = Dice > best_pred
        if is_best:
            best_pred = Dice
            best_jac = jaccard
            best_sen = Sensitivity
            best_spe = Specificity
            best_epoch = epoch + 1
        checkpoint_dir = os.path.join('./checkpoints',args.net_work,str(k_fold))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_latest_name = os.path.join(checkpoint_dir,'checkpoint_latest.pth.tar') # 保存最新的一个checkpoint用于中继训练
        u.save_checkpoint({
            'epoch': best_epoch,
            'state_dict': model.state_dict(),
            'best_dice': best_pred
        }, best_pred, epoch, is_best, checkpoint_dir, filename=checkpoint_latest_name) # 保存最好的
    # 记录该折分割效果最好一次epoch的所有参数
    best_indicator_message = "best pred in Epoch:{}\nMetric: Dice jaccard Sensitivity Specificity\n{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
        best_epoch, best_pred, best_jac, best_sen, best_spe)
    test_start_time = datetime.now().strftime('%b%d %H:%M:%S')
    with open("%s_test_indicator.txt" % args.save_model_path, mode='a') as f:
        print("fold {}".format(k_fold), file=f)
        print(best_indicator_message, file=f)
    para = best_pred, best_jac, best_sen, best_spe
    return para


def main(args=None, k_fold=1):
    # create dataset and dataloader
    dataset_path = os.path.join(args.data, args.dataset)
    dataset_train = CNV(dataset_path, scale=(args.crop_height, args.crop_width), k_fold_test=k_fold, mode='train')
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=False)
    dataset_val = CNV(dataset_path, scale=(args.crop_height, args.crop_width), k_fold_test=k_fold, mode='val')
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=1, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False)
    
    # 模型
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = net_builder(name=args.net_work, in_channels=3, n_class=args.num_classes).cuda()
    para_nums = sum(p.numel() for p in list(model.parameters()) if p.requires_grad)
    print('parameters_number = {}'.format(para_nums))
    with open("%s_test_indicator.txt" % args.save_model_path, mode='a') as f:
        print("parameters_number = {}".format(para_nums), file=f)
    cudnn.benchmark = True
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # 损失函数
    criterion_aux = nn.BCELoss()
    criterion_main = LS.DiceLoss()
    criterion = [criterion_aux, criterion_main]

    para = train(args, model, optimizer, criterion, dataloader_train, dataloader_val, k_fold)

    return para


if __name__ == "__main__":
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = cnv_single_config()

    dice_4_fold = list()
    jac_4_fold = list()
    sen_4_fold = list()
    spe_4_fold = list()
    comments = os.getcwd().split(os.sep)[-1]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(args.log_dirs, args.net_work + '_' + current_time + '_' + socket.gethostname())
    # writer = SummaryWriter(log_dir=log_dir)
    
    for i in range(0,args.k_fold):
        para = main(args=args, k_fold=int(i + 1))
        dice_4_fold.append(para[0])
        jac_4_fold.append(para[1])
        sen_4_fold.append(para[2])
        spe_4_fold.append(para[3])
    print("Train Finished!\n\nAverage Metric: \nDice={:.4f}±{:.4f}\njaccard={:.4f}±{:.4f}\nSensitivity={:.4f}±{:.4f}\nSpecificity={:.4f}±{:.4f}".format(
            np.mean(dice_4_fold),np.std(dice_4_fold),np.mean(jac_4_fold),np.std(jac_4_fold),np.mean(sen_4_fold),np.std(sen_4_fold),np.mean(spe_4_fold),np.std(spe_4_fold)))
    with open("%s_test_indicator.txt" % args.save_model_path, mode='a') as f:
        print("\n\nAverage Metric: \nDice jaccard Sensitivity Specificity\n{:.4f}±{:.4f}\t{:.4f}±{:.4f}\t{:.4f}±{:.4f}\t{:.4f}±{:.4f}".format(
            np.mean(dice_4_fold),np.std(dice_4_fold),np.mean(jac_4_fold),np.std(jac_4_fold),np.mean(sen_4_fold),np.std(sen_4_fold),np.mean(spe_4_fold),np.std(spe_4_fold)), file=f)

