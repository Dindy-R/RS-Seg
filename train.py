'''
@Description: 训练
'''
import os
import time
import datetime

import torch
import random
import shutil
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from torchsummary import summary
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
from utils import load_config, save_config
from utils.utils import AverageMeter, init_logger, ExpSmoothing, exp_smoothing
from utils.metric import Metric, PixelMetric
from utils.losses import JointLoss, SoftBCELoss, TLoss
from utils.optimzer import build_optimizer
from utils.lr_scheduler import PolyScheduler
from datasets.ImgMaskDataset import DualDataset, OwnDataset, get_train_transform, get_val_transform

from models import create_model
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss, SoftBCEWithLogitsLoss
from utils.early_stop import EarlyStopping


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfgD = cfg['dataset']
        self.cfgT = cfg['train']
        self.cfgN = cfg['network']
        self.cfgOptim = cfg['optimizer']
        self.log_dir = osp.join(cfg['run_dir'], cfg['run_name'])
        self.ckpt_dir = osp.join(self.log_dir, 'ckpt')
        self.nclass = len(self.cfgD['cls_info'])
        self.binary = (self.nclass == 2)
        self.early_stopping = EarlyStopping(patience=self.cfgT['patience'], verbose=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # datasets

        train_transforme = get_train_transform()
        val_transforme = get_val_transform()

        with open(os.path.join(self.cfgD['data_dirs'], "ImageSets/train.txt"), "r") as f:
            train_lines = f.readlines()
        with open(os.path.join(self.cfgD['data_dirs'], "ImageSets/val.txt"), "r") as f:
            val_lines = f.readlines()
        with open(os.path.join(self.cfgD['data_dirs'], "ImageSets/test.txt"), "r") as f:
            test_lines = f.readlines()

        train_data = OwnDataset(train_lines, True, self.cfgD['data_dirs'], transform=train_transforme)
        val_data = OwnDataset(val_lines, False, self.cfgD['data_dirs'], transform=val_transforme)
        test_data = OwnDataset(test_lines, False, self.cfgD['data_dirs'], transform=None)

        # train_data = DualDataset(data_dirs=self.cfgD['train_dirs'], transform=train_transforme)
        # val_data = DualDataset(data_dirs=self.cfgD['val_dirs'], transform=val_transforme)

        img_c = self.cfgN['in_channel']
        img_h = self.cfgN['in_size']
        img_w = self.cfgN['in_size']

        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.cfgT['batch_size'],
            shuffle=True,
            num_workers=4,
            drop_last=True)
        self.val_loader = DataLoader(
            dataset=val_data,
            batch_size=self.cfgT['batch_size'],
            shuffle=False,
            num_workers=4,
            drop_last=False)
        # network
        model = create_model(cfg=self.cfgN).cuda()
        # print(model)
        summary(model, input_size=(img_c, img_h, img_w), batch_size=self.cfgT['batch_size'])
        self.model = torch.nn.DataParallel(model)
        self.max_epochs = self.cfgT['epochs']
        self.start_epoch = 0

        self.optimizer = optim.AdamW([{"params": [param for name, param in model.named_parameters()
                                                  if "encoder" in name], "lr": self.cfgOptim['lr']},
                                      {"params": [param for name, param in model.named_parameters()
                                                  if "encoder" not in name], "lr": self.cfgOptim['lr'] * 10.0}],
                                     lr=self.cfgOptim['lr'], weight_decay=self.cfgOptim['weight_decay'])
        self.scheduler = PolyScheduler(self.optimizer, power=1, total_steps=self.max_epochs, min_lr=1e-6, last_epoch=-1)
        #定义损失函数
        fl_fn = FocalLoss(mode='multiclass', alpha=0.25, ignore_index=self.cfgD['ignore_index'])
        # t_loss = TLoss(img_size=self.cfgN['in_size'], nu=1.0, epsilon=1e-8, reduction='sum')
        dice_fn = DiceLoss(mode='multiclass', ignore_index=self.cfgD['ignore_index'])
        ce_fn = SoftCrossEntropyLoss(smooth_factor=self.cfgT['smoothing'], ignore_index=self.cfgD['ignore_index'])

        self.criterion = JointLoss(first=dice_fn, second=ce_fn, first_weight=0.5, second_weight=0.5).cuda()
        # self.criterion = t_loss.cuda()

        self.train_loss_epochs, self.val_loss_epochs, self.epoch_lr = [], [], []
        self.previous_best = 10000.0
        # logger
        self.writer = SummaryWriter(self.log_dir)
        self.logger = init_logger(osp.join(self.log_dir, time.strftime("%m-%d-%H-%M-%S", time.localtime()) + '.log'))
        self.logger.info('Total Epoch:{} Image_size:({}, {}) Training num:{}  Validation num:{}'.format(
            self.max_epochs, img_w, img_h, len(train_data), len(val_data)))

    def training(self, ei):
        self.model.train()
        loss_smoothing = ExpSmoothing()
        epoch_loss = AverageMeter()
        tbar = tqdm(self.train_loader)
        tbar.set_description('Train')
        start_time = time.time()
        for batch_idx, samples in enumerate(tbar):
            self.optimizer.zero_grad()
            imgs, gts = samples['image'], samples['label']
            preds = self.model(imgs.cuda())
            loss = self.criterion(preds, gts.cuda())
            loss.backward()
            self.optimizer.step()
            _loss = loss_smoothing(loss.item())
            epoch_loss.update(loss.item())
            tbar.set_postfix(dict(loss='{:.6f}'.format(_loss)))
            self.writer.add_scalar('train loss', loss.item(), ei*len(self.train_loader) + batch_idx)
        end_time = time.time()
        epoch_time = end_time - start_time
        self.train_loss_epochs.append(epoch_loss.avg)
        self.epoch_lr.append(self.scheduler.get_last_lr()[-1])
        self.scheduler.step()
        if self.early_stopping.early_stop:
            print("Early Stopping")
            return
        print(f"Training time for epoch {ei}: {epoch_time} second")

    @torch.no_grad()
    def validation(self, ei):
        self.model.eval()
        loss_smoothing = ExpSmoothing()
        epoch_loss = AverageMeter()
        metric = Metric(num_class=self.nclass, binary=self.binary)
        tbar = tqdm(self.val_loader)
        tbar.set_description('Val')
        for batch_idx, samples in enumerate(tbar):
            imgs, gts = samples['image'], samples['label']
            preds = self.model(imgs.cuda())
            loss = self.criterion(preds, gts.cuda())
            preds = torch.softmax(preds, dim=1).cpu().numpy()
            preds = np.argmax(preds, axis=1)
            # preds = (preds.squeeze(1).cpu().numpy() > 0.5)
            metric.add_batch(preds, gts.numpy())
            _loss = loss_smoothing(loss.item())
            epoch_loss.update(loss.item())
            tbar.set_postfix(dict(loss='{:.6f}'.format(_loss)))
            self.writer.add_scalar('val loss', loss.item(), ei * len(self.train_loader) + batch_idx)

        scores = metric.evaluate()
        self.logger.info('[Train] Loss:{:.6f}\t[Val] Loss:{:.6f} mIoU:{:.3f}'.format(
            self.train_loss_epochs[-1], epoch_loss.avg, scores['mean_iou']))
        self.val_loss_epochs.append(epoch_loss.avg)
        self.early_stopping(epoch_loss.avg, self.model)

        # 保存模型
        state = {'epoch': ei, 'state_dict': self.model.module.state_dict(), 'optimizer': self.optimizer.state_dict()}
        if self.cfgT['save_inter'] > 0 and ei % self.cfgT['save_inter'] == 0:
            save_path = osp.join(self.ckpt_dir, f'epoch{ei}.pth')
            torch.save(state, save_path)
        # 保存最优模型
        if epoch_loss.avg < self.previous_best:
            save_path = osp.join(self.ckpt_dir, 'best.pth')
            torch.save(state, save_path)
            self.previous_best = epoch_loss.avg
            self.logger.info('Best Model saved at epoch:{}'.format(ei))
        # self.logger.info('')    # 换行

    def run(self):
        for ei in range(self.start_epoch, self.max_epochs):
            self.training(ei)
            self.validation(ei)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

    @torch.no_grad()
    def finishing(self, ei):
        self.model.eval()
        # 保存最后模型
        state = {'epoch': ei, 'state_dict': self.model.module.state_dict(), 'optimizer': self.optimizer.state_dict()}
        save_path = osp.join(self.ckpt_dir, 'last.pth')
        torch.save(state, save_path)
        self.logger.info('Last Model saved at epoch:{}'.format(ei))

        if self.test_loader:
            metric = PixelMetric(self.nclass, logger=self.logger)
            tbar = tqdm(self.test_loader)
            tbar.set_description('Evaluate on Test set')
            for i, samples in enumerate(tbar):
                imgs, gts = samples['image'], samples['label']
                preds = self.model(imgs.cuda())
                preds = torch.softmax(preds, dim=1).cpu().numpy()
                preds = np.argmax(preds, axis=1)
                # pred = (pred.squeeze(1).cpu().numpy() > 0.5)
                y_true = gts.numpy().ravel()
                y_pred = preds.ravel()
                metric.forward(y_true, y_pred)
            metric.summary_all()

        # 训练loss曲线
        if self.cfgT['plot']:
            x = [i for i in range(self.max_epochs)]
            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(1, 2, 1)
            ax.plot(x, exp_smoothing(self.train_loss_epochs, 0.6), label='train loss')
            ax.plot(x, exp_smoothing(self.val_loss_epochs, 0.6), label='val loss')
            ax.set_xlabel('Epoch', fontsize=15)
            ax.set_ylabel('Loss', fontsize=15)
            ax.set_title('train curve', fontsize=15)
            ax.grid(True)
            plt.legend(loc='upper right', fontsize=15)
            ax = fig.add_subplot(1, 2, 2)
            ax.plot(x, self.epoch_lr, label='Learning Rate')
            ax.set_xlabel('Epoch', fontsize=15)
            ax.set_ylabel('Learning Rate', fontsize=15)
            ax.set_title('lr curve', fontsize=15)
            ax.grid(True)
            plt.legend(loc='upper right', fontsize=15)
            plt.savefig(osp.join(self.log_dir, 'plot.png'))
            plt.show()


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-c', '--config', default='deeplab', type=str, help='Name of the config file.') ##修改yml名称
    argparser.add_argument('-g', '--gpus', default='0', type=str, help='gpus')
    return argparser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    cfg = load_config(args.config, "configs")
    cfg['run_name'] = args.config
    for item in cfg.items():
        print(item)
    # setup log dir
    log_dir = osp.join(cfg['run_dir'], cfg['run_name'])
    if "train" in cfg['mode']:
        if os.path.exists(log_dir):
            while True:
                _input = input(f"> Folder [{log_dir}] already exists, overwrite? (Y/N):")
                if _input in ['y', 'Y']:
                    shutil.rmtree(log_dir)
                    break
                elif _input in ['n', 'N']:
                    exit(0)
        os.makedirs(log_dir)
        # save_config(cfg, log_dir)

    torch.manual_seed(0)
    trainer = Trainer(cfg)
    # main loop
    for e in range(trainer.max_epochs):
        trainer.logger.info("Epoche {}/{}\t lr = {:.6f}\t previous best = {:.6f}".format(
            e+1, trainer.max_epochs, trainer.optimizer.param_groups[0]["lr"], trainer.previous_best))
        trainer.training(e)
        trainer.validation(e)
    trainer.finishing(trainer.max_epochs-1)

