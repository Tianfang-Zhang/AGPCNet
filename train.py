import os
import os.path as osp
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models import get_segmentation_model
from utils.data import *
from utils.loss import SoftLoULoss
from utils.lr_scheduler import *
from utils.metrics import SegmentationMetricTPFNFP


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of AGPCNet')

    #
    # Dataset parameters
    #
    parser.add_argument('--base-size', type=int, default=256, help='base size of images')
    parser.add_argument('--dataset', type=str, default='sirstaug', help='choose datasets')

    #
    # Training parameters
    #
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--warm-up-epochs', type=int, default=0, help='warm up epochs')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--ngpu', type=int, default=0, help='GPU number')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--lr-scheduler', type=str, default='poly', help='learning rate scheduler')
    parser.add_argument('--save-iter-step', type=int, default=10, help='save model per step iters')

    #
    # Net parameters
    #
    parser.add_argument('--net-name', type=str, default='agpcnet',
                        help='net name: fcn')

    args = parser.parse_args()
    return args


def set_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, args):
        self.args = args

        ## dataset
        if args.dataset == 'sirstaug':
            trainset = SirstAugDataset(mode='train')
            valset = SirstAugDataset(mode='test')
        elif args.dataset == 'mdfa':
            trainset = MDFADataset(mode='train', base_size=args.base_size)
            valset = MDFADataset(mode='test', base_size=args.base_size)
        elif args.dataset == 'merged':
            trainset = MergedDataset(mode='train', base_size=args.base_size)
            valset = MergedDataset(mode='test', base_size=args.base_size)
        else:
            raise NotImplementedError

        self.train_data_loader = Data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        self.val_data_loader = Data.DataLoader(valset, batch_size=args.batch_size, shuffle=True)

        ## GPU
        if torch.cuda.is_available() and args.ngpu < torch.cuda.device_count():
            torch.cuda.set_device(args.ngpu)

        ## model
        self.net = get_segmentation_model(args.net_name)

        # self.net.apply(self.weight_init)
        self.net = self.net.cuda()

        ## criterion
        self.criterion = SoftLoULoss()

        ## lr scheduler
        self.scheduler = LR_Scheduler_Head(args.lr_scheduler, args.learning_rate,
                                      args.epochs, len(self.train_data_loader), lr_step=10)

        ## optimizer
        # self.optimizer = torch.optim.Adagrad(self.net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.learning_rate,
                                         momentum=0.9, weight_decay=1e-4)

        ## evaluation metrics
        self.metric = SegmentationMetricTPFNFP(nclass=1)
        self.best_miou = 0
        self.best_fmeasure = 0
        self.eval_loss = 0 # tmp values
        self.miou = 0
        self.fmeasure = 0

        ## folders
        folder_name = '%s_%s' % (time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time())),
                                         args.net_name)
        self.save_folder = osp.join('result/', args.dataset, folder_name)
        self.save_pkl = osp.join(self.save_folder, 'checkpoint')
        if not osp.exists('result'):
            os.mkdir('result')
        if not osp.exists(osp.join('result/', args.dataset)):
            os.mkdir(osp.join('result/', args.dataset))
        if not osp.exists(self.save_folder):
            os.mkdir(self.save_folder)
        if not osp.exists(self.save_pkl):
            os.mkdir(self.save_pkl)

        ## SummaryWriter
        self.writer = SummaryWriter(log_dir=self.save_folder)
        self.writer.add_text(folder_name, 'Args:%s, ' % args)

        self.iter_num = 0

        ## Print info
        current_device = torch.cuda.current_device()
        print('Folder: %s' % self.save_folder)
        print('Args: %s' % args)
        print('Net name: %s' % args.net_name)
        print('ngpu: %d-%d %s' % (current_device, torch.cuda.device_count()-1,
                                  torch.cuda.get_device_name(current_device)))

    def training(self, epoch):
        # training step
        losses = []

        tbar = tqdm(self.train_data_loader)
        for i, (data, labels) in enumerate(tbar):
            self.net.train()
            self.scheduler(self.optimizer, i, epoch, self.best_miou)

            data = data.cuda()
            labels = labels.cuda()
            output = self.net(data)
            loss = self.criterion(output, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if (self.iter_num % args.save_iter_step) == 0:
                self.validation()
                self.writer.add_scalar('Losses/train loss', np.mean(losses), self.iter_num)
                self.writer.add_scalar('Learning rate/', trainer.optimizer.param_groups[0]['lr'], self.iter_num)

            tbar.set_description('Epoch:%3d, lr:%f, train loss:%f, eval loss:%f, miou:%f/%f, fmeasure:%f/%f'
                             % (epoch, trainer.optimizer.param_groups[0]['lr'], np.mean(losses),
                                self.eval_loss, self.miou, self.best_miou, self.fmeasure, self.best_fmeasure))

            self.iter_num += 1

    def validation(self):
        self.metric.reset()

        eval_losses = []
        self.net.eval()
        # tbar = tqdm(self.val_data_loader)
        for i, (data, labels) in enumerate(self.val_data_loader):
            with torch.no_grad():
                output = self.net(data.cuda())
                output = output.cpu()

            loss = self.criterion(output, labels)
            eval_losses.append(loss.item())

            self.metric.update(labels, output)
            # miou, prec, recall, fmeasure = self.metric.get()

            # tbar.set_description('  Epoch:%3d, eval loss:%f, mIoU:%f, fmeasure:%f, prec:%f, recall:%f'
            #                      %(epoch, np.mean(eval_losses), miou, fmeasure, prec, recall))

        miou, prec, recall, fmeasure = self.metric.get()
        pkl_name = 'Iter-%5d_mIoU-%.4f_fmeasure-%.4f.pkl' % (self.iter_num, miou, fmeasure)

        if miou > self.best_miou:
            self.best_miou = miou
            torch.save(self.net, osp.join(self.save_pkl, pkl_name))
        if fmeasure > self.best_fmeasure:
            self.best_fmeasure = fmeasure

        self.writer.add_scalar('Losses/eval_loss', np.mean(eval_losses), self.iter_num)
        self.writer.add_scalar('Eval/mIoU', miou, self.iter_num)
        self.writer.add_scalar('Eval/Fmeasure', fmeasure, self.iter_num)
        self.writer.add_scalar('Best/mIoU', self.best_miou, self.iter_num)
        self.writer.add_scalar('Best/Fmeasure', self.best_fmeasure, self.iter_num)

        self.eval_loss, self.miou, self.fmeasure = np.mean(eval_losses), miou, fmeasure


if __name__ == '__main__':
    args = parse_args()

    # set_seeds(args.seed)

    trainer = Trainer(args)
    for epoch in range(args.epochs):
        trainer.training(epoch)

    print('Best mIoU: %.5f, Best Fmeasure: %.5f\n\n' % (trainer.best_miou, trainer.best_fmeasure))





