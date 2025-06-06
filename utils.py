import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from sklearn.metrics import confusion_matrix


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = config.mode, 
            factor = config.factor, 
            patience = config.patience, 
            threshold = config.threshold, 
            threshold_mode = config.threshold_mode, 
            cooldown = config.cooldown, 
            min_lr = config.min_lr, 
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler



def save_imgs(img, msk, msk_pred, key_points, gt_pre, id, save_path, datasets, threshold=0.5, test_data_name=None):
    if os.path.exists(save_path + str(id) +'.png'):
        return
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    # kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9, kp10, kp11, kp12 = key_points
    # gt1, gt2, gt3, gt4, gt5 = gt_pre
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)
        # kp1 = kp1.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        # gt = gt.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        # kp = (kp > threshold)
        # gt = (gt > threshold)

    plt.figure(figsize=(100,100))

    plt.subplot(3,5,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3,5,2)
    plt.imshow(msk, cmap= 'gray')
    plt.axis('off')

    plt.subplot(3,5,3)
    plt.imshow(msk_pred, cmap = 'gray')
    plt.axis('off')

    for i in range(3):
        kp = key_points[i]
        kp = kp.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        plt.subplot(3,5,i+6)
        plt.imshow(kp, cmap = 'gray')
        plt.axis('off')

    for i in range(5):
        gt = gt_pre[i]
        gt = gt.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        plt.subplot(3,5,i+11)
        plt.imshow(gt, cmap = 'gray')
        plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(save_path + str(id) +'.png')
    plt.close()
    
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss
    

class GT_BceDiceLoss(nn.Module):
    # def __init__(self, wb=1, wd=1):
    #     super(GT_BceDiceLoss, self).__init__()
    #     self.bcedice = BceDiceLoss(1, 2)
    #     self.bcedice2 = BceDiceLoss(0.5, 1)
  
    def __init__(self, wb=1, wd=1):
      super().__init__()
      self.wb = wb
      self.wd = wd

    def forward(self, out, target):
      # remove any use of points unless necessary
      bce = F.binary_cross_entropy(out, target)
      smooth = 1e-5
      intersection = (out * target).sum()
      dice = (2. * intersection + smooth) / (out.sum() + target.sum() + smooth)
      return self.wb * bce + self.wd * (1 - dice)



class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        if len(data) == 3:
            image, mask, points = data
            return torch.tensor(image).permute(2,0,1), torch.tensor(mask).permute(2,0,1), torch.tensor(points).permute(2,0,1)
        else: 
            image, mask = data
            return torch.tensor(image).permute(2,0,1), torch.tensor(mask).permute(2,0,1)
       

class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w
    def __call__(self, data):
        if len(data) == 3:
            image, mask, points = data
            return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w]), TF.resize(points, [self.size_h, self.size_w])
        else: 
            image, mask = data
            return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])
       

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        if len(data) == 3:
            image, mask, points = data
            if random.random() < self.p: return TF.hflip(image), TF.hflip(mask), TF.hflip(points)
            else: return image, mask, points
        else: 
            image, mask = data
            if random.random() < self.p: return TF.hflip(image), TF.hflip(mask)
            else: return image, mask
            

class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        if len(data) == 3:
            image, mask, points = data
            if random.random() < self.p: return TF.vflip(image), TF.vflip(mask), TF.vflip(points)
            else: return image, mask, points
        else: 
            image, mask = data
            if random.random() < self.p: return TF.vflip(image), TF.vflip(mask)
            else: return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0,360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p
    def __call__(self, data):
        if len(data) == 3:
            image, mask, points = data
            if random.random() < self.p: return TF.rotate(image,self.angle), TF.rotate(mask,self.angle), TF.rotate(points,self.angle)
            else: return image, mask, points
        else: 
            image, mask = data
            if random.random() < self.p: return TF.rotate(image,self.angle), TF.rotate(mask,self.angle)
            else: return image, mask 


class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'BUSI':
            if train:
                self.mean = 83.6784
                self.std = 21.2149
            else:
                self.mean = 85.2134
                self.std = 20.1582
            
    def __call__(self, data):
        if len(data) == 3:
            img, msk, pnt = data
            img_normalized = (img-self.mean)/self.std
            img_normalized = ((img_normalized - np.min(img_normalized)) 
                                / (np.max(img_normalized)-np.min(img_normalized))) * 255.
            return img_normalized, msk, pnt
        else: 
            img, msk = data
            img_normalized = (img-self.mean)/self.std
            img_normalized = ((img_normalized - np.min(img_normalized)) 
                                / (np.max(img_normalized)-np.min(img_normalized))) * 255.
            return img_normalized, msk