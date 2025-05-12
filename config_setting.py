from torchvision import transforms
from utils import myNormalize, myToTensor, myRandomHorizontalFlip, myRandomVerticalFlip, myRandomRotation, myResize, GT_BceDiceLoss
from datetime import datetime
import os

class setting_config:
    def __init__(self):
        self.network = 'lbunet'
        self.model_config = {
            'num_classes': 1,
            'input_channels': 3,
            'c_list': [8, 16, 24, 32, 48, 64],
            'bridge': True,
            'gt_ds': True,
        }

        self.datasets = 'isic18'
        if self.datasets == 'isic18':
            self.data_path = '/content/drive/MyDrive/KaggleDatasets/ISIC_Datasets/ISIC_2018_Dataset/'
        else:
            raise Exception('datasets is not right!')

        self.criterion = GT_BceDiceLoss(wb=1, wd=1)
        self.pretrained_path = './pre_trained/'
        self.num_classes = 1
        self.input_size_h = 256
        self.input_size_w = 256
        self.input_channels = 3
        self.distributed = False
        self.local_rank = -1
        self.num_workers = 4
        self.seed = 42
        self.world_size = None
        self.rank = None
        self.amp = False
        self.gpu_id = '0'  # default GPU ID
        self.batch_size = 8
        self.epochs = 300

        self.work_dir = os.path.join('results', self.network + '_' + self.datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss'))

        self.print_interval = 20
        self.val_interval = 15
        self.save_interval = 5
        self.threshold = 0.5

        self.train_transformer = transforms.Compose([
            myNormalize(self.datasets, train=True),
            myToTensor(),
            myRandomHorizontalFlip(p=0.5),
            myRandomVerticalFlip(p=0.5),
            myRandomRotation(p=0.5, degree=[0, 360]),
            myResize(self.input_size_h, self.input_size_w)
        ])

        self.test_transformer = transforms.Compose([
            myNormalize(self.datasets, train=False),
            myToTensor(),
            myResize(self.input_size_h, self.input_size_w)
        ])

        self.opt = 'AdamW'
        assert self.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
        self.lr = 0.001
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        self.weight_decay = 1e-2
        self.amsgrad = False

        self.sch = 'CosineAnnealingLR'
        self.T_max = 50
        self.eta_min = 0.00001
        self.last_epoch = -1
