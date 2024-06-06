from torch.utils.data import DataLoader
from dcmnet.loss import *
from dcmnet.dataset.uavid import *
from catalyst.contrib.nn import Lookahead
from catalyst import utils
from dcmnet.model.DCMNet import DCMNet

# training hparam
max_epoch = 60
ignore_index = 255
train_batch_size = 8
val_batch_size = 8
lr = 1e-3
weight_decay = 2.5e-4
backbone_lr = 1e-4
backbone_weight_decay = 2.5e-4
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "dcmnet"
weights_path = "model_weights/uavid/{}".format(weights_name)
test_weights_name = "last"
log_name = 'uavid/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = DCMNet(num_classes=num_classes)
# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)


# define the dataloader

train_dataset = UAVIDDataset(data_root='data/uavid/train', img_dir='images', mask_dir='masks',
                             mode='train', mosaic_ratio=0.25, transform=train_aug, img_size=(1024, 1024))

val_dataset = UAVIDDataset(data_root='data/uavid/val', img_dir='images', mask_dir='masks', mode='val',
                           mosaic_ratio=0.0, transform=val_aug, img_size=(1024, 1024))


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=8,
                          persistent_workers=True,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=8,
                        persistent_workers=True,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

