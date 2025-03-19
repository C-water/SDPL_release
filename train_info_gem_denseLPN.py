# -*- coding: utf-8 -*-

from __future__ import print_function, division

import os
# os.environ['CUDA_VISIBLE_DEVICES']='3'

# import neptune
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
# from PIL import Image
import copy
import time
from model_LPN_Gem_denseLPN import *
from random_erasing import RandomErasing
from autoaugment import ImageNetPolicy, CIFAR10Policy
import yaml
import math
from shutil import copyfile
from utils import *
import numpy as np
from folder import ImageFolder
from scipy import io
from PIL import Image

# print(torch.version.cuda,"torch.version.cuda")
# 11.1 torch.version.cuda

version = torch.__version__
# fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError:  # will be 3.x series
    print(
        'This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='3', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='debug', type=str, help='output model name')
parser.add_argument('--pool', default='avg', type=str, help='pool avg')
parser.add_argument('--data_dir_train',default='../Dataset/University1652/train',type=str, help='training dir path')
parser.add_argument('--data_dir_val',default='../Dataset/University1652/test',type=str, help='test dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--val_batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--stride', default=1, type=int, help='stride')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='the number of views')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_NAS', action='store_true', help='use NAS')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
parser.add_argument('--droprate', default=0.75, type=float, help='drop rate')
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation')
parser.add_argument('--resume', action='store_true', help='use resume trainning')
parser.add_argument('--share', action='store_true', help='share weight between different view')
parser.add_argument('--extra_Google', action='store_true', help='using extra noise Google')
parser.add_argument('--LPN', action='store_true', help='use LPN')
parser.add_argument('--dense_LPN', action='store_true', help='use dense LPN')
parser.add_argument('--swin', action='store_true', help='use swin')
parser.add_argument('--resnet', action='store_true', help='use resnet as backbone')
parser.add_argument('--decouple', action='store_true', help='use decouple')
parser.add_argument('--block', default=4, type=int, help='the num of block')
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory')
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
opt = parser.parse_args()

# def seed_torch(seed=opt.seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#     torch.use_deterministic_algorithms(False)
# print('random seed---------------------:', opt.seed)
# seed_torch(opt.seed)

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = 0


fp16 = opt.fp16
data_dir_train = opt.data_dir_train
data_dir_val = opt.data_dir_val

name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#
transform_train_list = [
    # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_satellite_list = [
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomAffine(90),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(opt.h, opt.w), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list
    transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                       hue=0)] + transform_satellite_list

if opt.DA:
    transform_train_list = [ImageNetPolicy()] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
    'satellite': transforms.Compose(transform_satellite_list)
}

transform_move_list = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_all = ''
if opt.train_all:
    train_all = '_all'


############# train image load  ###################################################
train_image_datasets = {}
train_image_datasets['satellite'] = ImageFolder(os.path.join(data_dir_train, 'satellite'),
                                          data_transforms['satellite'])
train_image_datasets['drone'] = ImageFolder(os.path.join(data_dir_train, 'drone'),
                                          data_transforms['train'])
train_dataloaders = {x: torch.utils.data.DataLoader(train_image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=4, pin_memory=False) # 8 workers may work faster
              for x in ['satellite', 'drone']}

train_dataset_sizes = {x: len(train_image_datasets[x]) for x in ['satellite','drone']}
class_names = train_image_datasets['satellite'].classes
print(train_dataset_sizes)


############ val image load  ###################################################
# val_image_datasets = {}
# val_image_datasets['gallery_satellite'] = ImageFolder(os.path.join(data_dir_val, 'gallery_satellite'),
#                                           data_transforms['val'])
# val_image_datasets['query_drone'] = ImageFolder(os.path.join(data_dir_val, 'query_drone'),
#                                           data_transforms['val'])
# val_image_datasets['query_satellite'] = ImageFolder(os.path.join(data_dir_val, 'query_satellite'),
#                                           data_transforms['val'])
# val_image_datasets['gallery_drone'] = ImageFolder(os.path.join(data_dir_val, 'gallery_drone'),
#                                           data_transforms['val'])
# val_dataloaders = {x: torch.utils.data.DataLoader(val_image_datasets[x], batch_size=opt.val_batchsize,
#                                               shuffle=False, num_workers=8) for x in ['gallery_satellite', 'query_drone','gallery_drone','query_satellite']}
#
val_image_datasets = {}
val_image_datasets['gallery_satellite'] = ImageFolder(os.path.join(data_dir_val, 'gallery_satellite'),
                                          data_transforms['val'])
val_image_datasets['query_drone'] = ImageFolder(os.path.join(data_dir_val, 'query_drone'),
                                          data_transforms['val'])
val_dataloaders = {x: torch.utils.data.DataLoader(val_image_datasets[x], batch_size=opt.val_batchsize,
                                              shuffle=False, num_workers=8) for x in ['gallery_satellite', 'query_drone']}


#### torch.cuda.is_available()  ######################################
use_gpu = torch.cuda.is_available()

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc

def evaluate(qf, ql, gf, gl):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index
    # print(good_index)
    # print(index[0:10])
    junk_index = np.argwhere(gl == -1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def block_n(block):#！！
    if block == 4:
        block_n = 10
    elif block == 3:
        block_n = 6
    elif block == 2:
        block_n = 3
    return block_n

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        # print(path, v)
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths

def extract_feature(model, dataloaders, view_index = 1):
    features = torch.FloatTensor()
    # count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        # count += 1
        # print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda()
        if opt.LPN:
            ff = torch.FloatTensor(n,512,opt.block).zero_().cuda()
        elif opt.dense_LPN:
            ff = torch.FloatTensor(n, 512, block_n(opt.block)).zero_().cuda()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                if opt.views ==2:
                    if view_index == 1:
                        outputs, _ = model(input_img, None)
                    elif view_index ==2:
                        _, outputs = model(None, input_img)
                ff += outputs
        # norm feature
        if opt.LPN:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(opt.block)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        elif opt.dense_LPN:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(block_n(opt.block))
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features


def one_LPN_output(outputs, labels, criterion, block, dense_LPN=False):
    # part = {}
    sm = nn.Softmax(dim=1)
    num_part = block
    score = 0
    loss = 0
    if dense_LPN:
        for i in range(block_n(block)):
            part = outputs[i] #torch.Size([8, 701])
            score += sm(part)
            loss += criterion(part, labels)
    else:
        for i in range(num_part):
            part = outputs[i]
            score += sm(part)
            loss += criterion(part, labels)

    _, preds = torch.max(score.data, 1)

    return preds, loss


def val(model):
    # global best_test_recall, best_test_epoch
    config_path = os.path.join('./model', opt.name, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    model_test = copy.deepcopy(model)
    if opt.LPN:
        if len(gpu_ids) > 1:
            for i in range(opt.block):
                cls_name = 'classifier' + str(i)
                c = getattr(model_test.module, cls_name)
                c.classifier = nn.Sequential()
        else:
            for i in range(opt.block):
                cls_name = 'classifier' + str(i)
                c = getattr(model_test, cls_name)
                c.classifier = nn.Sequential()
    elif opt.dense_LPN:
        if len(gpu_ids) > 1:
            for i in range(block_n(opt.block)):
                cls_name = 'classifier' + str(i)
                c = getattr(model_test.module, cls_name)
                c.classifier = nn.Sequential()

        else:
            for i in range(block_n(opt.block)):
                cls_name = 'classifier' + str(i)
                c = getattr(model_test, cls_name)
                c.classifier = nn.Sequential()
    else:
        model_test.classifier.classifier = nn.Sequential()

    model_test = model_test.eval()
    model_test = model_test.cuda()
    query_name = 'query_drone'
    gallery_name = 'gallery_satellite'

    gallery_path = val_image_datasets[gallery_name].imgs
    query_path = val_image_datasets[query_name].imgs

    gallery_label, gallery_path = get_id(gallery_path)
    query_label, query_path = get_id(query_path)

    with torch.no_grad():
        query_feature = extract_feature(model_test, val_dataloaders[query_name])
        gallery_feature = extract_feature(model_test, val_dataloaders[gallery_name])

        time_elapsed = time.time() - since
        print('Test complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))


        # Save to Matlab for check
        # result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_path': gallery_path,
        #           'query_f': query_feature.numpy(), 'query_label': query_label, 'query_path': query_path}
        # query_feature = torch.FloatTensor(result['query_f'])
        # query_label = result['query_label']
        # gallery_feature = torch.FloatTensor(result['gallery_f'])
        # gallery_label = result['gallery_label']


        result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_path': gallery_path,
                  'query_f': query_feature.numpy(), 'query_label': query_label, 'query_path': query_path}
        io.savemat('./model/'+opt.name +'/'+'pytorch_result.mat', result)

        print(opt.name)
        result = io.loadmat('./model/'+opt.name +'/'+'pytorch_result.mat')#result = scipy.io.loadmat('pytorch_result.mat')
        query_feature = torch.FloatTensor(result['query_f'])
        query_label = result['query_label'][0]
        gallery_feature = torch.FloatTensor(result['gallery_f'])
        gallery_label = result['gallery_label'][0]

        query_feature = query_feature.cuda()
        gallery_feature = gallery_feature.cuda()

        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0

        for i in range(len(query_label)):
            ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp

        CMC = CMC.float()
        CMC = CMC / len(query_label)  # average CMC
        # print(round(len(gallery_label) * 0.01))
        current_test_recall = CMC[0] * 100
        print('Current val recall:{:4f}'.format(current_test_recall))
        with open(dir_name+'/terminal_log.txt', "a") as terminal_log:
            terminal_log.write(f"Current val recall: {current_test_recall:.4f}%\n")
        return current_test_recall

def train_model(model, model_test, criterion, optimizer, scheduler, epoch, warm_up, warm_iteration, num_epochs=25):
    epoch = epoch + start_epoch
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train']:
        if phase == 'train':
            model.train(True)  # Set model to training mode
        else:
            model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0.0
        running_corrects2 = 0.0

        for data, data2 in zip(train_dataloaders['satellite'], train_dataloaders['drone']):
            # get the inputs
            inputs, labels = data
            inputs2, labels2 = data2
            now_batch_size, c, h, w = inputs.shape
            if now_batch_size < opt.batchsize:  # skip the last batch
                continue

            if use_gpu:
                inputs = Variable(inputs.cuda().detach())
                inputs2 = Variable(inputs2.cuda().detach())
                labels = Variable(labels.cuda().detach())
                labels2 = Variable(labels2.cuda().detach())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            if opt.views == 2:
                outputs, outputs2 = model(inputs, inputs2)

            if (not opt.LPN) and (not opt.dense_LPN):
                _, preds = torch.max(outputs.data, 1)
                _, preds2 = torch.max(outputs2.data, 1)
                if opt.views == 2:
                    loss = criterion(outputs, labels) + criterion(outputs2, labels2)
            else:
                preds, loss = one_LPN_output(outputs, labels, criterion, opt.block, opt.dense_LPN)
                preds2, loss2 = one_LPN_output(outputs2, labels2, criterion, opt.block, opt.dense_LPN)
                if opt.views == 2:       # no implement this LPN model
                    loss = loss + loss2

            # backward + optimize only if in training phase
            if epoch < opt.warm_epoch:
                warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                loss *= warm_up

            if fp16:  # we use optimier to backward loss
                with amp.scale_loss(loss, optimizer) as scaled_loss: scaled_loss.backward()
            else:
                loss.backward(retain_graph=True)
            optimizer.step()

            ##########
            if opt.moving_avg < 1.0:
                update_average(model_test, model, opt.moving_avg)

            # statistics
            if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                running_loss += loss.item() * now_batch_size
            else:  # for the old version like 0.3.0 and 0.3.1
                running_loss += loss.data[0] * now_batch_size
            running_corrects += float(torch.sum(preds == labels.data))
            running_corrects2 += float(torch.sum(preds2 == labels2.data))

        epoch_loss = running_loss / train_dataset_sizes['satellite']
        epoch_acc = running_corrects / train_dataset_sizes['satellite']
        epoch_acc2 = running_corrects2 / train_dataset_sizes['satellite']

        if opt.views == 2:
            print('train Loss: {:.4f} Satellite_Acc: {:.4f}  Drone_Acc: {:.4f}'.format(epoch_loss, epoch_acc, epoch_acc2))
            with open(dir_name+'/terminal_log.txt', "a") as terminal_log:
                terminal_log.write(f"train Loss: {epoch_loss:.4f}, Satellite_Acc: {epoch_acc:.4f},Drone_Acc: {epoch_acc2:.4f}%\n")
            scheduler.step()

        #### save model ####
        if epoch%20 == 19:
            save_network(model, opt.name, epoch)

    return model, warm_up

######################################################################
# Finetuning the convnet
# ----------------------
# Load a pretrainied model and reset final fully connected layer.

if opt.views == 2:
    model = two_view_net(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share,
                        LPN=opt.LPN, dense_LPN=opt.dense_LPN, swin=opt.swin,resnet=opt.resnet)

opt.nclasses = len(class_names)
print('nclass--------------------:', opt.nclasses)

if start_epoch >= 40:
    opt.lr = opt.lr * 0.1

if (not opt.LPN) and (not opt.dense_LPN):##
    ignored_params = list(map(id, model.classifier.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': model.classifier.parameters(), 'lr': opt.lr}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)

elif opt.LPN:
    # ignored_params = list(map(id, model.model.fc.parameters() ))
    ignored_params =list()
    for i in range(opt.block):
        cls_name = 'classifier'+str(i)
        c = getattr(model, cls_name)
        ignored_params += list(map(id, c.parameters() ))

    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optim_params = [{'params': base_params, 'lr': 0.1*opt.lr}]
    for i in range(opt.block):
        cls_name = 'classifier'+str(i)
        c = getattr(model, cls_name)
        optim_params.append({'params': c.parameters(), 'lr': opt.lr})

    optimizer_ft = optim.SGD(optim_params, weight_decay=5e-4, momentum=0.9, nesterov=True)

elif opt.dense_LPN:
    ignored_params =list()
    for i in range(block_n(opt.block)):
        cls_name = 'classifier'+str(i)
        c = getattr(model, cls_name)
        ignored_params += list(map(id, c.parameters() ))

    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optim_params = [{'params': base_params, 'lr': 0.1*opt.lr}]
    for i in range(block_n(opt.block)):
        cls_name = 'classifier'+str(i)
        c = getattr(model, cls_name)
        optim_params.append({'params': c.parameters(), 'lr': opt.lr})

    optimizer_ft = optim.SGD(optim_params, weight_decay=5e-4, momentum=0.9, nesterov=True)


######### train epoch number ###########
if opt.moving_avg<1.0:
    model_test = copy.deepcopy(model)
    # num_epochs = 400
    num_epochs = 1000

else:
    model_test = None
    # num_epochs = 400
    num_epochs = 1000


######### Decays the lr ###########
# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.75) # 40 0.2
# CosineAnnealingLR
# warmup_epochs = 3
# scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, num_epochs - warmup_epochs,
#                                                         eta_min=1e-6)
# scheduler = GradualWarmupScheduler(optimizer_ft, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
# scheduler.step()


######################################################################

# Train and evaluate
dir_name = os.path.join('./model', name)
if not opt.resume:
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        

terminal_log = open(dir_name+'/terminal_log.txt', 'w')

##### model to gpu ################################
model = model.cuda()
if fp16:
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")

####### base loss function ##############
criterion = nn.CrossEntropyLoss()

warm_up = 0.1  # We start from the 0.1*lrRate

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

####  Set test information  ######################
best_test_recall = 0.0
best_test_epoch = 0

since = time.time()

warm_iteration = round(train_dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch
copyfile('./run.sh', dir_name + '/run.sh')
copyfile('./train_info_gem_denseLPN.py', dir_name + '/train_info_gem_denseLPN.py')
copyfile('./model_LPN_Gem_denseLPN.py', dir_name + '/model_LPN_Gem_denseLPN.py')
for epoch in range(num_epochs - start_epoch):
    # record every run
    # save opts
    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

    # train stage
    model, warm_up = train_model(model, model_test, criterion, optimizer_ft, exp_lr_scheduler, epoch, warm_up,
                                 warm_iteration, num_epochs=num_epochs)

    # val stage
    if epoch > 300 and epoch % 5 == 0:
        print('Val stage begin !!!')
        current_test_recall = val(model)

        if current_test_recall > best_test_recall:
            best_test_recall = current_test_recall
            best_test_epoch = epoch
            save_network(model, opt.name, epoch)
        print('Best val Acc: {:4f} Best epoch: {}'.format(best_test_recall, best_test_epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

print('Best val Acc: {:4f} Best epoch: {}'.format(best_test_recall, best_test_epoch))

with open(dir_name+'/terminal_log.txt', "a") as terminal_log:
    terminal_log.write(f"Best val Acc:{best_test_recall:.4f},  Best epoch: {best_test_epoch:.4f}%\n")



