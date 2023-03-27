

import os
import sys
import time

import numpy as np
from datetime import datetime
import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from models.dump_helper import dump_results
from models.loss_helper_labeled import get_labeled_loss
from models.votenet_iou_branch import VoteNet as Detector
# from models.loss_helper import get_loss, get_loss_iou
# from models.loss_helper_iou_jitter import get_loss_iou_jitter
from utils import pc_util
from utils.nn_distance import nn_distance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pytorch_utils import BNMomentumScheduler
from tf_visualizer import Visualizer as TfVisualizer
from ap_helper import APCalculator, parse_predictions, parse_groundtruths
from munch import Munch
import json
from sunrgbd.sunrgbd_ssl_dataset import SunrgbdSSLLabeledDataset, SunrgbdSSLUnlabeledDataset
from sampler import ImbalancedDatasetSampler

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sunrgbd', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='pretrain_sunrgbd', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=40000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=128, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='seed_fps',
                    help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
parser.add_argument('--max_epoch', type=int, default=901, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='400, 600, 800',
                    help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.1, 0.1, 0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--dump_results', action='store_true', help='Dump results.')
parser.add_argument('--iou_weight', type=float, default=1.0)
parser.add_argument('--labeled_sample_list', default='sunrgbd_v1_train_0.05.txt', type=str)
parser.add_argument('--print_interval', type=int, default=10, help='batch inverval to print loss')
parser.add_argument('--eval_interval', type=int, default=50, help='epoch inverval to evaluate model')
parser.add_argument('--section_size', type=int, default=100, help='epochs in each section')
parser.add_argument('--save_interval', type=int, default=200, help='epoch interval to save model')


parser.add_argument('--resume', action='store_true')
FLAGS = parser.parse_args()


# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert (len(LR_DECAY_STEPS) == len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, LOG_DIR)
DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
FLAGS.DUMP_DIR = DUMP_DIR

PERFORMANCE_FOUT = open(os.path.join(LOG_DIR, 'best.txt'), 'w')
print(FLAGS)


# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)' % (LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s' % (LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# Create Dataset and Dataloader
if FLAGS.dataset == 'sunrgbd':
    from sunrgbd.sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, MAX_NUM_OBJ
    from sunrgbd.model_util_sunrgbd import SunrgbdDatasetConfig

    DATASET_CONFIG = SunrgbdDatasetConfig()
    TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=NUM_POINT,
                                                 augment=True,
                                                 use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
                                                 use_v1=(not FLAGS.use_sunrgbd_v2),
                                                 labeled_sample_list=FLAGS.labeled_sample_list)
    TEST_DATASET = SunrgbdDetectionVotesDataset('val', num_points=NUM_POINT,
                                                augment=False,
                                                use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
                                                use_v1=(not FLAGS.use_sunrgbd_v2))#, labeled_sample_list='sun_vis.txt')# single=FLAGS.vis_single)

    Validate_DATASET = SunrgbdSSLUnlabeledDataset(labeled_sample_list=FLAGS.labeled_sample_list,
                                                   num_points=NUM_POINT,
                                                   use_color=FLAGS.use_color,
                                                   use_height=(not FLAGS.no_height),
                                                   use_v1=(not FLAGS.use_sunrgbd_v2),
                                                   augment=False)

elif FLAGS.dataset == 'scannet':
    from scannet.scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
    from scannet.model_util_scannet import ScannetDatasetConfig

    DATASET_CONFIG = ScannetDatasetConfig()
    TRAIN_DATASET = ScannetDetectionDataset('train', num_points=NUM_POINT,
                                            augment=True,
                                            use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
                                            labeled_sample_list=FLAGS.labeled_sample_list)
    TEST_DATASET = ScannetDetectionDataset('val', num_points=NUM_POINT,
                                           augment=False,
                                           use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))#, labeled_sample_list='scan_vis.txt')#
else:
    print('Unknown dataset %s. Exiting...' % (FLAGS.dataset))
    exit(-1)

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, worker_init_fn=my_worker_init_fn)
Valid_DATALOADER = DataLoader(Validate_DATASET, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, worker_init_fn=my_worker_init_fn)

# Init the model and optimzier
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color) * 3 + int(not FLAGS.no_height) * 1


net = Detector(num_class=DATASET_CONFIG.num_class,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               num_size_cluster=DATASET_CONFIG.num_size_cluster,
               mean_size_arr=DATASET_CONFIG.mean_size_arr,
               dataset_config=DATASET_CONFIG,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling)

if torch.cuda.device_count() > 1:
    log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)
net.to(device)

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# Load checkpoint if there is any
it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])

    if FLAGS.resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE ** (int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch - 1)
if FLAGS.resume:
    bnm_scheduler.step(start_epoch)


def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i, lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# TFBoard Visualizers
TRAIN_VISUALIZER = TfVisualizer(FLAGS.log_dir, 'train')
TEST_VISUALIZER = TfVisualizer(FLAGS.log_dir, 'test')

# Used for AP calculation
CONFIG_DICT = {'remove_empty_box': False, 'use_3d_nms': True,
               'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True, 'use_iou_for_nms': False,
               'per_class_proposal': True, 'conf_thresh': 0.05, 'iou_weight': FLAGS.iou_weight,
               'dataset_config': DATASET_CONFIG }

print('CONFIG_DICT', CONFIG_DICT, DATASET_CONFIG)

# ------------------------------------------------------------------------- GLOBAL CONFIG END

AP_IOU_THRESHOLDS = [0.25, 0.5]


def tb_name(key):
    if 'loss' in key:
        return 'loss/' + key
    elif 'acc' in key:
        return 'acc/' + key
    elif 'ratio' in key:
        return 'ratio/' + key
    else:
        return 'other/' + key

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs
def update_dataloader():
    stat_dict = {}  # collect statistics
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type)
                          for iou_thresh in AP_IOU_THRESHOLDS]
    net.eval()  # set model to eval mode (for bn and dp)
    test_pred = []
    test_logits = []
    obj_logits = []
    test_min_confid = []
    dict_ave_validate_conf = Munch()
    for i in range(10):
        dict_ave_validate_conf['%d' %i] = 0
        dict_ave_validate_conf['%d_len' %i] = 0

    # for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
    # for batch_idx, batch_data_label in enumerate(Valid_DATALOADER):
    for batch_idx, batch_data_label in enumerate(Valid_DATALOADER):

        for key in batch_data_label:
            # batch_data_label[key] = torch.from_numpy(batch_data_label[key]).to(device)
            batch_data_label[key] = batch_data_label[key].to(device)


        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = net(inputs)

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]

        sem_cls_prob = softmax(end_points['sem_cls_scores'].detach().cpu().numpy()) # B,num_proposal
        obj_prob = softmax(end_points['objectness_scores'].detach().cpu().numpy()) # B,num_proposal
        pred_sem_cls = np.argmax(sem_cls_prob, -1)
        max_sem_prob = np.max(sem_cls_prob, -1)
        pos_obj = obj_prob[:, :, 1]
        objectness_mask = pos_obj > 0.9

        max_sem_prob = max_sem_prob*pos_obj
        valid_min_confid = np.zeros(pred_sem_cls.shape[0])
        for i in range(pred_sem_cls.shape[0]):
            current_min_confid = 1
            current_objectness_mask = objectness_mask[i]
            current_index = np.where(current_objectness_mask!=False)
            if current_index[0].shape[0] == 0:
                continue
            else:
                for index in current_index[0]:
                    current_label = pred_sem_cls[i, index]
                    dict_ave_validate_conf['%d' %current_label] += max_sem_prob[i, index]
                    dict_ave_validate_conf['%d_len' % current_label] +=1
                    if max_sem_prob[i, index] < current_min_confid:
                        current_min_confid = max_sem_prob[i, index]
                valid_min_confid[i] = current_min_confid
        test_min_confid.append(valid_min_confid)

    # Log statistics
    dict_ave_validate_conf_mean = Munch()


    test_min_confid = np.concatenate(test_min_confid)

    for i in range(10):
        if dict_ave_validate_conf['%d_len' % i] == 0:
            dict_ave_validate_conf_mean['%d' % i] = 0.1
        else:
            dict_ave_validate_conf_mean['%d' %i] = dict_ave_validate_conf['%d' % i]/dict_ave_validate_conf['%d_len' % i]

    f = open("dict_avgconf.txt", "w")
    js = json.dumps(dict_ave_validate_conf_mean)
    f.write(js)
    f.close()

    # TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=NUM_POINT,
    #                                              augment=True,
    #                                              use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
    #                                              use_v1=(not FLAGS.use_sunrgbd_v2),
    #                                              labeled_sample_list=FLAGS.labeled_sample_list)
    global TRAIN_DATALOADER
    TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
                              sampler=ImbalancedDatasetSampler(TRAIN_DATASET), num_workers=4, worker_init_fn=my_worker_init_fn)

def evaluate_one_epoch():
    stat_dict = {}  # collect statistics
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
                          for iou_thresh in AP_IOU_THRESHOLDS]
    net.eval()  # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = net(inputs)

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = get_labeled_loss(end_points, DATASET_CONFIG, CONFIG_DICT)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

        # Dump evaluation results for visualization
        if FLAGS.dump_results and batch_idx == 0 and EPOCH_CNT % 10 == 0:
            dump_results(end_points, DUMP_DIR, DATASET_CONFIG)

            # Log statistics
    TEST_VISUALIZER.log_scalars({tb_name(key): stat_dict[key] / float(batch_idx + 1) for key in stat_dict},
                                (EPOCH_CNT + 1) * len(TRAIN_DATALOADER) * BATCH_SIZE)
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f' % (key, stat_dict[key] / (float(batch_idx + 1))))

    map = []
    # Evaluate average precision
    for i, ap_calculator in enumerate(ap_calculator_list):
        print('-' * 10, 'iou_thresh: %f' % (AP_IOU_THRESHOLDS[i]), '-' * 10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string('eval %s: %f' % (key, metrics_dict[key]))
        TEST_VISUALIZER.log_scalars(
            {'metrics_' + str(AP_IOU_THRESHOLDS[i]) + '/' + key: metrics_dict[key] for key in metrics_dict if
             key in ['mAP', 'AR']},
            (EPOCH_CNT + 1) * len(TRAIN_DATALOADER) * BATCH_SIZE)
        map.append(metrics_dict['mAP'])

    mean_loss = stat_dict['loss'] / float(batch_idx + 1)
    return mean_loss, map


def train_one_epoch():
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step()  # decay BN momentum
    net.train()  # set model to training mode

    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        end_points = net.forward_with_pred_jitter(inputs)

        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]

        loss, end_points = get_labeled_loss(end_points, DATASET_CONFIG, CONFIG_DICT)
        loss.backward()
        optimizer.step()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = FLAGS.print_interval
        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            TRAIN_VISUALIZER.log_scalars({tb_name(key): stat_dict[key] / batch_interval for key in stat_dict},
                                         (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * BATCH_SIZE)
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0


BEST_MAP = [0.0, 0.0]


def train(start_epoch):
    global EPOCH_CNT
    loss = 0
    global BEST_MAP
    EPOCH_CNT = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f' % (bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))

        # in numpy 1.18.5 this actually sets `np.random.get_state()[1][0]` to default value
        # so the test data is consistent as the initial seed is the same
        np.random.seed()
        train_one_epoch()
        if EPOCH_CNT % FLAGS.eval_interval == 0 and EPOCH_CNT > 0:
            loss, map = evaluate_one_epoch()

            if map[0] + map[1] > BEST_MAP[0] + BEST_MAP[1]:
                BEST_MAP = map
                save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                             'optimizer_state_dict': optimizer.state_dict(),
                             'loss': loss
                             }
                try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                    save_dict['model_state_dict'] = net.module.state_dict()
                except:
                    save_dict['model_state_dict'] = net.state_dict()
                torch.save(save_dict, os.path.join(LOG_DIR, 'best_checkpoint_sum.tar'))
            PERFORMANCE_FOUT.write('epoch: ' + str(EPOCH_CNT) + '\n' + 'best: ' + \
                                   str(BEST_MAP[0].item()) + ', ' + str(BEST_MAP[1].item()) + '\n')
            PERFORMANCE_FOUT.flush()

        if EPOCH_CNT % FLAGS.section_size == 0 and EPOCH_CNT > 0:
            update_dataloader()

        # Save checkpoint
        save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': loss,
                     }
        try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))

        if EPOCH_CNT % FLAGS.save_interval == 0:
            # Save checkpoint
            save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                         'optimizer_state_dict': optimizer.state_dict(),
                         'loss': loss,
                         }
            try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                save_dict['model_state_dict'] = net.module.state_dict()
            except:
                save_dict['model_state_dict'] = net.state_dict()
            torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint_%d.tar' % EPOCH_CNT))


if __name__ == '__main__':
    train(start_epoch)