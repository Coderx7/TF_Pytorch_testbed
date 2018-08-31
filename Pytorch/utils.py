
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator
# from layers import LearnedGroupConv, CondensingLinear, CondensingConv, Conv

import os, sys, time
import numpy as np
import matplotlib
import pdb, shutil, random

matplotlib.use('agg')
import matplotlib.pyplot as plt

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy

  def update(self, idx, train_loss, train_acc, val_loss, val_acc):
    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
    self.epoch_losses  [idx, 0] = train_loss
    self.epoch_losses  [idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    #print('val_acc {0} == {1}'.format(float("{0:.4f}".format(val_acc)), float("{0:.4f}".format(self.epoch_accuracy[:self.current_epoch, 1].max()))))
    return self.max_accuracy(False) == float("{0:.4f}".format(val_acc))

  def max_accuracy(self, istrain):
    if self.current_epoch <= 0: return 0
    if istrain: 
        return self.epoch_accuracy[:self.current_epoch, 0].max()
    else:       
        # print('all acc: {0} / max: {1}'.format(self.epoch_accuracy[:self.current_epoch, 1],
        #                 self.epoch_accuracy[:self.current_epoch, 1].max() ))
        return float("{0:.4f}".format(self.epoch_accuracy[:self.current_epoch, 1].max()))
  
  def plot_curve(self, save_path):
    title = 'the accuracy/loss curve of train/val'
    dpi = 90  
    width, height = 1200, 800
    legend_fontsize = 10
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
    y_axis = np.zeros(self.total_epoch)

    plt.xlim(0, self.total_epoch)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 5
    plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('the training epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
  
    y_axis[:] = self.epoch_accuracy[:, 0]
    plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    
    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 1]
    plt.plot(x_axis, y_axis*50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    if save_path is not None:
      fig.savefig(save_path, dpi=300)#, bbox_inches='tight')
      print ('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)
    

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.localtime(time.time()) ))
  return string

def convert_secs2time(epoch_time):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  return need_hour, need_mins, need_secs

def time_file_str():
  ISOTIMEFORMAT='%Y-%m-%d'
  string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.localtime(time.time()) ))
  return string + '-{}'.format(random.randint(1, 10000))





# count_ops = 0
# count_params = 0


# def get_num_gen(gen):
#     return sum(1 for x in gen)


# def is_pruned(layer):
#     try:
#         layer.mask
#         return True
#     except AttributeError:
#         return False


# def is_leaf(model):
#     return get_num_gen(model.children()) == 0


# def convert_model(model, args):
#     for m in model._modules:
#         child = model._modules[m]
#         if is_leaf(child):
#             if isinstance(child, nn.Linear):
#                 model._modules[m] = CondensingLinear(child, 0.5)
#                 del(child)
#         elif is_pruned(child):
#             model._modules[m] = CondensingConv(child)
#             del(child)
#         else:
#             convert_model(child, args)


# def get_layer_info(layer):
#     layer_str = str(layer)
#     type_name = layer_str[:layer_str.find('(')].strip()
#     return type_name


# def get_layer_param(model):
#     return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


# ### The input batch size should be 1 to call this function
# def measure_layer(layer, x):
#     global count_ops, count_params
#     delta_ops = 0
#     delta_params = 0
#     multi_add = 1
#     type_name = get_layer_info(layer)

#     ### ops_conv
#     if type_name in ['Conv2d']:
#         out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
#                     layer.stride[0] + 1)
#         out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
#                     layer.stride[1] + 1)
#         delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
#                 layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
#         delta_params = get_layer_param(layer)

#     ### ops_learned_conv
#     elif type_name in ['LearnedGroupConv']:
#         measure_layer(layer.relu, x)
#         measure_layer(layer.norm, x)
#         conv = layer.conv
#         out_h = int((x.size()[2] + 2 * conv.padding[0] - conv.kernel_size[0]) /
#                     conv.stride[0] + 1)
#         out_w = int((x.size()[3] + 2 * conv.padding[1] - conv.kernel_size[1]) /
#                     conv.stride[1] + 1)
#         delta_ops = conv.in_channels * conv.out_channels * conv.kernel_size[0] * \
#                 conv.kernel_size[1] * out_h * out_w / layer.condense_factor * multi_add
#         delta_params = get_layer_param(conv) / layer.condense_factor

#     ### ops_nonlinearity
#     elif type_name in ['ReLU']:
#         delta_ops = x.numel()
#         delta_params = get_layer_param(layer)

#     ### ops_pooling
#     elif type_name in ['AvgPool2d']:
#         in_w = x.size()[2]
#         kernel_ops = layer.kernel_size * layer.kernel_size
#         out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
#         out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
#         delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
#         delta_params = get_layer_param(layer)

#     elif type_name in ['AdaptiveAvgPool2d']:
#         delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
#         delta_params = get_layer_param(layer)

#     ### ops_linear
#     elif type_name in ['Linear']:
#         weight_ops = layer.weight.numel() * multi_add
#         bias_ops = layer.bias.numel()
#         delta_ops = x.size()[0] * (weight_ops + bias_ops)
#         delta_params = get_layer_param(layer)

#     ### ops_nothing
#     elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout','MaxPool2d']:
#         delta_params = get_layer_param(layer)

#     ### unknown layer type
#     else:
#         raise TypeError('unknown layer type: %s' % type_name)

#     count_ops += delta_ops
#     count_params += delta_params
#     return


# def measure_model(model, H, W):
#     global count_ops, count_params
#     count_ops = 0
#     count_params = 0
#     data = Variable(torch.zeros(1, 3, H, W))

#     def should_measure(x):
#         return is_leaf(x) or is_pruned(x)

#     def modify_forward(model):
#         for child in model.children():
#             if should_measure(child):
#                 def new_forward(m):
#                     def lambda_forward(x):
#                         measure_layer(m, x)
#                         return m.old_forward(x)
#                     return lambda_forward
#                 child.old_forward = child.forward
#                 child.forward = new_forward(child)
#             else:
#                 modify_forward(child)

#     def restore_forward(model):
#         for child in model.children():
#             # leaf node
#             if is_leaf(child) and hasattr(child, 'old_forward'):
#                 child.forward = child.old_forward
#                 child.old_forward = None
#             else:
#                 restore_forward(child)

#     modify_forward(model)
#     model.forward(data)
#     restore_forward(model)

#     return count_ops, count_params


# import os
# import os.path
# import hashlib
# import errno
# from tqdm import tqdm


# def gen_bar_updater(pbar):
#     def bar_update(count, block_size, total_size):
#         if pbar.total is None and total_size:
#             pbar.total = total_size
#         progress_bytes = count * block_size
#         pbar.update(progress_bytes - pbar.n)

#     return bar_update


# def check_integrity(fpath, md5=None):
#     if md5 is None:
#         return True
#     if not os.path.isfile(fpath):
#         return False
#     md5o = hashlib.md5()
#     with open(fpath, 'rb') as f:
#         # read in 1MB chunks
#         for chunk in iter(lambda: f.read(1024 * 1024), b''):
#             md5o.update(chunk)
#     md5c = md5o.hexdigest()
#     if md5c != md5:
#         return False
#     return True


# def download_url(url, root, filename, md5):
#     from six.moves import urllib

#     root = os.path.expanduser(root)
#     fpath = os.path.join(root, filename)

#     try:
#         os.makedirs(root)
#     except OSError as e:
#         if e.errno == errno.EEXIST:
#             pass
#         else:
#             raise

#     # downloads file
#     if os.path.isfile(fpath) and check_integrity(fpath, md5):
#         print('Using downloaded and verified file: ' + fpath)
#     else:
#         try:
#             print('Downloading ' + url + ' to ' + fpath)
#             urllib.request.urlretrieve(
#                 url, fpath,
#                 reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True))
#             )
#         except:
#             if url[:5] == 'https':
#                 url = url.replace('https:', 'http:')
#                 print('Failed download. Trying https -> http instead.'
#                       ' Downloading ' + url + ' to ' + fpath)
#                 urllib.request.urlretrieve(
#                     url, fpath,
#                     reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True))
#                 )


# def list_dir(root, prefix=False):
#     """List all directories at a given root

#     Args:
#         root (str): Path to directory whose folders need to be listed
#         prefix (bool, optional): If true, prepends the path to each result, otherwise
#             only returns the name of the directories found
#     """
#     root = os.path.expanduser(root)
#     directories = list(
#         filter(
#             lambda p: os.path.isdir(os.path.join(root, p)),
#             os.listdir(root)
#         )
#     )

#     if prefix is True:
#         directories = [os.path.join(root, d) for d in directories]

#     return directories


# def list_files(root, suffix, prefix=False):
#     """List all files ending with a suffix at a given root

#     Args:
#         root (str): Path to directory whose folders need to be listed
#         suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
#             It uses the Python "str.endswith" method and is passed directly
#         prefix (bool, optional): If true, prepends the path to each result, otherwise
#             only returns the name of the files found
#     """
#     root = os.path.expanduser(root)
#     files = list(
#         filter(
#             lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
#             os.listdir(root)
#         )
#     )

#     if prefix is True:
#         files = [os.path.join(root, d) for d in files]

#     return files