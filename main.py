import huva.th_util as thu
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torchvision
from pprint import pprint, pformat
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name',  type=str, default='')
parser.add_argument('-m', '--mode',  type=str, default='')
parser.add_argument('-o', '--optimizer',    type=str, default='SGD')
parser.add_argument('-b', '--batch-size',   type=int, default=128)
parser.add_argument('-nb','--num-base',     type=int, default=32)
parser.add_argument('-nc','--num-centroids',type=int, default=40)
parser.add_argument('-wd','--weight-decay', type=float, default=0.0005)
parser.add_argument('-lr','--learning-rate',type=float, default=0.1) # default for SGD
parser.add_argument('--use-batchnorm',      type=bool, default=True)
parser.add_argument('--logfile',     type=str, default='')
parser.add_argument('--graphfolder', type=str, default='')
parser.add_argument('--force-name', action='store_true')
args = parser.parse_args()

def make_data(batch_size=128):
    global dataset, loader, dataset_test, loader_test
    (dataset, loader), (dataset_test, loader_test) = thu.make_data_cifar10(batch_size)

def make_model(base=32, centroids=40, use_batchnorm=True):
    global model, model_conf
    bn = base # shorter name
    model_conf = [
        ('input',   (3,   None)),
        ('conv1_1', (bn*1,None)),
        ('conv1_2', (bn*1,None)),
        ('pool1'  , (2,   2)),
        ('conv2_1', (bn*2,None)),
        ('conv2_2', (bn*2,None)),
        ('pool2'  , (2,   2)),
        ('conv3_1', (bn*4,None)),
        ('conv3_2', (bn*4,None)),
       #('conv3_3', (bn*4,None)),
        ('pool3'  , (2,   2)),
        ('conv4_1', (bn*8,None)),
        ('conv4_2', (bn*8,None)),
       #('conv4_3', (bn*8,None)),
        ('pool4'  , (2,   2)),
        ('conv5_1', (bn*8,None)),
        ('conv5_2', (bn*8,None)),
       #('conv5_3', (bn*8,None)),
        ('pool5'  , (2,   2)),
        ('fc6'    , (bn*8,None)),
        ('logit'  , (centroids, None)),
        ('flatter', (None,None)),
    ]
    batchnorm = nn.BatchNorm2d if use_batchnorm else None
    model = thu.make_cnn_with_conf(model_conf, batchnorm=batchnorm)
    model = model.cuda()

def make_optimizer(optimizer_name='SGD', weight_decay=0.0005, lr=0.1):
    global optimizer
    if optimizer_name=='Adam':
        lr = 0.001 # ignore lr advice
        optimizer = thu.MonitoredAdam(model.parameters(), lr, weight_decay=weight_decay)
    elif optimizer_name=='SGD':
        optimizer = thu.MonitoredSGD(model.parameters(), lr, weight_decay=weight_decay)
    else:
        assert False, 'unknown optimizer mode: {}'.format(optimizer_name)

def make_all():
    global logger
    make_data(args.batch_size)
    make_model(base=args.num_base, centroids=args.num_centroids, use_batchnorm=args.use_batchnorm)
    make_optimizer(args.optimizer, args.weight_decay, args.learning_rate)
    if args.graphfolder == '': 
        args.graphfolder = 'logs/{}_graphs/'.format(args.name)
    if args.logfile == '':
        args.logfile = 'logs/{}.log'.format(args.name)
    model.args = args
    if not args.force_name and os.path.exists(args.logfile):
        assert False, 'abort because {} already exists'.format(args.logfile)
    logger = thu.LogPrinter(args.logfile)
    logger.log(str(model))
    logger.log(pformat(args.__dict__))

epoch_trained = 0
T = 1.0

def train(num_epochs, report_interval=50):
    global epoch_trained, T
    model.train()
    max_stat = None
    softmax = nn.Softmax()
    for epoch in xrange(num_epochs):
        for batch, (imgs, labels) in enumerate(loader):
            """ forward """
            imgs = imgs.cuda()
            v_imgs = Variable(imgs)
            v_out0 = model(v_imgs)
            v_out1 = v_out0 / T
            v_out2 = softmax(v_out1)
            #
            max_mask = thu.gumbel_max(v_out2.data).float()
            """ backward """
            optimizer.zero_grad()
            v_out0.backward(-max_mask) # we want to perform maximization, so negate the ones
            optimizer.step()
            if max_stat is None:
                max_stat = max_mask.sum(0)
            else:
                max_stat += max_mask.sum(0)
            """ report """
            if (batch+1) % report_interval == 0:
                logger.log("Epoch {}, Batch {}, [{:5f}/{:5f}]".format(
                    epoch, batch+1, 
                    optimizer.update_norm, thu.get_model_param_norm(model)))
                logger.log([max_stat[0,i] for i in xrange(max_stat.size(1))])
                max_stat.fill_(0)
        if T > 0.0000001:
            T /= 2

def save_layer_hist(layer):
    name_layer = {name:module for name,module in model.named_modules() if module is layer}
    name = name_layer.keys()[0]
    name_output = thu.collect_output_over_loader(model, name_layer, loader_test, max_batches=20)
    output = name_output[name]
    stats = thu.get_output_stats(output)
    num_units = stats.outputt.size(0)
    if not os.path.exists(args.graphfolder):
        os.mkdir(args.graphfolder)
    for i in xrange(num_units):
        savepath = os.path.join(args.graphfolder, 'hist_{}_{}.jpg'.format(name, i))
        thu.save_output_hist(stats.outputt, i, savepath)

def save_layer_gbp(layer, K, mode='show'):
    name_layer = {name:module for name,module in model.named_modules() if module is layer}
    name = name_layer.keys()[0]
    name_output = thu.collect_output_over_loader(model, name_layer, loader_test, max_batches=20)
    output = name_output[name]
    num_units = output.size(1)
    for i in xrange(num_units):
        inputs, grads = thu.guided_backprop_layer_unit(model, layer, output, i, dataset_test, top_k=K)
        inputs = thu.normalize(inputs)
        grads  = thu.normalize(grads)
        plt.close()
        f, axs = plt.subplots(2, K, figsize=(19,5))
        for j in xrange(K):
            axs[0,j].imshow(inputs[j].cpu().numpy().transpose([1,2,0]))
            axs[0,j].axis('off')
            axs[1,j].imshow(grads[j].cpu().numpy().transpose([1,2,0]))
            axs[1,j].axis('off')
        plt.tight_layout()
        if mode=='show':
            plt.show()
        elif mode=='save':
            savepath = os.path.join(args.graphfolder, 'gbp_{}_{}.jpg'.format(name, i))
            plt.savefig(savepath)
            plt.close()
        else:
            assert False, 'unknown mode {}'.format(mode)

def save_model():
    torch.save(model, 'logs/{}.pth'.format(args.name))

def load_model(path):
    global model, args
    model = torch.load(path)
    args = model.args
