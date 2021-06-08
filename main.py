import random
import time
import warnings
import sys
import argparse
import copy

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from PIL import Image
import argparse
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
sys.path.append('.')
from model import DomainDiscriminator, Ensemble, DomainDiscriminator1
from model import DomainAdversarialLoss, ImageClassifier, resnet50,DomainAdversarialLoss1
import datasets
from datasets import esem_dataloader
from lib import AverageMeter, ProgressMeter, accuracy, ForeverDataIterator, AccuracyCounter
from lib import ResizeImage
from lib import StepwiseLR, get_entropy, get_confidence, get_consistency, norm, single_entropy
import yaml
import easydict
from os.path import join
from scipy.spatial.distance import cdist
import pdb

from easydl import FileListDataset
from eval import batch_hard_triplet_loss
from eval import batch_all_triplet_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['Normal', 'Pneumonia', 'COVID-19']
class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)
def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.0f" % (c,), color='Black', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()
def main(args: argparse.Namespace):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    dataset1 = None
    if args.data == 'Xray':
        dataset1 = Dataset(
            path=args.root,
            domains=['source', 'target'],
            files=[
                'source.txt',
                'target.txt',
            ],
            prefix=args.root)
    else:
        raise Exception(f'dataset {args.data} not supported!')

    print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
    print ("strat")

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        ResizeImage(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_tranform = transforms.Compose([
        ResizeImage(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    a, b, c = args.n_share, args.n_source_private, args.n_total
    common_classes = [i for i in range(a)]
    source_private_classes = [i + a for i in range(b)]
    target_private_classes = [i + a + b for i in range(c - a - b)]
    source_classes = common_classes + source_private_classes
    target_classes = common_classes + target_private_classes
    dataset= datasets.Xray
    train_source_dataset = dataset(root=args.root, data_list_file=args.source, filter_class=source_classes,
                                   transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = dataset(root=args.root, data_list_file=args.target, filter_class=target_classes,
                                   transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, data_list_file=args.target, filter_class=target_classes,
                          transform=val_tranform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    test_loader = val_loader

    target_test_ds = FileListDataset(list_path='D:\Works\datasets\data\Xray\\target.txt', path_prefix='',
                                 transform=val_tranform, filter=(lambda x: x in target_classes))
    g = list(target_test_ds.datas)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)
    
    #或者可以使用预训练好的模型
    backbone = resnet50(pretrained=True)
    classifier = ImageClassifier(backbone, train_source_dataset.num_classes).to(device)
    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)
    domain_discri1 = DomainDiscriminator1(in_feature=classifier.features_dim, hidden_size=1024).to(device)

    esem = Ensemble(classifier.features_dim, train_source_dataset.num_classes).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(esem.get_parameters() +classifier.get_parameters() + domain_discri1.get_parameters() + domain_discri.get_parameters(),#
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = StepwiseLR(optimizer, init_lr=args.lr, gamma=0.001, decay_rate=0.75)

    #预训练模型参数
    optimizer_pre = SGD(esem.get_parameters() + classifier.get_parameters() + domain_discri1.get_parameters(), 
                        args.lr/5, momentum=args.momentum,
                        weight_decay=args.weight_decay, nesterov=True)
    domain_adv = DomainAdversarialLoss(domain_discri, reduction='none').to(device)
    for epoch in range(30):
         pretrain(train_source_loader, classifier,esem, optimizer_pre, args)
    
    print("fininsh training pretrained model")

    # start training
    acc_resul = 0.0
    h_scoresu = 0.0
   #先加载模型的参数
    cls_pth = 'D:\\Works\\UDAML-master\\CMUM\\officemodel\\pre_classifier4143.pkl' 
    best_cls = copy.deepcopy(torch.load(cls_pth))
    classifier.load_state_dict(best_cls)
    esm_pth = 'D:\\Works\\UDAML-master\\CMUM\\officemodel\\pre_esem4143.pkl' 
    best_esm = copy.deepcopy(torch.load(esm_pth))
    esem.load_state_dict(best_esm)
    for epoch in range(args.epochs):
        print("The epoch is '{}'".format(epoch))
        # if epoch == 0:
        initc,labelset = evaluate_sourcecentroid(train_source_loader, classifier, esem, source_classes, args)
            #计算均值
        w_s = evaluate_source_common(val_loader, classifier, esem, source_classes, args,initc, g, epoch)#, domain_discri1
        
        target1_dataset = dataset(root=args.root, data_list_file='D:\\Works\\UDAML-master\\CMUM1 copy2.txt', filter_class=source_classes,
                            transform=val_tranform)
        target1_loader = DataLoader(target1_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                    drop_last=True)
        target1_iter = ForeverDataIterator(target1_loader)

        target2_dataset = dataset(root=args.root, data_list_file='D:\\Works\\UDAML-master\\CMUM1 copy3.txt', filter_class=target_classes,
                            transform=val_tranform)
        target2_loader = DataLoader(target2_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                    drop_last=True)
        target2_iter = ForeverDataIterator(target2_loader)

        f = open(r"D:\\Works\\UDAML-master\\CMUM1 copy5.txt", "r")
       
        new = [] #定义一个空列表，zhuan用来存储结果
        for line in f.readlines():
            temp1 = line.strip('\n') #去掉每行最内后的换行符'\n'
            new.append(temp1) #将上一步得到的列表添加到new中
        f.close()
        
        new_id = []
        for n in new:
            new_id.append(float(n))
        target3_dataset = dataset(root=args.root, data_list_file='D:\\Works\\UDAML-master\\CMUM1 weight.txt', filter_class=new_id,
                            transform=val_tranform)
        target3_loader = DataLoader(target3_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                    drop_last=True)
        target3_iter = ForeverDataIterator(target3_loader)

        train(train_source_iter, target3_iter, classifier, domain_adv, esem, optimizer,
              lr_scheduler, epoch,  args,val_loader,source_classes,target1_iter,target2_iter, w_s, labelset,domain_discri1)#source_class_weight, feat2, all_feat1,train_target_iter
       # evaluate on validation set
        acc,h_score= validate(val_loader, classifier, esem, source_classes, args,labelset) 
        print("<<<<<<<<<<")
        print(acc)
        print("<<<<<<<<<<")

        # remember best acc@1 and save checkpoint
        if acc > acc_resul:
            best_model = copy.deepcopy(classifier.state_dict())
            acc_resul = acc
            h_scoresu = h_score  
        print(acc_resul , 'bestacc')
        print(h_scoresu , 'besth_score')
    print("best_acc = {:3.3f}".format(acc_resul))
    print("h_scoresu = {:3.3f}".format(h_scoresu))

def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, domain_adv: DomainAdversarialLoss, esem, optimizer: SGD,
          lr_scheduler: StepwiseLR, epoch: int, args: argparse.Namespace, val_loader, source_class, 
          target1_iter:ForeverDataIterator, target2_iter:ForeverDataIterator, w_s, labelset,
          domain_discri1: DomainDiscriminator1):#source_class_weight,  feat2, all_feat1,
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_discri1.train()
    domain_adv.train()
    esem.train()
    end = time.time()
    pred_num = 0
    loss_num = 0.0
    count_loss_num = 0
    for i in range(args.iters_per_epoch):
        count_loss_num += 1
        lr_scheduler.step()
        w_s1 = [random.random()  for _ in range(32)]
        data_time.update(time.time() - end)

        x_s, labels_s = next(train_source_iter)
        x_t, weight = next(train_target_iter)

        xt1, labelt1 = next(target1_iter)
        xt2, labelt2 = next(target2_iter)

        x_s = x_s.to(device)
        ys1, fs1 = model(x_s)
        ys1= esem(fs1, index=1)
        ys1= nn.Softmax(dim=-1)(ys1)

        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        xt1 = xt1.to(device)
        labelt1 = labelt1.to(device)

        xt2 = xt2.to(device)
        labelt2 = labelt2.to(device)

        weight = weight.to(device)
        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)

        yt1, ft1 = model(xt1)
        y_1= esem(ft1, index=1)
        y_1 = nn.Softmax(dim=-1)(y_1)
        yt2, ft2 = model(xt2)
        d = domain_discri1(ft2)
        bal = 0
        for cls in range(args.n_share):
            bal += w_s[cls]
        with torch.no_grad():
            labels_s1 = np.array(labels_s.cpu())
            a = 0
            for i in labels_s1:
                w_s1[a] = w_s[int(i)]
                a = a+1
            w_s1 = torch.tensor(w_s1).to(device)
            # w_t = torch.tensor(weight).to(device)
        #加权的交叉熵损失
        cls_loss = F.cross_entropy(y_s, labels_s.long())
        #
        cls_loss1 = F.cross_entropy(yt1, labelt1.long())
        transfer_loss = domain_adv(f_s, f_t)#.to(device)
        
        # 计算IMloss
        entropy_loss = torch.mean(torch.sum(- y_1* torch.log(y_1 + 1e-10), dim=1, keepdim=True))
        msoftmax = y_1.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-10))
        entropy_loss -= gentropy_loss

        
        Tfeat_t = ft2.tolist()
        Tlable = labelt2
        Tfeat_t = np.array(Tfeat_t)
        Tlable = np.array(Tlable.cpu())
        if Tfeat_t != []:               
            Tloss_t = batch_all_triplet_loss(Tlable, Tfeat_t, 0.2, d, False)
        if Tfeat_t == []:
            Tloss_t = 0    
        Tloss =  0.3 * Tloss_t
        cls_loss1 = (bal/2)*cls_loss1
        
        # loss = 0.5 * cls_loss + entropy_loss + cls_loss1 + Tloss  +  transfer_loss
        # loss = 0.5 * cls_loss + entropy_loss + cls_loss1 + Tloss 
        loss = 0.5 * cls_loss +  entropy_loss + cls_loss1 + Tloss 
        loss_num += loss
        cls_acc = accuracy(y_s, labels_s)[0]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()
    print(loss_num / float(count_loss_num))
            
def validate(val_loader: DataLoader, model: ImageClassifier, esem, source_classes: list,
             args: argparse.Namespace,  labelset) -> float:
    model.eval()
    esem.eval()
    start_test = True
    all_labels = list()
    a = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            output, f = model(images)
            y_1 = esem(f, index=1)
            if(start_test):
                all_f = f.float().cpu()
                all_sm= y_1.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_f = torch.cat((all_f, f.float().cpu()), 0)
                all_sm = torch.cat((all_sm, y_1.float().cpu()), 0)      
                all_label = torch.cat((all_label, labels.float()), 0)     
            all_labels.extend(labels)
    if args.distance == 'cosine':
        all_f = torch.cat((all_f, torch.ones(all_f.size(0), 1)), 1)
        all_f = (all_f.t() / torch.norm(all_f, p=2, dim=1)).t()

    all_f = all_f.float().cpu().numpy()
    all_softmax = nn.Softmax(dim=1)(all_sm)  
    _, predict = torch.max(all_softmax, 1)
    K = all_sm.size(1)
    #分别进行总的特征计算
    aff = all_softmax.float().cpu().numpy()
    initc = aff.transpose().dot(all_f)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])

    h_dict = {}
    h_dict_true = {}
    thre_dis = {}
    val_filter = {}
    thre_filter_3= {}
    pred_num1 = []
    all_feat1 = [[random.random() for _ in range(2)]for _ in range(1031)]
    
    for cls in range(args.n_share):
        h_dict[cls] = [] 
        h_dict_true[cls] = [] 
        thre_dis[cls] = []
        val_filter[cls] = []
        thre_filter_3[cls] = []
            
    dd2 = cdist(all_f, initc[labelset], args.distance)
    pred_label = dd2.argmin(axis=1)
    pred_num1 = [dd2.min(axis =1 ),dd2.argmin(axis = 1 )]

    for cls in range(args.n_share):
        cls_filter = (pred_num1[1] == cls)
        list_loc = cls_filter.tolist()
        list_loc = [i for i,x in enumerate(list_loc) if x ==1 ]
        list_loc = torch.Tensor(list_loc)
        list_loc = list_loc.long()
        filtered_val = torch.gather(torch.tensor(pred_num1[0]), dim=0, index = list_loc)
        filtered_tru = torch.gather(torch.tensor(np.array(all_label.cpu())), dim=0, index = list_loc)
        val_filter[cls].append(filtered_val.cpu().data.numpy())
        h_dict_true[cls].append(filtered_tru.cpu().data.numpy())

    #计算临界值
    for cls in range(args.n_share):
        ents_np = np.concatenate(val_filter[cls], axis=0)
        h_ava_size = len(ents_np)
        ent_idxs = np.argsort(ents_np)
        thre_filter_3[cls] = ents_np[ent_idxs[int(len(ents_np)*0.8)]]
    for cls in range(args.n_share):
        for i in range(len(pred_label)):
            if(pred_label[i] == cls):
                if pred_num1[0][i] > thre_filter_3[cls] : #
                    pred_num1[1][i] = 2

    acct2 = np.sum(pred_num1[1] == torch.tensor(all_labels).float().numpy()) / len(all_f)
    #计算h_score和混淆矩阵
    for cls in range(args.n_total):
        h_dict[cls] = [] 
        h_dict_true[cls] = []

    for cls in range(args.n_total):
        cls_filter = (pred_num1[1] == cls)
        list_loc = cls_filter.tolist()
        list_loc = [i for i,x in enumerate(list_loc) if x ==1 ]
        list_loc = torch.Tensor(list_loc)
        list_loc = list_loc.long()
        filtered_val = torch.gather(torch.tensor(pred_num1[1]), dim=0, index = list_loc)
        filtered_tru = torch.gather(torch.tensor(np.array(all_label.cpu())), dim=0, index = list_loc)
        h_dict[cls].append(filtered_val.cpu().data.numpy())
        h_dict_true[cls].append(filtered_tru.cpu().data.numpy())

    #画混淆矩阵
    # cm = confusion_matrix(np.array(all_label.cpu()), np.array(pred_num1[1]))
    # plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')
    #计算common acc 和  open acc
    h_score = 0
    common_num = 0
    common_all = 0
    open_num = 0
    open_all = 0
    for cls in range(args.n_total):
        if(cls < args.n_share):
            common_all  += len(np.concatenate(h_dict[cls] , axis=0)) 
            common_num += np.sum(h_dict[cls] == torch.tensor(h_dict_true[cls]).float().numpy())
        else:
            open_all  = len(np.concatenate(h_dict[cls] , axis=0)) 
            open_num  = np.sum(h_dict[cls] == torch.tensor(h_dict_true[cls]).float().numpy())
    common_acc = common_num / common_all
    open_acc = open_num / open_all
    h_score = 2 * common_acc * open_acc / (common_acc + open_acc)

    
    return acct2,h_score #counters.mean_accuracy()

#计算源域数据的聚类中心
def evaluate_sourcecentroid(val_loader: DataLoader, model: ImageClassifier, esem, source_classes: list,
                           args: argparse.Namespace ):
    model.eval()
    esem.eval()
    all_output = list()
    cnt = 0
    start_test = True
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            _, f = model(images)
            y_1 = esem(f, index=1)
            #全部的特征
            if(start_test):
                all_f = f.float().cpu()
                all_sm= y_1.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_f = torch.cat((all_f, f.float().cpu()), 0)#按行拼接
                all_sm = torch.cat((all_sm, y_1.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    #计算聚类得到的标签损失,先对特征进行归一化
    if args.distance == 'cosine':
        all_f = torch.cat((all_f, torch.ones(all_f.size(0), 1)), 1)
        all_f = (all_f.t() / torch.norm(all_f, p=2, dim=1)).t()#torch.norm按行求2范数
    all_softmax = nn.Softmax(dim=1)(all_sm)  
    _, predict = torch.max(all_softmax, 1)
    K = all_sm.size(1)
    aff = np.eye(K)[all_label.int()]
    initc = aff.transpose().dot(all_f)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)                                                                                                                                                                                                   
    #如果对应的类别号是predict，那么转成one-hot的形式
    labelset = np.where(cls_count > args.thresholdd)
    labelset = labelset[0]
    return initc, labelset


def evaluate_source_common(val_loader: DataLoader, model: ImageClassifier, esem, source_classes: list,
                           args: argparse.Namespace, initc: np, g,epoch):#domain_discri1: DomainDiscriminator1
    temperature = 1
    model.eval()
    esem.eval()
    common = []
    target_private = []

    all_confidece = list()
    all_consistency = list()
    all_entropy = list()

    all_labels = list()
    all_output = list()
    all_pseudo_label = list()
    all_feat = list()
    a = 0
    source_weight = torch.zeros(len(source_classes)).to(device)
    cnt = 0
    start_test = True
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            outputl, f = model(images)
            output = F.softmax(outputl, -1) / temperature
            pseudo_label = torch.argmax(output, dim=1)
            yt_1= esem(f)
            all_labels.extend(labels)
            all_pseudo_label.extend(pseudo_label)
            for each_output in output:
                all_output.append(each_output)
            if(start_test):
                all_f = f.float().cpu()
                all_f1 = f.float().cpu()
                all_sm = yt_1.float().cpu()
                all_outputl= outputl.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_f = torch.cat((all_f, f.float().cpu()), 0)
                all_f1 = torch.cat((all_f1, f.float().cpu()), 0)
                all_sm = torch.cat((all_sm, yt_1.float().cpu()), 0)
                all_outputl = torch.cat((all_outputl, outputl.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)    
    all_feat = all_f
    _, predict = torch.max(all_sm, 1)
    #计算聚类得到的标签损失,先对特征进行归一化
    if args.distance == 'cosine':
        all_f = torch.cat((all_f, torch.ones(all_f.size(0), 1)), 1)
        all_f = (all_f.t() / torch.norm(all_f, p=2, dim=1)).t()
        all_f1 = torch.cat((all_f1, torch.ones(all_f1.size(0), 1)), 1)
        all_f1 = (all_f1.t() / torch.norm(all_f1, p=2, dim=1)).t()
    
    all_f = all_f.float().cpu().numpy()
    K = all_sm.size(1)
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.thresholdd)
    labelset = labelset[0]
    h_dict = {}
    thre_dis = {}
    val_filter = {}
    thre_filter_1= {}
    thre_filter_2= {}
    thre_filter_3= {}
    thre_filter_4= {}
    pred_num1 = []
    all_feat1 = [[random.random() for _ in range(2)]for _ in range(1031)]
    feat2 = [[random.random() for _ in range(2)]for _ in range(1031)]
    pred_n = [[random.random() for _ in range(2)]for _ in range(1031)]
    for cls in range(args.n_share):
        h_dict[cls] = [] 
        thre_dis[cls] = []
        val_filter[cls] = []
        thre_filter_1[cls] = []
        thre_filter_2[cls] = []
        thre_filter_3[cls] = []
        thre_filter_4[cls] = []
    if(epoch >= 0):
        dd = cdist(all_f, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
        pred_num = dd.min(axis = 1)
        acc1 = np.sum(pred_label == torch.tensor(all_labels).float().numpy()) / len(all_f)
        print(acc1,'无新类')
    
        for cls in range(args.n_share):
            cls_filter = (pred_label == cls)
            list_loc = cls_filter.tolist()
            list_loc = [i for i,x in enumerate(list_loc) if x ==1 ]
            list_loc = torch.Tensor(list_loc)
            list_loc = list_loc.long()
            filtered_avg = torch.gather(torch.tensor(pred_num), dim=0, index = list_loc)
            h_dict[cls].append(filtered_avg.cpu().data.numpy())
        h_true = calculate(h_dict,2)
        #将当前的临界值记下来
        #重新构建聚类中心
        for cls in range(args.n_share):
            ents_np = np.concatenate(h_dict[cls], axis=0)
            h_ava_size = len(ents_np)
            ent_idxs = np.argsort(ents_np)
            if(h_true[cls] == True):
                thre_dis[cls] = ents_np[ent_idxs[int(len(ents_np)*0.8)]]
            else:
                thre_dis[cls] = ents_np[ent_idxs[int(len(ents_np)*0.2)]]
        ###按照值进行过滤的操作
        for cls in range(len(pred_label)-1):
            if(cls == len(pred_label)):
                break
            if(h_true[pred_label[cls]] == True):
                if(pred_num[cls] >= thre_dis[pred_label[cls]]):
                    all_f = np.delete(all_f,cls,0)
                    all_outputl = np.delete(all_outputl,cls,0)
                    all_sm = np.delete(all_sm,cls,0)
                    pred_label = np.delete(pred_label,cls,0)
                    pred_num = np.delete(pred_num,cls,0)
                    cls = cls - 1
            else:
                if(pred_num[cls] <= thre_dis[pred_label[cls]]):
                    all_f = np.delete(all_f,cls,0)
                    pred_label = np.delete(pred_label,cls,0)
                    all_outputl = np.delete(all_outputl,cls,0)
                    all_sm = np.delete(all_sm,cls,0)
                    pred_num = np.delete(pred_num,cls,0)
                    cls = cls - 1
        all_softmax = nn.Softmax(dim=1)(all_sm)
        aff =all_softmax.float().cpu().numpy()
        initc = aff.transpose().dot(all_f)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    else:    
       #计算聚类中心
        all_softmax = nn.Softmax(dim=1)(all_sm)
        aff = all_softmax.float().cpu().numpy()
        initc = aff.transpose().dot(all_f)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    w_s = [0.0,0.0]
    w_s0 = 0.0
    w_s1 = 0.0
    n_s0 = 0.0
    n_s1 = 0.0
    dd1 = cdist(all_f1, initc[labelset], args.distance)
    pred_label1 = dd1.argmin(axis=1)#argmin表示使目标函数f(x)取最小值时的变量值，axis = 1，表示对行操作
    pred_label1 = labelset[pred_label1]

    acc = np.sum(pred_label1 == torch.tensor(all_labels).float().numpy()) / len(all_f1)
    pred_num1 = [dd1.min(axis =1),dd1.argmin(axis = 1 )]

    for cls in range(args.n_share):
        cls_filter = (pred_num1[1] == cls)
        list_loc = cls_filter.tolist()
        list_loc = [i for i,x in enumerate(list_loc) if x ==1 ]
        list_loc = torch.Tensor(list_loc)
        list_loc = list_loc.long()
        filtered_val = torch.gather(torch.tensor(pred_num1[0]), dim=0, index = list_loc)
        val_filter[cls].append(filtered_val.cpu().data.numpy())
    all_feat = np.array(all_feat)
  
    #计算临界值
    for cls in range(args.n_share):
        ents_np = np.concatenate(val_filter[cls], axis=0)
        h_ava_size = len(ents_np)
        ent_idxs = np.argsort(ents_np)
        thre_filter_2[cls] = ents_np[ent_idxs[int(len(ents_np)*0.5)]]
        thre_filter_3[cls] = ents_np[ent_idxs[int(len(ents_np)*0.9)]]
            
    for i in range(len(pred_label1)):
        if(pred_label1[i] == 0):
            if pred_num1[0][i] > thre_filter_3[0]:
                pred_num1[1][i] = 2
        elif(pred_label1[i] == 1):
            if(pred_num1[0][i] > thre_filter_3[1]):
                pred_num1[1][i] = 2
  
    #计算源域的权重
    for i in range(len(pred_label1)):
        if(pred_label1[i] == 0):
            if pred_num1[0][i] <= thre_filter_3[0] :#and d_all[i] < 0.7
                w_s0 += (1 - pred_num1[0][i])
                n_s0 += 1
        if(pred_label1[i] == 1):
            if(pred_num1[0][i] <= thre_filter_3[1]):
                w_s1 += (1 - pred_num1[0][i])
                n_s1 += 1
    #计算源域和目标域的权重
    w_s[0] = w_s0/(n_s0 + 1e-10)
    w_s[1] = w_s1/(n_s1 + 1e-10)
    print(w_s,'w_s')

    s_w_0 = 0.0
    s_c_0 = 0.0 
    s_w_1 = 0.0
    s_c_1 = 0.0
    
    a = 0
    b = 0
    c = 0
    d = 0
    #计算目标域的权重和所有可信的样本
    for i in range(len(pred_label1)):
        if(pred_label1[i] == 0):
            if(pred_num1[0][i] <= thre_filter_2[0]):
                pred_num1[0][i] = (1- pred_num1[0][i])
                pred_n[c][0] = g[i]
                pred_n[c][1] = pred_num1[0][i] 
                c = c+1
                s_w_0 += pred_num1[0][i] 
                s_c_0 += 1  
             
                all_feat1[a][0] = g[i]
                all_feat1[a][1] = pred_num1[1][i]
                a = a+1
                feat2[b][0] = g[i]
                feat2[b][1] = 0
                b = b+1
            else:
                pred_num1[0][i] = 0#(1 - pred_num1[0][i])/(4*(1 + pred_num1[0][i]))
                pred_n[c][0] = g[i]
                pred_n[c][1] = 0#(1 - pred_num1[0][i])/(4*(1 + pred_num1[0][i]))
                c = c+1

        elif(pred_label1[i] == 1):
            if(pred_num1[0][i] <= thre_filter_2[1]):
                pred_num1[0][i] = 1- pred_num1[0][i]
                s_c_1 += 1
                pred_n[c][0] = g[i]
                pred_n[c][1] = pred_num1[0][i]
                c = c+1  
                all_feat1[a][0] = g[i]
                all_feat1[a][1] = pred_num1[1][i]
                a = a+1
                feat2[b][0] = g[i]
                feat2[b][1] = 0
                b = b+1
            else:
                pred_num1[0][i] = 0#(1 - pred_num1[0][i])/(4*(1 + pred_num1[0][i]))
                pred_n[c][0] = g[i]
                pred_n[c][1] = 0#(1 - pred_num1[0][i])/(4*(1 + pred_num1[0][i]))
                c = c+1
    for i in range(len(pred_label1)):
        if(pred_num1[1][i] == 2):
            feat2[b][0] = g[i]
            feat2[b][1] = pred_num1[1][i]
            b = b+1


        if all_labels[i] in source_classes:
            common.append(pred_num1[0][i])
        else:
            target_private.append(pred_num1[0][i]) 
    with open("D:\\Works\\UDAML-master\\CMUM1 copy2.txt", "w") as output:
        #i = 0
        for i in range(a):
            s = str(all_feat1[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
            s = s.replace("'",'').replace(',','') +'\n'  #去除单引号，逗号，每行末尾追加换行符  
            output.write(s)
    with open("D:\\Works\\UDAML-master\\CMUM1 copy3.txt", "w") as output:
        #i = 0
        for i in range(b):
            s = str(feat2[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
            s = s.replace("'",'').replace(',','') +'\n'  #去除单引号，逗号，每行末尾追加换行符  
            output.write(s)
    with open("D:\\Works\\UDAML-master\\CMUM1 weight.txt", "w") as output:
        #i = 0
        for i in range(c):
            s = str(pred_n[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
            s = s.replace("'",'').replace(',','') +'\n'  #去除单引号，逗号，每行末尾追加换行符  
            output.write(s)
    with open("D:\\Works\\UDAML-master\\CMUM1 copy5.txt", "w") as output:
        #i = 0
        for i in range(c):
            s = str(pred_n[i][1]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
            s = s.replace("'",'').replace(',','') +'\n'  #去除单引号，逗号，每行末尾追加换行符  
            output.write(s)
    source_weight = norm(source_weight / cnt)
    print('---source_weight---')
    return w_s
#计算均值,计算当前处于平均水平之上的数值多还是之下的数值多
def calculate(h_dict,cls_name):   
    h_true = [] 
    for cls in range(cls_name):
        sub_total_size = 0
        sub_total_distance = 0
        ents_np = np.concatenate(h_dict[cls], axis=0)
        avg_distance = np.mean(ents_np)
        cls_filter = (ents_np> avg_distance)
        list_loc = cls_filter.tolist()
        list_loc = [i for i,x in enumerate(list_loc) if x ==1 ] 
        if len(ents_np)*0.5 >= len(list_loc):
            h_true.append(True)#中位值大，取q\
        else:
            h_true.append(False)#平均值大，取前
    return h_true

            

def pretrain( train_source_iter, model, esem, optimizer, args):
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [losses, cls_accs],
        prefix="Esem: [{}-{}]".format(0, 0))

    model.train()
    esem.train()
    best_acc = 0.0
    esem_loader = ForeverDataIterator(train_source_iter)
    for i in range(args.iters_per_epoch):
        x_s1, labels_s1 = next(esem_loader)
        x_s1 = x_s1.to(device)
        labels_s1 = labels_s1.to(device)
        y_s1, f_s1 = model(x_s1)
        y_s1 = esem(f_s1, index=1)
        loss1 = F.cross_entropy(y_s1, labels_s1.long())
        loss1 = loss1
    
        cls_acc = accuracy(y_s1, labels_s1)[0]
        cls_accs.update(cls_acc.item(), x_s1.size(0))

        losses.update(loss1.item(), x_s1.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        if i % (args.print_freq) == 0:
            progress.display(i)

        #将模型存一下
    if cls_accs.avg >= best_acc:
        best_acc = cls_accs.avg
        best_classifier = copy.deepcopy(model.state_dict()) 
        best_esem = copy.deepcopy(esem.state_dict()) 
        with open(join(args.pretrain_model_path, 'pre_classifier4143.pkl'), 'wb') as f:
            torch.save(best_classifier, f)
        with open(join(args.pretrain_model_path, 'pre_esem4143.pkl'), 'wb') as f:
            torch.save(best_esem, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('-root', metavar='DIR',default='D:\Works\datasets\data\Xray',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Xray')
    parser.add_argument('-s', '--source', help='source domain(s)' , default='D:\Works\datasets\data\Xray\\source.txt')
    parser.add_argument('-t', '--target', help='target domain(s)' , default='D:\Works\datasets\data\Xray\\target.txt')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=1234, type=int,#None
                        help='seed for initializing training. ')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('-i', '--iters-per-epoch', default=33, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--n_share', type=int, default=2, help=" ")
    parser.add_argument('--n_source_private', type=int, default=0, help=" ")
    parser.add_argument('--n_total', type=int, default=3, help="")
    parser.add_argument('--threshold', type=float, default=0.5, help=" ")
    parser.add_argument('--sourcepath', type=str, default='D:\Works\datasets\data\Xray\\source.txt', help="")
    parser.add_argument('--targetpath', type=str, default='D:\Works\datasets\data\Xray\\target.txt', help="")
    parser.add_argument('--pretrain_model_path', type=str, default='D:\\Works\\UDAML-master\\CMUM\\officemodel', help="")
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--thresholdd', type=int, default=0)

    args = parser.parse_args()
    main(args)