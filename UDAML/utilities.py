import numpy as np
import tensorflow as tf
import tensorlayer as tl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
import os
from collections import Counter
import matplotlib.pyplot as plt


class Accumulator(dict): #累加器
    def __init__(self, name_or_names, accumulate_fn=np.concatenate):
        super(Accumulator, self).__init__()
        self.names = [name_or_names] if isinstance(name_or_names, str) else name_or_names
        self.accumulate_fn = accumulate_fn
        for name in self.names:
            self.__setitem__(name, [])
            #__setitem__(self,key,value)：该方法应该按一定的方式存储和key相关的value。

    def updateData(self, scope):
        for name in self.names:
            self.__getitem__(name).append(scope[name])
            # __getitem__(name)获取数据
            # append() 方法用于在列表末尾添加新的对象。

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb:
            print(exc_tb)
            return False

        for name in self.names:
            self.__setitem__(name, self.accumulate_fn(self.__getitem__(name)))
            #accumulate函数的功能是对传进来的iterable对象逐个进行某个操作（默认是累加，
            # 如果传了某个fun就是应用此fun

        return True
		
class TrainingModeManager: #训练模式管理器
    def __init__(self, nets, train=False):
        self.nets = nets 
        self.modes = [net.training for net in nets]
        self.train = train
    def __enter__(self):
        for net in self.nets:
            net.train(self.train)
    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for (mode, net) in zip(self.modes, self.nets):
            net.train(mode)
        self.nets = None 
        # release reference, to avoid imexplicit reference 释放引用，以避免imexplicit引用
        if exceptionTraceback: # 异常回溯
            print(exceptionTraceback)
            return False
        return True
		
def clear_output(): # 清除输出
    def clear():
        return
    try:
        from IPython.display import clear_output as clear
    except ImportError as e:
        pass
    import os
    def cls():
        os.system('cls' if os.name == 'nt' else 'clear')
        # system函数可以将字符串转化成命令在服务器上运行

    clear()
    cls()

def addkey(diction, key, global_vars): # 地址键
    diction[key] = global_vars[key]

def track_scalars(logger, names, global_vars): # 跟踪标量
    values = {}
    for name in names:
        addkey(values, name, global_vars)
    for k in values:
        values[k] = variable_to_numpy(values[k])
    for k, v in values.items():
        logger.log_scalar(k, v)
    print(values)

def variable_to_numpy(x): #variable变为numpy
    ans = x.cpu().data.numpy()
    # numpy()将Tensor变量转换为ndarray变量，
    if torch.numel(x) == 1:
        # numel()函数：返回数组中元素的个数
        return float(np.sum(ans))
    return ans

def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    #反向调度程序
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))
    # ** 在python里面表示幂运算

def aToBSheduler(step, A, B, gamma=10, max_iter=10000): # a到b调度程序
    ans = A + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    #numpy.exp()：返回e的幂次方
    return float(ans)

def one_hot1(n_class, index):
    tmp = np.zeros((n_class,), dtype=np.float32)
    tmp[index] = 1.0
    return tmp

class OptimWithSheduler: # 优化调度程序
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']
    def zero_grad(self):
        self.optimizer.zero_grad()
        #optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
    def step(self):
        for g in self.optimizer.param_groups: 
            # optimizer.param_groups： 是长度为2的list，其中的元素是2个字典；
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr = g['initial_lr'])
        self.optimizer.step()  
        # optimizer.step()通常用在每个mini-batch之中，而scheduler.step()通常用在epoch里面,但是不绝对，
        # 可以根据具体的需求来做。只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整。
        self.global_step += 1
		
class OptimizerManager: # 优化器管理器
    def __init__(self, optims):
        self.optims = optims #if isinstance(optims, Iterable) else [optims]
    def __enter__(self):
        for op in self.optims:
            op.zero_grad()
    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None 
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True

def setGPU(i):
    global os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(i)
    gpus = [x.strip() for x in (str(i)).split(',')]
    NGPU = len(gpus)
    print('gpu(s) to be used: %s'%str(gpus))
    return NGPU

class Logger(object): # 记录器
    def __init__(self, log_dir, clear=False):
        if clear:
            os.system('rm %s -r'%log_dir)
        tl.files.exists_or_mkdir(log_dir)
        self.writer = tf.summary.create_file_writer(log_dir)
        self.step = 0
        self.log_dir = log_dir

    def log_scalar(self, tag, value, step = None):
        if not step:
            step = self.step
        tf.summary.scalar(tag,value,step=step)
        #将所有summary全部保存到磁盘，以便tensorboard显示
        #self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_images(self, tag, images, step = None):# 日志图像
        if not step:
            step = self.step
        
        im_summaries = []
        for nr, img in enumerate(images):
            s = StringIO()
            
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
                #扩展数组的形状。插入新的轴，该轴将出现在扩展数组形状的轴位置上。
                #axis：int。在扩展轴中放置新轴的位置。
            
            if img.shape[-1] == 1:
                img = np.tile(img, [1, 1, 3])
                # tile(A，rep)
                # 功能：重复A的各个维度
                # 参数类型：
                # - A: Array类的都可以
                # - rep：A沿着各个维度重复的次数

            img = to_rgb_np(img)
            plt.imsave(s, img, format = 'png')

            img_sum = tf.Summary.Image(encoded_image_string = s.getvalue(),
                                       height = img.shape[0],
                                       width = img.shape[1])
            # 用来输出Summary的图像。 输出Summary带有图像的协议缓冲区。
            im_summaries.append(tf.Summary.Value(tag = '%s/%d' % (tag, nr),
                                                 image = img_sum))
            # append（）函数用于合并两个数组。 该函数返回一个新数组，原始数组保持不变。
        summary = tf.Summary(value = im_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_histogram(self, tag, values, step = None, bins = 1000): # 对数直方图
        if not step:
            step = self.step
        values = np.array(values)
        counts, bin_edges = np.histogram(values, bins=bins) 
        # 直方统计图
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)
            
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_bar(self, tag, values, xs = None, step = None): #日志栏
        if not step:
            step = self.step

        values = np.asarray(values).flatten()
        if not xs:
            axises = list(range(len(values)))
        else:
            axises = xs
        hist = tf.HistogramProto()
        hist.min = float(min(axises))
        hist.max = float(max(axises))
        hist.num = sum(values)
        hist.sum = sum([y * x for (x, y) in zip(axises, values)])
        hist.sum_squares = sum([y * (x ** 2) for (x, y) in zip(axises, values)])

        for edge in axises:
            hist.bucket_limit.append(edge - 1e-10)
            hist.bucket_limit.append(edge + 1e-10)
        for c in values:
            hist.bucket.append(0)
            hist.bucket.append(c)

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, self.step)
        self.writer.flush()

class AccuracyCounter1: # 准确度
    def __init__(self):
        self.Ncorrect = 0.0
        self.Ntotal = 0.0
        
    def addOntBatch(self, predict, label): #添加批处理
        assert predict.shape == label.shape
        correct_prediction = np.equal(np.argmax(predict, 1), np.argmax(label, 1))
        # 如果使用两个类似数组的对象不相等，我们可以得到断言错误 
        # argmax返回的是最大数的索引
        Ncorrect = np.sum(correct_prediction.astype(np.float32))
        # astype 修改数据类型
        Ntotal = len(label)
        self.Ncorrect += Ncorrect
        self.Ntotal += Ntotal
        return Ncorrect / Ntotal
    
    def reportAccuracy(self): # 报告准确性
        return np.asarray(self.Ncorrect, dtype=float) / np.asarray(self.Ntotal, dtype=float)
        #np.asarray将输入转为矩阵格式

def CrossEntropyLoss(label, predict_prob, class_level_weight = None, instance_level_weight = None, epsilon = 1e-12):
    # 交叉熵损失
    N, C = label.size()
    N_, C_ = predict_prob.size()
    #print('%d,%d',N, C)
    #print('%d,%d',N_, C_)
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)

def BCELossForMultiClassification(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon = 1e-12):
    # 多分类的BCE损失
    N, C = label.size()
    N_, C_ = predict_prob.size()
    print('%d,%d',N, C)
    print('%d,%d',N_, C_)
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    bce = -label * torch.log(predict_prob + epsilon) - (1.0 - label) * torch.log(1.0 - predict_prob + epsilon)
    return torch.sum(instance_level_weight * bce * class_level_weight) / float(N)
	
def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon= 1e-20):
    # 熵损失
    N, C = predict_prob.size()
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob*torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)

def plot_confusion_matrix(cm, true_classes,pred_classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    # 绘图混淆矩阵
    import itertools
    pred_classes = pred_classes or true_classes
    if normalize:
        cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    true_tick_marks = np.arange(len(true_classes))
    plt.yticks(true_classes, true_classes)
    pred_tick_marks = np.arange(len(pred_classes))
    plt.xticks(pred_tick_marks, pred_classes, rotation=45)


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def extended_confusion_matrix(y_true, y_pred, true_labels, pred_labels):# true_labels=None, pred_labels=None):
    # 扩展混淆矩阵
    if not true_labels:
        true_labels = sorted(list(set(list(y_true)))) 
        #sorted() 函数对所有可迭代的对象进行排序操作。
    true_label_to_id = {x : i for (i, x) in enumerate(true_labels)}
    if not pred_labels:
        pred_labels = true_labels
    pred_label_to_id = {x : i for (i, x) in enumerate(pred_labels)}
    confusion_matrix = np.zeros([len(true_labels),len(pred_labels)])
    for (true, pred) in zip(y_true, y_pred):
        # print('%d,%d',true, pred)
        confusion_matrix[true_label_to_id[true]][pred_label_to_id[pred]] += 1.0
    return confusion_matrix