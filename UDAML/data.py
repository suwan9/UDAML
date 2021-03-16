from config import *
from easydl import *
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler


import os
import numpy as np
import tensorpack
import time
import random
import numbers
#from scipy.misc import imresize
from PIL import Image
import numpy as np

from imageio import imread
import tensorlayer as tl
from six.moves import cPickle
from utilities import *

import warnings

'''
assume classes across domains are the same.
[0 1 ..................................................................... N - 1]
|----common classes --||----source private classes --||----target private classes --|
'''
a, b, c = args.data.dataset.n_share, args.data.dataset.n_source_private, args.data.dataset.n_total
c = c - a - b
common_classes = [i for i in range(a)]
source_private_classes = [i + a for i in range(b)]
target_private_classes = [i + a + b for i in range(c)]

source_classes = common_classes + source_private_classes
target_classes = common_classes + target_private_classes

train_transform = Compose([
    Resize(256),
    RandomCrop(224),
    RandomHorizontalFlip(),
    ToTensor()
])

test_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor()
])

source_train_ds = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                            transform=train_transform, filter=(lambda x: x in source_classes))
source_test_ds = FileListDataset(list_path=source_file,path_prefix=dataset.prefixes[args.data.dataset.source],
                            transform=test_transform, filter=(lambda x: x in source_classes))
target_train_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                            transform=train_transform, filter=(lambda x: x in target_classes))
target_test_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                            transform=test_transform, filter=(lambda x: x in target_classes))

classes = source_train_ds.labels
freq = Counter(classes)
class_weight = {x : 1.0 / freq[x] if args.data.dataloader.class_balance else 1.0 for x in freq}

source_weights = [class_weight[x] for x in source_train_ds.labels]
sampler = WeightedRandomSampler(source_weights, len(source_train_ds.labels))

source_train_dl = DataLoader(dataset=source_train_ds, batch_size=args.data.dataloader.batch_size,
                             sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
source_test_dl = DataLoader(dataset=source_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                             num_workers=1, drop_last=False)
target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.data.dataloader.batch_size,shuffle=True,
                             num_workers=args.data.dataloader.data_workers, drop_last=True)
target_test_dl = DataLoader(dataset=target_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                             num_workers=1, drop_last=False)


#———————————————————————————————————————分割线————————————————————————————————————

warnings.filterwarnings('ignore', message='.*', category=Warning)
class CustomDataLoader(object):#自定义数据加载器
    def __init__(self, dataset, batch_size, num_threads=8,remainder=None):
        self.ds0 = dataset
        self.batch_size = batch_size
        self.num_threads = num_threads
        
        if not remainder:
            try:
                is_train = self.ds0.is_train
                remainder = False if is_train else True 
                # if is_train, there is no need to set reminder 如果是训练，则无需设置提醒
            except Exception as e:
                # self.ds0 maybe doesn't have is_train attribute, then it has no test mode, set remainder = False
                # self.ds0可能没有is_train属性，那么它没有测试模式，设置remainment=False
                remainder = False
        
        # use_list=False, for each in data point, add a batch dimension (return in numpy array)
        self.ds1 = tensorpack.dataflow.BatchData(self.ds0, self.batch_size,remainder=remainder, use_list=False,) 
        #将数据点成批堆叠。它生成的数据点与ds相同数量的组件，但每个组件都有一个新的额外维度，即批处理大小。
        #批处理可以是原始组件的列表，也可以是原始组件的numpy数组。
        
        # use 1 thread in test to avoid randomness (test should be deterministic)
        self.ds2 = tensorpack.dataflow.PrefetchDataZMQ(self.ds1, num_proc=self.num_threads if not remainder else 1)
        #在>=1进程中运行数据流，使用ZeroMQ进行通信。它将分叉以下调用过程：方法：重置状态（），
        #并通过ZeroMQ IPC管道从每个进程的给定数据流中收集数据点。这通常比：类：MultiProcessRunner。

        # required by tensorlayer package
        self.ds2.reset_state()
    
    def generator(self):
        return self.ds2.get_data()

class BaseDataset(tensorpack.dataflow.RNGDataFlow):#基本数据集
    def __init__(self, is_train=True, skip_pred=None, transform=None, sample_weight=None):
        self.is_train = is_train
        self.skip_pred = skip_pred or (lambda data, label, is_train : False)
        self.transform = transform or (lambda data, label, is_train : (data, label))
        self.sample_weight = sample_weight or (lambda data, label : 1.0)

        self.datas = []
        self.labels = []

        self._fill_data()

        self._post_init()

    def _fill_data(self):
        raise NotImplementedError("not implemented!")
        #如果这个方法没有被子类重写，但是调用了，就会报错。

    def _post_init(self):
        tmp = [[data, label]  for (data, label) in zip(self.datas, self.labels) if not self.skip_pred(data, label, self.is_train) ]
        self.datas = [x[0] for x in tmp]
        self.labels = [x[1] for x in tmp]

        if callable(self.sample_weight):
            # callable返回对象是否可调用（即某种函数）。请注意，类是可调用的，具有调用函数。
            self._weight = [self.sample_weight(x, y) for (x, y) in zip(self.datas, self.labels)]
        else:
            self._weight = self.sample_weight
        self._weight = np.asarray(self._weight, dtype=np.float32).reshape(-1)
        assert len(self._weight) == len(self.datas), 'dimension not match!'
        #尺寸不匹配
        self._weight = self._weight / np.sum(self._weight)

    def size(self):
        return len(self.datas)

    def _get_one_data(self, data, label):
        raise NotImplementedError("not implemented!")

    def get_data(self):
        size = self.size()
        ids = list(range(size))
        for _ in range(size):
            id = np.random.choice(ids, p=self._weight) if self.is_train else _
            #np.random.choice处理数据时经常需要从数组中随机抽取元素
            #从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
            #replace:True表示可以取相同数字，False表示不可以取相同数字
            #数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
            data, label = self._get_one_data(self.datas[id], self.labels[id])
            data, label = self.transform(data, label, self.is_train)
            yield np.asarray(data), np.asarray([label]) if isinstance(label, numbers.Number) else label
            #np.asarray将输入转为矩阵格式
            #isinstance() 函数来判断一个对象是否是一个已知的类型
            #python有yield表达式，它只能用于定义生成器函数，生成器可以控制函数的执行，
            #函数可以再生成器语句出暂停执行，当前使用的变量，堆栈等都会保留，直到下次使用生成器方法。
    
class BaseImageDataset(BaseDataset):#基本图像数据集
    def __init__(self, imsize=224, is_train=True, skip_pred=None, transform=None, sample_weight=None):
        self.imsize = imsize
        super(BaseImageDataset, self).__init__(is_train, skip_pred, transform, sample_weight=sample_weight)

    def _get_one_data(self, data, label):
        im = imread(data, pilmode='RGB') #图像读取
        if self.imsize:
            '''
            norm_map = imresize(raw_hm, (height, width))
            #换成
            norm_map = np.array(Image.fromarray(raw_hm).resize( (height, width)))
            '''
            #im = imresize(im, (self.imsize, self.imsize))
            im = np.array(Image.fromarray(im).resize( (self.imsize, self.imsize)))
            #输入固定大小调整shape
        return im, label


def one_hot(n_class, index):
    tmp = np.zeros((n_class,), dtype=np.float32)
    tmp[index] = 1.0
    return tmp

class FileListDataset1(BaseImageDataset):#文件列表数据集
    
    def __init__(self, list_path, path_prefix='', imsize=224, is_train=True, skip_pred=None, transform=None, sample_weight=None):
        self.list_path = list_path
        self.path_prefix = path_prefix

        super(FileListDataset1, self).__init__(imsize=imsize, is_train=is_train, skip_pred=skip_pred, transform=transform, sample_weight=sample_weight)

    def _fill_data(self):
        with open(self.list_path, 'r') as f:
            data = [[line.split()[0], line.split()[1]] for line in f.readlines() if line.strip()] 
            # avoid empty lines 避免空行
            # split() 通过指定分隔符对字符串进行切片
            # readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表
            # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。

            self.datas = [os.path.join(self.path_prefix, x[0]) for x in data]
            #os.path.join()函数：连接两个或更多的路径名组件
            try:
                self.labels = [int(x[1]) for x in data]
            except ValueError as e:
                print('invalid label number, maybe there is space in image path?')
                #标签号无效，可能是图像路径中有空格
                raise e