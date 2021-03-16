from config import *
from easydl import *
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler
from lib import *
import numpy as np

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
    RandomHorizontalFlip(),
    RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.2, resample=Image.BICUBIC,
                 fillcolor=(255, 255, 255)),
    CenterCrop(224),
    RandomGrayscale(p=0.1),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_transform2 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    MyRandomPerspective(),
    FiveCrop(224),
    Lambda(lambda crops: crops[0]),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_transform3 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.2, resample=Image.BICUBIC,
                 fillcolor=(255, 255, 255)),
    FiveCrop(224),
    Lambda(lambda crops: crops[1]),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_transform4 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1, resample=Image.BICUBIC,
                 fillcolor=(255, 255, 255)),
    MyRandomPerspective(),
    FiveCrop(224),
    Lambda(lambda crops: crops[2]),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_transform5 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    MyRandomPerspective(),
    FiveCrop(224),
    Lambda(lambda crops: crops[3]),
    RandomGrayscale(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

test_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

source_train_ds = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                  transform=train_transform, filter=(lambda x: x in source_classes))
source_train_ds2 = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                   transform=train_transform2, filter=(lambda x: x in source_classes))
source_train_ds3 = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                   transform=train_transform3, filter=(lambda x: x in source_classes))
source_train_ds4 = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                   transform=train_transform4, filter=(lambda x: x in source_classes))
source_train_ds5 = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                   transform=train_transform5, filter=(lambda x: x in source_classes))
source_test_ds = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                 transform=test_transform, filter=(lambda x: x in source_classes))
target_train_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                  transform=train_transform, filter=(lambda x: x in target_classes))
target_test_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                 transform=test_transform, filter=(lambda x: x in target_classes))

classes = source_train_ds.labels
freq = Counter(classes)
class_weight = {x: 1.0 / freq[x] if args.data.dataloader.class_balance else 1.0 for x in freq}

source_weights = [class_weight[x] for x in source_train_ds.labels]

#sampler = WeightedRandomSampler(source_weights, len(source_train_ds.labels))
sampler = WeightedRandomSampler(source_weights, 1325)
#print(source_weights, len(source_train_ds.labels), '8888')
#print(list(sampler),np.size(list(sampler)))
#input()


source_train_dl = DataLoader(dataset=source_train_ds, batch_size=args.data.dataloader.batch_size,
                             sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
source_train_dl2 = DataLoader(dataset=source_train_ds2, batch_size=args.data.dataloader.batch_size,
                              sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
source_train_dl3 = DataLoader(dataset=source_train_ds3, batch_size=args.data.dataloader.batch_size,
                              sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
source_train_dl4 = DataLoader(dataset=source_train_ds4, batch_size=args.data.dataloader.batch_size,
                              sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
source_train_dl5 = DataLoader(dataset=source_train_ds5, batch_size=args.data.dataloader.batch_size,
                              sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)

source_test_dl = DataLoader(dataset=source_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                            num_workers=1, drop_last=False)
target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                             num_workers=args.data.dataloader.data_workers, drop_last= True)
target_test_dl = DataLoader(dataset=target_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                            num_workers=1, drop_last=False)


'''
dataset(Dataset): 传入的数据集
    batch_size(int, optional): 每个batch有多少个样本
shuffle(bool, optional): 在每个epoch开始的时候，对数据进行重新排序
sampler(Sampler, optional): 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
    batch_sampler(Sampler, optional): 与sampler类似，但是一次只返回一个batch的indices（索引），需要注意的是，一旦指定了这个参数，那么batch_size,shuffle,sampler,drop_last就不能再制定了（互斥——Mutually exclusive）
    num_workers (int, optional): 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
    collate_fn (callable, optional): 将一个list的sample组成一个mini-batch的函数
    pin_memory (bool, optional)： 如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中.

drop_last (bool, optional): 如果设置为True：这个是对最后的未完成的batch来说的，比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…
    如果为False（默认），那么会继续正常执行，只是最后的batch_size会小一点。

    timeout(numeric, optional): 如果是正数，表明等待从worker进程中收集一个batch等待的时间，若超出设定的时间还没有收集到，那就不收集这个内容了。这个numeric应总是大于等于0。默认为0
    worker_init_fn (callable, optional): 每个worker初始化函数 If not None, this will be called on each
    worker subprocess with the worker id (an int in [0, num_workers - 1]) as
    input, after seeding and before data loading. (default: None) 
'''