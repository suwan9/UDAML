from data import *
from net import *
from lib import *
import datetime
from tqdm import tqdm
if is_in_notebook():
    from tqdm import tqdm_notebook as tqdm
from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import operator
from os import listdir

from eval import batch_hard_triplet_loss
from eval import batch_all_triplet_loss

cudnn.benchmark = True
cudnn.deterministic = True

seed_everything()

if args.misc.gpus < 1:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    gpu_ids = []
    output_device = torch.device('cpu')
else:
    # gpu_ids = select_GPUs(args.misc.gpus)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    gpu_ids = [0]
    output_device = gpu_ids[0]

now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

log_dir = f'{args.log.root_dir}/{now}'

logger = SummaryWriter(log_dir)

'''
with open(join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(save_config))

f = open(r"txt1/1.txt", "r")
#with open("txt1/1.txt", "r") as f:       #文件bai为du123.txt
#sourceInLines= f.readlines()    #按行读出文件zhidao
#f.close()
new = [] #定义一个空列表，zhuan用来存储结果
for line in f.readlines():
    temp1 = line.strip('\n') #去掉每行最内后的换行符'\n'
    temp2 = temp1.split(',') #以','为标志，将每容行分割成列表
    new.append(temp2) #将上一步得到的列表添加到new中
f.close()
#print(new)
#numbers = list(map(int, new))

#n = 0
#j = 0
new_id = []
new_label = []

for n in new:
    #print(n)
    new_id.append(int(n[0]))
    new_label.append(int(n[1]))
    #print(n)
#new_id = new_id
#new_id = np.array(new_id)
#new_label = np.array(new_label)

new_t = [[random.random() for _ in range(2)]for _ in range(np.size(new_id))]
i = 0
for i in range(np.size(new_id)):
    new_t[i][0] = new_id[i]
    new_t[i][1] = new_label[i]
    i = i + 1
#print(new_id,'11111')
#print(new_label,'77777')
#print(new_t,'22222')
'''

model_dict = {
    'resnet50': ResNet50Fc,
    'vgg16': VGG16Fc
}


class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.discriminator = AdversarialNetwork(256)
        self.discriminator_separate = AdversarialNetwork(256)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        d_0 = self.discriminator_separate(_)
        return y, d, d_0


totalNet = TotalNet()

feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=gpu_ids, output_device=output_device).train(True)
classifier = nn.DataParallel(totalNet.classifier, device_ids=gpu_ids, output_device=output_device).train(True)
discriminator = nn.DataParallel(totalNet.discriminator, device_ids=gpu_ids, output_device=output_device).train(True)
discriminator_separate = nn.DataParallel(totalNet.discriminator_separate, device_ids=gpu_ids, output_device=output_device).train(True)

if args.test.test_only:
    assert os.path.exists(args.test.resume_file)
    data = torch.load(open(args.test.resume_file, 'rb'))
    feature_extractor.load_state_dict(data['feature_extractor'])
    classifier.load_state_dict(data['classifier'])
    discriminator.load_state_dict(data['discriminator'])
    discriminator_separate.load_state_dict(data['discriminator_separate'])
    feat_all = []
    label_all = []

    counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    with TrainingModeManager([feature_extractor, classifier, discriminator_separate], train=False) as mgr, \
            Accumulator(['feature', 'predict_prob', 'label', 'domain_prob', 'before_softmax',
                         'target_share_weight']) as target_accumulator, \
            torch.no_grad():    
        for i,(im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
            #print(i,'uuuuuu')
            #input()
            im = im.to(output_device)
            label = label.to(output_device)
            #print(label)
            #input()
            label_all = list(label_all) + list(label)


            feature = feature_extractor.forward(im)
            feature, __, before_softmax, predict_prob = classifier.forward(feature)
            domain_prob = discriminator_separate.forward(__)
            ss = feature.tolist()
            #print(ss,'9999')
            #feattt = [[random.random() for _ in range(2048)]for _ in range(8)]
            #print(feattt,'0000')
            #for j in range(8):
                #feattt[j] = feature[j].tolist()
            feat_all = list(feat_all) + list(ss)
            #print(feature,'qqqqq')
            #print(before_softmax,'wwwww')
            #print(predict_prob,'eeeeee')
            #input()

            target_share_weight = get_target_share_weight(domain_prob, before_softmax, domain_temperature=1.0,
                                                            class_temperature=1.0)

            for name in target_accumulator.names:
                globals()[name] = variable_to_numpy(globals()[name])

            target_accumulator.updateData(globals())
    #print(label_all,'qqqqq')
    
    for x in target_accumulator:
        globals()[x] = target_accumulator[x]

    def outlier(each_target_share_weight):
        return each_target_share_weight < args.test.w_0         #T未知


#################################################################################################
    def calculate(list_val):
        total = 0
        T = 10
        for ele in range(0, len(list_val)):
            total = total + list_val[ele]
        return T*np.log(total)

   
    T = 10
    #i = 0
    #Ttfeat = np.array(Tfeatall)
    #Ttfeat = Ttfeat.cpu().numpy()
    for i in range(1031):
        list_logit=[np.exp(feat_all[i])/T]    #Tfeatall[i]         for x in enumerate(logit_t_energy)Tfeat_2f[i]
    #logit_t_energy = Tfeat_2f .detach().cpu().numpy()
    #print(logit_t_energy) 
    #-E(X)  值越大，表示其越是分布内的样本，否则表示其越是分布外的样本
    energy = [calculate(x) for x in enumerate(list_logit)]#Tfeat_2f[i]
    #rr = a+b+c
    energy = energy/np.log(80)#1031
        
    #energy = torch.Tensor(energy)
    #energy = energy.cpu()
    energye = energy[0]
    
    print(energye,'8888')     
    #i = 0
    #j = 0
    #x = 0
    #lab = [0 for i in range(c+b+a)]
    #lab = [0 for n in range(1031)]
    
    #i = 0
    #for i in range(1031):
        #lab[i][0] = Tfeatal[i][0]
    #for i in range(1031):
        #if energy[0][i] < -4.5:
            #lab[i] = 2  #Tfeat_2e[i][2] = 2
            #x = x + 1
    #print(x,'99999')
        #else:
            #lab[i][1] = 4            
                
    


#################################################################################################
    #print(label)
    counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    ln = 0
    #print(np.size(counters),'1111111')
    for (each_predict_prob, each_label, each_target_share_weight, each_energy) in zip(predict_prob, label, target_share_weight, energye):
        #print(each_energy)
        #input()
        if outlier(each_target_share_weight[0])  and each_energy < -4.55:
            ln = ln + 1 

        if each_label in source_classes:
            counters[each_label].Ntotal += 1.0
            each_pred_id = np.argmax(each_predict_prob)
            if not outlier(each_target_share_weight[0]) and each_pred_id == each_label or each_energy >= -4.55:
                
                #print(each_target_share_weight[0],'777777')
                #如果是已知类并且分类正确
                    counters[each_label].Ncorrect += 1.0
                #print(counters[each_label].Ncorrect)
        else:
            counters[-1].Ntotal += 1.0
            if outlier(each_target_share_weight[0]) and each_energy < -4.55:
                counters[-1].Ncorrect += 1.0
                #print(counters[-1].Ncorrect)
    #print(counters,'333333333')
    print(counters[0].Ntotal,counters[0].Ncorrect,'44444')
    print(counters[1].Ntotal,counters[1].Ncorrect,'55555')
    print(counters[-1].Ntotal,counters[-1].Ncorrect,'66666')
    print(ln,'777777')

    acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
    acc_test = torch.ones(1, 1) * np.mean(acc_tests)
    #print(np.size(counters),'2222222')
    print(f'test accuracy is {acc_test.item()}')
    exit(0)

# ===================optimizer
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
optimizer_finetune = OptimWithSheduler(
    optim.SGD(feature_extractor.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_cls = OptimWithSheduler(
    optim.SGD(classifier.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_discriminator = OptimWithSheduler(
    optim.SGD(discriminator.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_discriminator_separate = OptimWithSheduler(
    optim.SGD(discriminator_separate.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)

global_step = 0
best_acc = 0

total_steps = tqdm(range(args.train.min_step),desc='global step')
epoch_id = 0



while global_step < args.train.min_step:

    #id = 0
########################################################################################################
    if global_step % args.test.test_interval == 0:
        #print('99999999')
        with open(join(log_dir, 'config.yaml'), 'w') as f:
            f.write(yaml.dump(save_config))

        f = open(r"txt1/1.txt", "r")
        #with open("txt1/1.txt", "r") as f:       #文件bai为du123.txt
        #sourceInLines= f.readlines()    #按行读出文件zhidao
        #f.close()
        new = [] #定义一个空列表，zhuan用来存储结果
        for line in f.readlines():
            temp1 = line.strip('\n') #去掉每行最内后的换行符'\n'
            temp2 = temp1.split(',') #以','为标志，将每容行分割成列表
            new.append(temp2) #将上一步得到的列表添加到new中
        f.close()
        #print(new)
        #numbers = list(map(int, new))

        #n = 0
        #j = 0
        new_id = []
        new_label = []

        for n in new:
            #print(n)
            new_id.append(int(n[0]))
            new_label.append(int(n[1]))
            #print(n)
        #new_id = new_id
        #new_id = np.array(new_id)
        #new_label = np.array(new_label)

        new_t = [[random.random() for _ in range(2)]for _ in range(np.size(new_id))]
        i = 0
        for i in range(np.size(new_id)):
            new_t[i][0] = new_id[i]
            new_t[i][1] = new_label[i]
            i = i + 1
        #print(new_id,'11111')
        #print(new_label,'77777')
        #print(new_t,'22222')



#########################################################################################################
    r = 0
    e = 0
    iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id} ', total=min(len(source_train_dl), len(target_train_dl)))
    epoch_id += 1

    for i, ((im_source, label_source), (im_target, label_target)) in enumerate(iters): 

        #print(i, im_source, label_source, im_target, label_target,'111111111')


        save_label_target = label_target  # for debug usage

        label_source = label_source.to(output_device)
        label_target = label_target.to(output_device)
        label_target = torch.zeros_like(label_target)

        # =========================forward pass
        im_source = im_source.to(output_device)
        im_target = im_target.to(output_device)

        fc1_s = feature_extractor.forward(im_source)
        fc1_t = feature_extractor.forward(im_target)


        Tfc1_s = fc1_s.tolist()
        Tfc1_t = fc1_t.tolist()
        #Tfc1_s = fc1_s.cpu().detach().numpy()
        #Tfeat_0f[i][j] = Tfeat_0f[i][j].cpu().numpy()
        #Tfc1_t = fc1_t.cpu().detach().numpy()
        #Tfc1_s = np.array(Tfc1_s)
        #Tfc1_t = np.array(Tfc1_t)

        #new_id = []
        #new_label = []
        #print(Tfc1_s,Tfc1_t,np.size(Tfc1_s,axis=0),np.size(Tfc1_s,axis=1),np.size(Tfc1_t,axis=0),np.size(Tfc1_t,axis=1),'22222222')#################################
        
        #Tfeat_s = []
        Tfeat_s = Tfc1_s
        Tfeat_t = []
        Tfeat_t22 = []
        Tlable = []
        Tlable_t22 = label_source
        #Tlable = label_source
        Tlable_t22 = Tlable_t22.tolist()
        #print(label_target,'99999999')
        i = 0
        #for id in new_id:
        
        for i in range(np.size(new_id)):
            idd = r
            if e != 0:
                y = idd-(32*e)
            if e == 0:
                y = idd
            #y = idd-(32*e)
            for y in range(32):
                if idd == new_t[i][0]:
                    if e != 0:
                        x = idd-(32*e)
                    if e == 0:
                        x = idd

                    #print(Tfc1_s[x])
                    #Tfeat_s += list([Tfc1_s[x]])
                    #print(new_t[i][0],new_t[i][1],'1111')
                    Tfeat_t += list([Tfc1_t[x]])
                    #new_label
                    Tlable.append(new_t[i][1])
                    if new_t[i][1] == 2:
                        Tfeat_t22 += list([Tfc1_t[x]])
                        Tlable_t22.append(new_t[i][1])
                    #print(Tfeat_s,Tfeat_t,Tlable,'333333333')
                    #print(idd,'11111111111')
                    #input() 
                idd = idd + 1
                y = y + 1
        Tfeat_t = np.array(Tfeat_t)
        #input()
        Tfeat_s = list(Tfeat_s) + list(Tfeat_t22) 
        Tfeat_s = np.array(Tfeat_s)
        Tlable = np.array(Tlable)
        Tlable_t22 = np.array(Tlable_t22)
        #print(label_target)
        #print(Tfeat_s,Tlable)
        #input()
        #print(Tfeat_s,Tfeat_t,Tlable,'333333333')
        #print(np.size(Tfeat_s,axis=1),np.size(Tfeat_t,axis=1),Tlable,'333333333')
        #print(idd,'11111111111')
        #input() 
        #print(Tfeat_s,Tfeat_t,Tlable)
        #input()
        if Tfeat_s != []: 
            Tloss_s =  batch_all_triplet_loss(Tlable_t22, Tfeat_s, 0.4, False)#batch_hard_triplet_loss(Tlable, Tfeat_s, 0.3, False)#
        if Tfeat_t != []:               
            Tloss_t = batch_all_triplet_loss(Tlable, Tfeat_t, 0.2, False)#batch_all_triplet_loss(Tlable, Tfeat_t, 0.3, False)#
        if Tfeat_s == []:
            Tloss_s = 0
        if Tfeat_t == []:
            Tloss_t = 0    
        Tloss = Tloss_s + Tloss_t 
        #print(Tloss)
        #Tloss1 += 
        #print(Tloss_s,Tloss_t,'888888')
        #input() 

        fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
        fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)

        #Tfeature_source = feature_source.cpu().detach().numpy()
        #Tfeature_target = feature_target.cpu().detach().numpy()
        #Tfeature_source = np.array(Tfeature_source)
        #Tfeature_target = np.array(Tfeature_target)

        #print(predict_prob_source,predict_prob_target,'333333333')

        #print(fc1_s, fc2_s, predict_prob_source, fc1_t, fc2_t, predict_prob_target,'333333333')
        #print(Tfeature_source, Tfeature_target, np.size(Tfeature_source,axis=0),np.size(Tfeature_target,axis=0),'444444444')

        #for j in range(32): 

            #Tfeat[j] = fc1_s[j]
            #r = 0
            #for r in range(2048):
                #Tfeat[j][r] = 


        #input()


        domain_prob_discriminator_source = discriminator.forward(feature_source)
        domain_prob_discriminator_target = discriminator.forward(feature_target)

        domain_prob_discriminator_source_separate = discriminator_separate.forward(feature_source.detach())
        domain_prob_discriminator_target_separate = discriminator_separate.forward(feature_target.detach())

        source_share_weight = get_source_share_weight(domain_prob_discriminator_source_separate, fc2_s, domain_temperature=1.0, class_temperature=10.0)
        source_share_weight = normalize_weight(source_share_weight)
        target_share_weight = get_target_share_weight(domain_prob_discriminator_target_separate, fc2_t, domain_temperature=1.0, class_temperature=1.0)
        target_share_weight = normalize_weight(target_share_weight)

        # ==============================compute loss
        adv_loss = torch.zeros(1, 1).to(output_device)
        adv_loss_separate = torch.zeros(1, 1).to(output_device)
    
        tmp = source_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
        adv_loss += torch.mean(tmp, dim=0, keepdim=True)
        #print(tmp,adv_loss)
        tmp = target_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))
        adv_loss += torch.mean(tmp, dim=0, keepdim=True)
        #print(adv_loss,'111')

        adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_source_separate, torch.ones_like(domain_prob_discriminator_source_separate))
        adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_target_separate, torch.zeros_like(domain_prob_discriminator_target_separate))
        #print(adv_loss_separate,'55555555')
        # ============================== cross entropy loss
        ce = nn.CrossEntropyLoss(reduction='none')(predict_prob_source, label_source)
        #print(ce,'55555555')
        ce = torch.mean(ce, dim=0, keepdim=True)
        #print(ce,'eeeeeee')
        with OptimizerManager(
                [optimizer_finetune, optimizer_cls, optimizer_discriminator, optimizer_discriminator_separate]):
            loss = ce + adv_loss + adv_loss_separate + Tloss
            #print(loss,'7777777')
            #input()
            loss.backward()

        global_step += 1
        total_steps.update()

        if global_step % args.log.log_interval == 0:
            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))), variable_to_numpy(predict_prob_source))
            acc_train = torch.tensor([counter.reportAccuracy()]).to(output_device)
            logger.add_scalar('adv_loss', adv_loss, global_step)
            logger.add_scalar('ce', ce, global_step)
            logger.add_scalar('adv_loss_separate', adv_loss_separate, global_step)
            logger.add_scalar('acc_train', acc_train, global_step)

        if global_step % args.test.test_interval == 0:

            counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
            with TrainingModeManager([feature_extractor, classifier, discriminator_separate], train=False) as mgr, \
                 Accumulator(['feature', 'predict_prob', 'label', 'domain_prob', 'before_softmax', 'target_share_weight']) as target_accumulator, \
                 torch.no_grad():

                txt_id = [[0 for _ in range(3)]for _ in range(1031)]
                idd = [0 for _ in range(1031)]
                txt_00 = [[0 for _ in range(3)]for _ in range(1031)]
                txt_11 = [[0 for _ in range(3)]for _ in range(1031)]
                txt_22 = [[0 for _ in range(3)]for _ in range(1031)]
                for i in range(1031):
                    txt_id[i][0] = i
                
                    idd[i] = i
                    #print(txt_id[i][0],'111111')

                h = 0
                feat_all = []
                for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
                    #print(i,'00000')
                    im = im.to(output_device)
                    label = label.to(output_device)

                    

                    feature = feature_extractor.forward(im)
                    feature, __, before_softmax, predict_prob = classifier.forward(feature)
                    domain_prob = discriminator_separate.forward(__)

                    
                    predict_prob1 = predict_prob.tolist()
                    #print(np.size(predict_prob1,0),'22222')
                    for j in range(np.size(predict_prob1,0)):
                        if predict_prob1[j][0] > predict_prob1[j][1]:
                            #print('u')
                            #print(predict_prob1[j][0],'333333')
                            txt_id[h][2] = predict_prob1[j][0]
                        else:
                            txt_id[h][2] = predict_prob1[j][1]
                        #print(txt_id[h],'22222')
                        h = h + 1
                        
                    #input()
                    ss = feature.tolist()
                    feat_all = list(feat_all) + list(ss)
                    

                    target_share_weight = get_target_share_weight(domain_prob, before_softmax, domain_temperature=1.0,
                                                                  class_temperature=1.0)

                    for name in target_accumulator.names:
                        globals()[name] = variable_to_numpy(globals()[name])

                    target_accumulator.updateData(globals())

            #print(txt_id,'22222')

            for x in target_accumulator:
                globals()[x] = target_accumulator[x]

            def outlier(each_target_share_weight):
                return each_target_share_weight < args.test.w_0
#################################################################################################
            def calculate(list_val):
                total = 0
                T = 10
                for ele in range(0, len(list_val)):
                    total = total + list_val[ele]
                return T*np.log(total)

   
            T = 10
    
            for i in range(1031):
                list_logit=[np.exp(feat_all[i])/T]    
    
            energy = [calculate(x) for x in enumerate(list_logit)]#Tfeat_2f[i]  
            energy = energy/np.log(80)#1031
            energye = energy[0]
            
            print(energye,'8888')   
####################################################################################################################

            counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
            ln = 0
            a = 0
            b = 0
            c = 0
            for (each_predict_prob, each_label, each_target_share_weight, each_energy, i) in zip(predict_prob, label,
                                                                                 target_share_weight, energye, idd):
                
                #print(i)
                if outlier(each_target_share_weight[0]) and each_energy < -4.55:
                    txt_id[i][1] = 2
                    ln = ln + 1
                else:
                    txt_id[i][1] = np.argmax(each_predict_prob)
                #print(txt_id[i])
                #input()
                if txt_id[i][1] == 2:
                    #txt_22[c].append(txt_id[i])
                    txt_22[c] = txt_id[i]
                    c = c + 1
                if txt_id[i][1] == 0:
                    #txt_00[a].append(txt_id[i])
                    txt_00[a] = txt_id[i]
                    a = a + 1
                if txt_id[i][1] == 1:
                    #txt_11[b].append(txt_id[i])
                    txt_11[b] = txt_id[i]
                    b = b + 1
                



                if each_label in source_classes:
                    counters[each_label].Ntotal += 1.0
                    each_pred_id = np.argmax(each_predict_prob)
                    if not outlier(each_target_share_weight[0]) and each_pred_id == each_label or each_energy >= -4.55:
                        counters[each_label].Ncorrect += 1.0
                        
                else:
                    counters[-1].Ntotal += 1.0
                    if outlier(each_target_share_weight[0]) and each_energy < -4.55:
                        counters[-1].Ncorrect += 1.0



            #print(txt_00,txt_11,'wwwwwwwwwww')
            #print(a,b,c,'zzzzzzzzzzz')
            sorted(txt_00, key=lambda s: s[2], reverse=True)
            sorted(txt_11, key=lambda s: s[2], reverse=True)
            a = int(a*0.35)
            b = int(b*0.35)
            txt_0 = [[random.random() for _ in range(2)]for _ in range(a)]
            txt_1 = [[random.random() for _ in range(2)]for _ in range(b)]
            txt_2 = [[random.random() for _ in range(2)]for _ in range(c)]
            txt_all = []
            for i in range(a):
                txt_0[i][0] = txt_00[i][0]
                txt_0[i][1] = txt_00[i][1]
            for i in range(b):
                txt_1[i][0] = txt_11[i][0]
                txt_1[i][1] = txt_11[i][1]
            for i in range(c):
                txt_2[i][0] = txt_22[i][0]
                txt_2[i][1] = txt_22[i][1]
            txt_all = list(txt_0) + list(txt_1) + list(txt_2)



            #print(txt_0,'rrrrrrrr')
            #print(txt_1,'qqqqqqqq')
            #print(txt_2,'eeeeeeee')
            print(counters[0].Ntotal,counters[0].Ncorrect,'44444')
            print(counters[1].Ntotal,counters[1].Ncorrect,'55555')
            print(counters[-1].Ntotal,counters[-1].Ncorrect,'66666')
            print(ln,'777777')


            acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
            acc_test = torch.ones(1, 1) * np.mean(acc_tests)
            print(f'test accuracy is {acc_test.item()}')
            #input()
            with open("txt1/1.txt", "w") as output:
                i = 0
                for i in range(a+b+c):
                    s = str(txt_all[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
                    s = s.replace("'",'')+'\n'  #去除单引号，逗号，每行末尾追加换行符  .replace(',','') 
                    output.write(s)

            logger.add_scalar('acc_test', acc_test, global_step)
            clear_output()

            data = {
                "feature_extractor": feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'discriminator': discriminator.state_dict() if not isinstance(discriminator, Nonsense) else 1.0,
                'discriminator_separate': discriminator_separate.state_dict(),
            }

            if acc_test > best_acc:
                best_acc = acc_test
                with open(join(log_dir, 'best.pkl'), 'wb') as f:
                    torch.save(data, f)

            with open(join(log_dir, 'current.pkl'), 'wb') as f:
                torch.save(data, f)


        #input()
        r = r + 32
        e = e + 1  
        #print(r,e)
        #input()