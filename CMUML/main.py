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

from eval import batch_hard_triplet_loss
from eval import batch_all_triplet_loss
from eval import convert_label_to_similarity
from eval import CircleLoss


cudnn.benchmark = True
cudnn.deterministic = True

seed_everything()

# if args.misc.gpus < 1:
#     import os
# 
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""
#     gpu_ids = []
#     device = torch.device('cpu')
# else:
#     # gpu_ids = select_GPUs(args.misc.gpus)
#     gpu_ids = [0]
#     device = gpu_ids[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

log_dir = f'{args.log.root_dir}/{now}'

logger = SummaryWriter(log_dir)

with open(join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(save_config))

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
        # self.discriminator_separate = AdversarialNetwork(256)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        d_0 = self.discriminator_separate(_)
        return y, d, d_0


totalNet = TotalNet()

# feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=gpu_ids, device=device).train(
#     True)
# classifier = nn.DataParallel(totalNet.classifier, device_ids=gpu_ids, device=device).train(True)
# # discriminator = nn.DataParallel(totalNet.discriminator, device_ids=gpu_ids, device=device).train(True)
# discriminator_separate = nn.DataParallel(totalNet.discriminator_separate, device_ids=gpu_ids,
#                                          device=device).train(True)
feature_extractor = totalNet.feature_extractor.to(device)
classifier = totalNet.classifier.to(device)
discriminator = totalNet.discriminator.to(device)
# discriminator_separate = totalNet.discriminator_separate.to(device)

if args.test.test_only:
    assert os.path.exists(args.test.resume_file)
    data = torch.load(open(args.test.resume_file, 'rb'))
    feature_extractor.load_state_dict(data['feature_extractor'])
    classifier.load_state_dict(data['classifier'])
    discriminator.load_state_dict(data['discriminator'])
    # discriminator_separate.load_state_dict(data['discriminator_separate'])
    feat_all = []

    counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    with TrainingModeManager([feature_extractor, classifier, discriminator], train=False) as mgr, \
            Accumulator(['feature', 'predict_prob', 'label', 'fc2_s',
                         'entropy', 'consistency', 'confidence']) as target_accumulator, \
            torch.no_grad():
        for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
            im = im.to(device)
            label = label.to(device)

            feature = feature_extractor.forward(im)
            feature, __, fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5, predict_prob = classifier.forward(feature)
            # domain_prob = discriminator_separate.forward(__)

            ss = feature.tolist()
            feat_all = list(feat_all) + list(ss)


            entropy = get_entropy(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5, domain_temperature=1.0,
                                  class_temperature=1.0)
            consistency = get_consistency(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5)

            confidence, indices = torch.max(predict_prob, dim=1)

            # predict_prob = get_predict_prob(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5)
            for name in target_accumulator.names:
                globals()[name] = variable_to_numpy(globals()[name])

            target_accumulator.updateData(globals())

    for x in target_accumulator:
        globals()[x] = target_accumulator[x]

    entropy = normalize_weight(torch.tensor(entropy))
    consistency = normalize_weight(torch.tensor(consistency))
    confidence = nega_weight(torch.tensor(confidence))
#######################################################################
    # print(entropy.size())
    # print(consistency.size())
    # target_share_weight = (entropy + consistency) / 2
    #target_share_weight = (entropy + consistency + confidence) / 3
    target_share_weight = (confidence + 1 - consistency + 1 - entropy) / 3
############################################################################
    entropy_common = []
    entropy_private = []
    consistency_common = []
    consistency_private = []
    confidence_common = []
    confidence_private = []
    weight_common = []
    weight_private = []
    for (each_entropy, each_consistency, each_confidence, each_weight, each_label) \
            in zip(entropy, consistency, confidence, target_share_weight, label):
        if each_label < 10:
            entropy_common.append(each_entropy)
            consistency_common.append(each_consistency)
            confidence_common.append(each_confidence)
            weight_common.append(each_weight)
        else:
            entropy_private.append(each_entropy)
            consistency_private.append(each_consistency)
            confidence_private.append(each_confidence)
            weight_private.append(each_weight)

    # for x in target_accumulator:
    # print(target_accumulator['target_share_weight'])
    # print(entropy.size())
    # hist, bin_edges = np.histogram(entropy_common, bins=10, range=(0, 1))
    # print(hist)
    # print(bin_edges)
    #
    # hist, bin_edges = np.histogram(entropy_private, bins=10, range=(0, 1))
    # print(hist)
    # print(bin_edges)

    hist, bin_edges = np.histogram(confidence_common, bins=10, range=(0, 1))
    #print(hist)
    #print(bin_edges)

    hist, bin_edges = np.histogram(confidence_private, bins=10, range=(0, 1))
    #print(hist)
    #print(bin_edges)

    #
    # hist, bin_edges = np.histogram(consistency, bins=20, range=(0, 1))
    # print(hist)
    # print(bin_edges)

    # ana = list(zip(entropy, consistency, confidence, target_share_weight, label))
    # array = sorted(ana, key=lambda x: x[0])
    # np.savetxt("ana.csv", array, delimiter=',')

    # print(array)
    #
    # a1, a2, a3 = zip(*array)
    # print(a1)
    # print(a2)
    # print(a3)
    '''
    estimate_counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    
    for (each_predict_prob, each_label, each_target_share_weight) in zip(predict_prob, label, target_share_weight):
        each_pred_id = np.argmax(each_predict_prob)
        if each_target_share_weight < (args.test.w_0/2):
            estimate_counters[int(each_pred_id)].Npred += 1.0

    class_ratio = [x.Npred for x in estimate_counters]
    print(class_ratio)

    common_threshold = np.mean(class_ratio) / 4
    common_estimate = []
    for i in range(len(estimate_counters)):
        if estimate_counters[i].Npred > common_threshold:
            common_estimate.append(i)
    '''

    # print(common_estimate)

    # def outlier(each_target_share_weight, each_pred_id):
    #     return each_target_share_weight > args.test.w_0 or each_pred_id not in common_estimate
    #def outlier(each_target_share_weight, each_pred_id):
        #return each_target_share_weight > args.test.w_0

    def outlier(each_target_share_weight):
        #return each_target_share_weight > args.test.w_0
        return each_target_share_weight < args.test.w_0

############################################################################################
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
    
    #print(energye,'8888')    
    
################################################################################


    counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    '''
    for (each_predict_prob, each_label, each_target_share_weight) in zip(predict_prob, label, target_share_weight):
        if each_label in source_classes:
            counters[each_label].Ntotal += 1.0
            each_pred_id = np.argmax(each_predict_prob)
            if not outlier(each_target_share_weight, each_pred_id):
                counters[int(each_pred_id)].Npred += 1.0
            if not outlier(each_target_share_weight, each_pred_id) and each_pred_id == each_label:
                counters[each_label].Ncorrect += 1.0
        else:
            counters[-1].Ntotal += 1.0
            each_pred_id = np.argmax(each_predict_prob)
            if outlier(each_target_share_weight, each_pred_id):
                counters[-1].Ncorrect += 1.0
                counters[-1].Npred += 1.0
            else:
                counters[int(each_pred_id)].Npred += 1.0

    # class_ratio = [x.Npred for x in counters]
    # print(class_ratio)

    acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
    correct = [x.Ncorrect for x in counters]
    amount = [x.Ntotal for x in counters]
    common_acc = np.sum(correct[0:-1]) / np.sum(amount[0:-1])
    outlier_acc = correct[-1] / amount[-1]
    '''

    ln = 0
    ee1 = 0
    ee0 = 0
    ee = 0
    for (each_predict_prob, each_label, each_target_share_weight, each_energy) in zip(predict_prob, label, target_share_weight, energye):
    #for (each_predict_prob, each_label, each_target_share_weight) in zip(predict_prob, label, target_share_weight):
        if outlier(each_target_share_weight) and each_energy < -1:
            ln = ln + 1 
        if not outlier(each_target_share_weight) or each_energy >= -1:
            ee += 1
        each_pred_id = np.argmax(each_predict_prob)

        if not outlier(each_target_share_weight) or each_energy >= -1:
            if each_pred_id == 0:
                ee0 += 1
        if not outlier(each_target_share_weight) or each_energy >= -1:
            if each_pred_id == 1:
                ee1 += 1

        if each_label in source_classes:
            counters[each_label].Ntotal += 1.0
            each_pred_id = np.argmax(each_predict_prob)
            if not outlier(each_target_share_weight) or each_energy >= -1:
                if each_pred_id == each_label:
                    counters[each_label].Ncorrect += 1.0
        else:
            counters[-1].Ntotal += 1.0
            if outlier(each_target_share_weight) and each_energy < -1:
                counters[-1].Ncorrect += 1.0

    acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]

    correct = [x.Ncorrect for x in counters]
    amount = [x.Ntotal for x in counters]
    common_acc = np.sum(correct[0:-1]) / np.sum(amount[0:-1])
    outlier_acc = correct[-1] / amount[-1]

    print(ln,'eeeeeeeee')
    print(ee,'eeeeeeeee')
    print(counters[0].Ntotal,counters[0].Ncorrect,'11111111111')
    print(counters[1].Ntotal,counters[1].Ncorrect,'22222222222')
    print(counters[-1].Ntotal,counters[-1].Ncorrect,'3333333333')
    print(ee0,ee1,'44444444444')

    print('common_acc={}, outlier_acc={}'.format(common_acc, outlier_acc))
    bscore = 2 / (1 / common_acc + 1 / outlier_acc)
    acc_test = torch.ones(1, 1) * np.mean(acc_tests)        

    #print('common_acc={}, outlier_acc={}'.format(common_acc, outlier_acc))
    #bscore = 2 / (1 / common_acc + 1 / outlier_acc)
    print('hscore={}'.format(bscore))
    #acc_test = torch.ones(1, 1) * np.mean(acc_tests)
    print('perclass accuracy is {}'.format(acc_test.item()))
    exit(0)

# ===================optimizer
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
optimizer_finetune = OptimWithSheduler(
    optim.SGD(feature_extractor.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay,
              momentum=args.train.momentum, nesterov=True), scheduler)
optimizer_cls = OptimWithSheduler(
    optim.SGD(classifier.bottleneck.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay,
              momentum=args.train.momentum, nesterov=True), scheduler)
fc_para = [{"params": classifier.fc.parameters()}, {"params": classifier.fc2.parameters()},
           {"params": classifier.fc3.parameters()}, {"params": classifier.fc4.parameters()},
           {"params": classifier.fc5.parameters()}]
optimizer_fc = OptimWithSheduler(
    optim.SGD(fc_para, lr=args.train.lr * 5, weight_decay=args.train.weight_decay,
              momentum=args.train.momentum, nesterov=True), scheduler)
optimizer_discriminator = OptimWithSheduler(
    optim.SGD(discriminator.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay,
              momentum=args.train.momentum, nesterov=True), scheduler)
# optimizer_discriminator_separate = OptimWithSheduler(
#     optim.SGD(discriminator_separate.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay,
#               momentum=args.train.momentum, nesterov=True), scheduler)

global_step = 0
best_acc = 0

total_steps = tqdm(range(args.train.min_step), desc='global step')
epoch_id = 0
threshold = torch.zeros(1).to(device)
while global_step < args.train.min_step:

#####################################################################################
    
    if global_step % args.test.test_interval == 0:
        #print('99999999')
        with open(join(log_dir, 'config.yaml'), 'w') as f:
            f.write(yaml.dump(save_config))

        f = open(r"txt1/1 copy.txt", "r")
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
    r = 0
    e = 0
    
###################################################################################

    iters = tqdm(
        zip(source_train_dl, source_train_dl2, source_train_dl3, source_train_dl4, source_train_dl5, target_train_dl),
        desc=f'epoch {epoch_id} ', total=min(len(source_train_dl), len(target_train_dl)))
    epoch_id += 1

    for i, ((im_source, label_source), (im_source2, label_source2), (im_source3, label_source3),
            (im_source4, label_source4), (im_source5, label_source5), (im_target, label_target)) in enumerate(iters):

        feature_extractor.train()
        classifier.train()

        save_label_target = label_target  # for debug usage

        label_source = label_source.to(device)
        label_source2 = label_source2.to(device)
        label_source3 = label_source3.to(device)
        label_source4 = label_source4.to(device)
        label_source5 = label_source5.to(device)
        label_target = label_target.to(device)

        # =========================forward pass
        im_source = im_source.to(device)
        im_source2 = im_source2.to(device)
        im_source3 = im_source3.to(device)
        im_source4 = im_source4.to(device)
        im_source5 = im_source5.to(device)
        im_target = im_target.to(device)

        fc1_s = feature_extractor.forward(im_source)
        fc1_s2 = feature_extractor.forward(im_source2)
        fc1_s3 = feature_extractor.forward(im_source3)
        fc1_s4 = feature_extractor.forward(im_source4)
        fc1_s5 = feature_extractor.forward(im_source5)
        fc1_t = feature_extractor.forward(im_target)

        Tfc1_s = fc1_s.tolist()
        Tfc1_t = fc1_t.tolist()
        #print(Tfc1_s, Tfc1_t, 'tttttttttttttttt')
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
                y = idd-(36*e)
            if e == 0:
                y = idd
            #y = idd-(32*e)
            for y in range(36):
                if idd == new_t[i][0]:
                    if e != 0:
                        x = idd-(36*e)
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
            Tloss_s =  batch_all_triplet_loss(Tlable_t22, Tfeat_s, 0.4, False)
            #Tloss_s =  batch_hard_triplet_loss(Tlable_t22, Tfeat_s, 0.4, False)#
        if Tfeat_t != []:               
            Tloss_t = batch_all_triplet_loss(Tlable, Tfeat_t, 0.2, False)
            #Tloss_t = batch_hard_triplet_loss(Tlable, Tfeat_t, 0.2, False)#
        if Tfeat_s == []:
            Tloss_s = 0
        if Tfeat_t == []:
            Tloss_t = 0    
        Tloss = Tloss_s + Tloss_t 
        '''
        Tlable = torch.from_numpy(Tlable)
        Tfeat_s1 = torch.from_numpy(Tfeat_s)
        Tlable_t22 = torch.from_numpy(Tlable_t22)
        Tfeat_t1 = torch.from_numpy(Tfeat_t)
        if Tfeat_s != []: 
            inp_sp, inp_sn = convert_label_to_similarity(Tfeat_s1, Tlable_t22)
            criterion = CircleLoss(m=0.1, gamma=256)
            circle_loss = criterion(inp_sp, inp_sn)
            Tloss_s = circle_loss
            Tloss_s = np.array(Tloss_s)
        if Tfeat_t != []:               
            inp_sp, inp_sn = convert_label_to_similarity(Tfeat_t1, Tlable)
            criterion = CircleLoss(m=0.4, gamma=256)
            circle_loss = criterion(inp_sp, inp_sn)
            Tloss_t = circle_loss
            Tloss_t = np.array(Tloss_t)
        if Tfeat_s == []:
            Tloss_s = 0
        if Tfeat_t == []:
            Tloss_t = 0    
        Tloss = Tloss_s + Tloss_t
        '''
#########################################################################


        fc1_s, feature_source, fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5, predict_prob_source = classifier.forward(fc1_s)
        fc1_s2, feature_source2, fc2_s_2, fc2_s2_2, fc2_s3_2, fc2_s4_2, fc2_s5_2, predict_prob_source2 = \
            classifier.forward(fc1_s2)
        fc1_s3, feature_source3, fc2_s_3, fc2_s2_3, fc2_s3_3, fc2_s4_3, fc2_s5_3, predict_prob_source3 = \
            classifier.forward(fc1_s3)
        fc1_s4, feature_source4, fc2_s_4, fc2_s2_4, fc2_s3_4, fc2_s4_4, fc2_s5_4, predict_prob_source4 = \
            classifier.forward(fc1_s4)
        fc1_s5, feature_source5, fc2_s_5, fc2_s2_5, fc2_s3_5, fc2_s4_5, fc2_s5_5, predict_prob_source5 = \
            classifier.forward(fc1_s5)
        fc1_t, feature_target, fc2_t, fc2_t2, fc2_t3, fc2_t4, fc2_t5, predict_prob_target = classifier.forward(fc1_t)

        domain_prob_discriminator_source = discriminator.forward(feature_source)
        domain_prob_discriminator_target = discriminator.forward(feature_target)

        source_share_weight = get_label_weight(label_source, common_classes).view(36, 1).to(device)

        entropy = get_entropy(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5,
                              domain_temperature=1.0, class_temperature=1.0).detach()
        consistency = get_consistency(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5).detach()
        # confidence, indices = torch.max(predict_prob_target, dim=1)
        target_share_weight = get_target_weight(entropy, consistency, threshold).view(36, 1).to(device)

        if global_step < 500:
            source_share_weight = torch.zeros_like(source_share_weight)
            target_share_weight = torch.zeros_like(target_share_weight)

        # ==============================compute loss
        adv_loss = torch.zeros(1, 1).to(device)
        # adv_loss_separate = torch.zeros(1, 1).to(device)

        tmp = source_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source,
                                                                 torch.ones_like(domain_prob_discriminator_source))
        adv_loss += torch.mean(tmp, dim=0, keepdim=True)
        tmp = target_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_target,
                                                                 torch.zeros_like(domain_prob_discriminator_target))
        adv_loss += torch.mean(tmp, dim=0, keepdim=True)

        # ============================== cross entropy loss, it receives logits as its inputs

        ce = nn.CrossEntropyLoss()(fc2_s, label_source)
        ce2 = nn.CrossEntropyLoss()(fc2_s2_2, label_source2)
        ce3 = nn.CrossEntropyLoss()(fc2_s3_3, label_source3)
        ce4 = nn.CrossEntropyLoss()(fc2_s4_4, label_source4)
        ce5 = nn.CrossEntropyLoss()(fc2_s5_5, label_source5)

        with OptimizerManager(
                [optimizer_finetune, optimizer_cls, optimizer_fc, optimizer_discriminator]):
            # [optimizer_finetune, optimizer_cls, optimizer_discriminator, optimizer_discriminator_separate]):
            # loss = ce + adv_loss + adv_loss_separate
            loss = (ce + ce2 + ce3 + ce4 + ce5) / 5 + adv_loss + Tloss
            loss.backward()

        global_step += 1
        total_steps.update()

        if global_step % args.log.log_interval == 0:
            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))),
                                variable_to_numpy(predict_prob_source))
            acc_train = torch.tensor([counter.reportAccuracy()]).to(device)
            logger.add_scalar('adv_loss', adv_loss, global_step)
            logger.add_scalar('ce', ce, global_step)
            # logger.add_scalar('adv_loss_separate', adv_loss_separate, global_step)
            logger.add_scalar('acc_train', acc_train, global_step)

        if global_step % args.test.test_interval == 0:

            feature_extractor.eval()
            classifier.eval()
            entropy = None
            consistency = None

            counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
            with TrainingModeManager([feature_extractor, classifier, discriminator], train=False) as mgr, \
                    Accumulator(['feature', 'predict_prob', 'label', 'entropy', 'consistency',
                                 'confidence']) as target_accumulator, torch.no_grad():

#############################################################
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
##############################################################                

                for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing')):
                    im = im.to(device)
                    label = label.to(device)

                    feature = feature_extractor.forward(im)
                    feature, __, fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5, predict_prob = classifier.forward(
                        feature)

                    entropy = get_entropy(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5,
                                          domain_temperature=1.0, class_temperature=1.0).detach()
                    consistency = get_consistency(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5).detach()
                    confidence, indices = torch.max(predict_prob, dim=1)

###################################################################
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
##############################################################

                    for name in target_accumulator.names:
                        globals()[name] = variable_to_numpy(globals()[name])

                    target_accumulator.updateData(globals())

            for x in target_accumulator:
                globals()[x] = target_accumulator[x]

            entropy = normalize_weight(torch.tensor(entropy))
            consistency = normalize_weight(torch.tensor(consistency))
            #confidence = nega_weight(torch.tensor(confidence))
            confidence = nega_weight(torch.tensor(confidence))
            #target_share_weight = (entropy + consistency) / 2
            target_share_weight = (confidence + 1 - consistency + 1 - entropy) / 3

            threshold = torch.mean(target_share_weight).to(device)

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
            
            #print(energye,'8888')   
####################################################################################################################


            counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
            ln = 0
            a = 0
            b = 0
            c = 0
            
            for (each_predict_prob, each_label, each_target_share_weight, each_energy, i) in zip(predict_prob, label,
                                                                                 target_share_weight, energye, idd):
        
                
                if outlier(each_target_share_weight) and each_energy < -3:
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
                    if not outlier(each_target_share_weight) or each_energy >= -3:
                        if each_pred_id == each_label:
                            counters[each_label].Ncorrect += 1.0
                else:
                    counters[-1].Ntotal += 1.0
                    if outlier(each_target_share_weight) and each_energy < -3:
                        counters[-1].Ncorrect += 1.0
                

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

            acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]

            correct = [x.Ncorrect for x in counters]
            amount = [x.Ntotal for x in counters]
            common_acc = np.sum(correct[0:-1]) / np.sum(amount[0:-1])
            outlier_acc = correct[-1] / amount[-1]


            print(ln,'eeeeeeeee')
            print(counters[0].Ntotal,counters[0].Ncorrect,'11111111111')
            print(counters[1].Ntotal,counters[1].Ncorrect,'22222222222')
            print(counters[-1].Ntotal,counters[-1].Ncorrect,'3333333333')

            print('common_acc={}, outlier_acc={}'.format(common_acc, outlier_acc))
            bscore = 2 / (1 / common_acc + 1 / outlier_acc)
            acc_test = torch.ones(1, 1) * np.mean(acc_tests)

            
            with open("txt1/1 copy.txt", "w") as output:
                i = 0
                for i in range(a+b+c):
                    s = str(txt_all[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
                    s = s.replace("'",'')+'\n'  #去除单引号，逗号，每行末尾追加换行符  .replace(',','') 
                    output.write(s)
            

            logger.add_scalar('acc_test', acc_test, global_step)
            logger.add_scalar('bscore', bscore, global_step)
            # clear_output()

            data = {
                "feature_extractor": feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'discriminator': discriminator.state_dict() if not isinstance(discriminator, Nonsense) else 1.0,
                # 'discriminator_separate': discriminator_separate.state_dict(),
            }

            if acc_test > best_acc:
                best_acc = acc_test
                with open(join(log_dir, 'best.pkl'), 'wb') as f:
                    torch.save(data, f)

            with open(join(log_dir, 'current.pkl'), 'wb') as f:
                torch.save(data, f)

        
        r = r + 36
        e = e + 1 
        
