from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import random
import os
import argparse
import numpy as np
from PreResNet_cifar import *
import dataloader_cifar_svm as dataloader
from math import log2
from Contrastive_loss import *
from util_svmfix_cifar_svm import svmfix_train_flex
from sklearn.mixture import GaussianMixture

import warnings
warnings.filterwarnings('ignore')


## For plotting the logs
# import wandb
# wandb.init(project="noisy-label-project", entity="..")

## Arguments to pass 
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=350, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--d_u',  default=0.7, type=float)
parser.add_argument('--tau', default=5, type=float, help='filtering coefficient')
parser.add_argument('--metric', type=str, default = 'JSD', help='Comparison Metric')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--resume', default=False, type=bool, help = 'Resume from the warmup checkpoint')
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./data/cifar10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--lambda_tri', default=0.005, type=float)
parser.add_argument('--flex_threshold', default=0.5, type=float)
parser.add_argument('--gamma', default=1, type=float)
parser.add_argument('--epoch_start_svmfix', default=31, type=int)
parser.add_argument('--warm_up', default=10, type=int)
parser.add_argument('--margin', default=10, type=float)
parser.add_argument('--sample_threshold', default=0.5, type=float)
parser.add_argument('--up_threshold', default=0.6, type=float)
parser.add_argument('--lower_threshold', default=0.4, type=float)
parser.add_argument('--low_dim', default=128, type=int)


args = parser.parse_args()

## GPU Setup 
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

## Download the Datasets
if args.dataset== 'cifar10':
    torchvision.datasets.CIFAR10(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR10(args.data_path,train=False, download=True)
else:
    torchvision.datasets.CIFAR100(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR100(args.data_path,train=False, download=True)

## Checkpoint Location
# folder = args.dataset + '_' + args.noise_mode + '_' + str(args.r) + 'test2_flex<0.9=0.9+avgconjs+Lu0+Rjs-g0.7+simclr' + '_DivideJSGMM_lamtri0.005_st0.7_ft0.7'  # + 'svmfix_LC+NCE'
folder = args.dataset + '_' + args.noise_mode + '_' + str(args.r) + '_'+str(args.lambda_u) + '_'+str(args.sample_threshold) +'nolu'
model_save_loc = './checkpoint/' + 'svm_ablation/' + folder

if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)

## Log files
stats_log=open(model_save_loc +'/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
test_log=open(model_save_loc +'/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')     
test_loss_log = open(model_save_loc +'/test_loss.txt','w')
train_acc = open(model_save_loc +'/train_acc.txt','w')
train_loss = open(model_save_loc +'/train_loss.txt','w')


## For Standard Training 
def warmup_standard(epoch,net,optimizer,dataloader):

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        _, outputs = net(inputs)               
        loss    = CEloss(outputs, labels)    

        if args.noise_mode=='asym':     # Penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty

        else:   
            L = loss

        L.backward()  
        optimizer.step()                

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

## For Training Accuracy
def warmup_val(epoch,net,optimizer,dataloader):

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    total = 0
    correct = 0
    loss_x = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
            inputs, labels = inputs.cuda(), labels.cuda() 
            optimizer.zero_grad()
            _, outputs  = net(inputs)               
            _, predicted = torch.max(outputs, 1)    
            loss    = CEloss(outputs, labels)    
            loss_x += loss.item()                      

            total   += labels.size(0)
            correct += predicted.eq(labels).cpu().sum().item()

    acc = 100.*correct/total
    print("\n| Train Epoch #%d\t Accuracy: %.2f%%\n" %(epoch, acc))  
    
    train_loss.write(str(loss_x/(batch_idx+1)))
    train_acc.write(str(acc))
    train_acc.flush()
    train_loss.flush()

    return acc

## Test Accuracy
def test(epoch,net1,net2):
    net1.eval()
    net2.eval()

    num_samples = 1000
    correct = 0
    total = 0
    loss_x = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs1 = net1(inputs)
            _, outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
            loss = CEloss(outputs, targets)  
            loss_x += loss.item()

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()  

    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write(str(acc)+'\n')
    test_log.flush()  
    test_loss_log.write(str(loss_x/(batch_idx+1))+'\n')
    test_loss_log.flush()
    return acc




def eval_train(model, all_loss):
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs = model(inputs)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)


    if  args.r >= 0.9:  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)


    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss

def eval_train_svm(model, all_loss):
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))
    labels = torch.zeros(len(eval_loader.dataset), dtype=torch.long).cuda()
    cl_features = torch.zeros(len(eval_loader.dataset), args.low_dim).cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            cl_outputs, outputs = model(inputs)
            loss = CE(outputs, targets)


            cl_features[index] = cl_outputs
            labels[index] = targets
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)


    if  args.r >= 0.9:  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)


    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]

    return prob, all_loss, cl_features, labels


class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

def update_support_vectors(support_vectors_mean, support_vectors, support_labels, update_ratio=0.05):
    if support_vectors.size(0) == support_vectors_mean.size(0):
        return support_vectors_mean * (1 - update_ratio) + support_vectors * update_ratio
    else:
        for i in range(len(support_labels)):
            support_vectors_mean[support_labels[i]] = support_vectors_mean[support_labels[i]] \
                                                      * (1 - update_ratio) + support_vectors[i] * update_ratio
        return support_vectors_mean

def compute_mean(clean_loader, model, support_mean, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(clean_loader.dataset), args.low_dim).cuda()
    labels = torch.zeros(len(clean_loader.dataset), dtype=torch.long).cuda()
    class_sample_counts = torch.zeros(args.num_class).cuda()
    with torch.no_grad():
        for batch_idx, (inputs, _, _, _, targets, _, index) in enumerate(clean_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = model(inputs)
            features[index] = outputs
            labels[index] = targets

    for i in range(features.size(0)):
        label = labels[i]
        support_mean[label] += features[i]
        class_sample_counts[label] += 1

    support_vectors_mean = support_mean / class_sample_counts.view(-1, 1)
    return support_vectors_mean

def get_clean_uncertain_indices(prob_gmm, cl_features, labels, support_vectors_mean, args, epoch):
    pred_clean = (prob_gmm > args.sample_threshold)

    pred_uncertain = (prob_gmm > args.lower_threshold) & (prob_gmm < args.up_threshold)
    pred_uncertain_indices = pred_uncertain.nonzero()[0]
    print("len(pred_uncertain_indices) ", len(pred_uncertain_indices) )
    if len(pred_uncertain_indices) > 0:
        pred_uncertain_features = cl_features[pred_uncertain_indices]

        # 与各类别支持向量的余弦相似度
        cos_similarities = F.cosine_similarity(pred_uncertain_features.unsqueeze(1), support_vectors_mean.unsqueeze(0))

        # 找到最相似的中心向量对应的类别
        most_similar_classes = torch.argmax(cos_similarities, dim=1).to(torch.int)

        # 比较最相似类别与训练样本标签是否相同
        correct_indices_mask = (most_similar_classes == labels[pred_uncertain_indices].to(torch.int))

        values = torch.masked_select(correct_indices_mask, correct_indices_mask)

        # 计算选择出的张量的总和
        sum_values = torch.sum(values)

        print("sum_values", sum_values)

        pred_clean_tensor = torch.from_numpy(pred_clean).cuda()

        # 创建一个全为 False 的布尔张量，形状与 pred_clean 相同
        extended_correct_indices_mask = torch.zeros_like(pred_clean_tensor, dtype=torch.bool).cuda()

        # 将 correct_indices_mask 的值更新到 pred_uncertain_indices 指示的位置
        extended_correct_indices_mask[pred_uncertain_indices] = torch.tensor(correct_indices_mask).cuda()

        # # 现在可以将 pred_clean 和 extended_correct_indices_mask 合并
        # combined_mask = pred_clean | extended_correct_indices_mask.cpu().numpy()
        combined_mask = extended_correct_indices_mask.cpu().numpy()

        return combined_mask
    else:
        # 创建一个全为 False 的布尔张量，形状与 pred_clean 相同
        pred_clean_mask = torch.from_numpy(pred_clean).cuda()
        mask = torch.zeros_like(pred_clean_mask, dtype=torch.bool).cuda()
        return mask.cpu().numpy()

    ## Choose Warmup period based on Dataset
num_samples = 50000
if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30

## Call the dataloader
loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=4,\
    root_dir=model_save_loc,log=stats_log, noise_file='%s/clean_%.4f_%s.npz'%(args.data_path,args.r, args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True


## Optimizer and Scheduler
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, 280, 2e-4)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, 280, 2e-4)

## Loss Functions
CEloss   = nn.CrossEntropyLoss()
CE       = nn.CrossEntropyLoss(reduction='none')
criterion_triplet = nn.MarginRankingLoss(margin=args.margin)
contrastive_criterion = SupConLoss()

if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

## Resume from the warmup checkpoint 
model_name_1 = 'Net1_warmup.pth'
model_name_2 = 'Net2_warmup.pth'    

if args.resume:
    start_epoch = warm_up
    net1.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_1))['net'])
    net2.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_2))['net'])
else:
    start_epoch = 0


acc_hist = []
best_acc = 0
all_loss = [[],[]] # save the history of losses from two networks

classwise_acc_net1 = torch.zeros((args.num_class,)).cuda()
classwise_acc_net2 = torch.zeros((args.num_class,)).cuda()

support_vectors_mean = torch.zeros(args.num_class, 128).cuda()

## Warmup and SSL-Training 
for epoch in range(start_epoch,args.num_epochs+1):

    test_loader = loader.run(0, 'test')
    eval_loader = loader.run(0, 'eval_train')
    warmup_trainloader = loader.run(0,'warmup')

    ## Warmup Stage
    if epoch<warm_up:
        warmup_trainloader = loader.run(0, 'warmup')

        print('Warmup Model')
        warmup_standard(epoch, net1, optimizer1, warmup_trainloader)

        print('\nWarmup Model')
        warmup_standard(epoch, net2, optimizer2, warmup_trainloader)


    else:

        prob_gmm1, all_loss[1], cl_features1, labels1 = eval_train_svm(net2, all_loss[1])
        pred1 = get_clean_uncertain_indices(prob_gmm1, cl_features1, labels1, support_vectors_mean, args, epoch)
        threshold1 = np.mean(prob_gmm1)

        prob_gmm2, all_loss[0], cl_features2, labels2 = eval_train_svm(net1, all_loss[0])
        pred2 = get_clean_uncertain_indices(prob_gmm2, cl_features2, labels2, support_vectors_mean, args, epoch)
        threshold2 = np.mean(prob_gmm2)

        SR_gmm1 = np.sum(prob_gmm1 > args.sample_threshold) / num_samples
        SR_gmm2 = np.sum(prob_gmm2 > args.sample_threshold) / num_samples

        # SR_gmm1 = np.sum(prob_gmm1 > threshold1) / num_samples
        # SR_gmm2 = np.sum(prob_gmm2 > threshold2) / num_samples

        print('SR_gmm1\n', SR_gmm1)
        print('SR_gmm2\n', SR_gmm2)

        print('Train Net1\n')

        # GMM
        labeled_trainloader1, unlabeled_trainloader1 = loader.run(mode='train', sample_ratio=SR_gmm1, pred=pred1,
                                                                  prob=prob_gmm1)
        support_vectors_update1 = svmfix_train_flex(args, epoch, net1, net2,
                                                             optimizer1, labeled_trainloader1,
                                                             unlabeled_trainloader1,
                                                             criterion_triplet,
                                                             classwise_acc_net1,
                                                             contrastive_criterion, support_vectors_mean)    # train net1


        print('\nTrain Net2')
        labeled_trainloader2, unlabeled_trainloader2 = loader.run(mode='train', sample_ratio=SR_gmm2, pred=pred2,
                                                                  prob=prob_gmm2)

        support_vectors_update2 = svmfix_train_flex(args, epoch, net2, net1,
                                                     optimizer2, labeled_trainloader2,
                                                     unlabeled_trainloader2,
                                                     criterion_triplet,
                                                     classwise_acc_net2,
                                                     contrastive_criterion, support_vectors_mean)  # train net2

        if epoch == warm_up:
            support_vectors_mean = compute_mean(labeled_trainloader1, net1, support_vectors_mean, args)
            support_vectors_mean = compute_mean(labeled_trainloader2, net2, support_vectors_mean, args)
        # elif epoch > args.epoch_start_svmfix:
        #     support_vectors_mean = update_support_vectors(support_vectors_mean, support_vectors1, support_labels1, update_ratio=0.05)
        #     support_vectors_mean = update_support_vectors(support_vectors_mean, support_vectors2, support_labels2, update_ratio=0.05)
        else:
            support_vectors_mean = (support_vectors_update1 + support_vectors_update2) / 2



    acc = test(epoch,net1,net2)
    # acc_hist.append(acc)
    scheduler1.step()
    scheduler2.step()

    if acc > best_acc:
        if epoch <warm_up:
            model_name_1 = 'Net1_warmup.pth'
            model_name_2 = 'Net2_warmup.pth'
        else:
            model_name_1 = 'Net1.pth'
            model_name_2 = 'Net2.pth'

        print("Save the Model-----")
        checkpoint1 = {
            'net': net1.state_dict(),
            'Model_number': 1,
            'Noise_Ratio': args.r,
            'Loss Function': 'CrossEntropyLoss',
            'Optimizer': 'SGD',
            'Noise_mode': args.noise_mode,
            'Accuracy': acc,
            'Pytorch version': '1.4.0',
            'Dataset': 'TinyImageNet',
            'Batch Size': args.batch_size,
            'epoch': epoch,
        }

        checkpoint2 = {
            'net': net2.state_dict(),
            'Model_number': 2,
            'Noise_Ratio': args.r,
            'Loss Function': 'CrossEntropyLoss',
            'Optimizer': 'SGD',
            'Noise_mode': args.noise_mode,
            'Accuracy': acc,
            'Pytorch version': '1.4.0',
            'Dataset': 'TinyImageNet',
            'Batch Size': args.batch_size,
            'epoch': epoch,
        }

        torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
        torch.save(checkpoint2, os.path.join(model_save_loc, model_name_2))
        best_acc = acc
        print("best_acc", best_acc)

