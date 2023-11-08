import sys
import torch
import numpy as np
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
import torch.nn as nn
from random import sample
from collections import Counter
import random
from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def cal_support_vectors_svm(con_x_w2,label_x):

    con_x_w2_sv = con_x_w2.cpu().numpy()
    label_x_sv = label_x.long().cpu().numpy()
    svm_model = SVC(kernel='rbf')

    svm_model.fit(con_x_w2_sv, label_x_sv)

    # 获取每个类别的支持向量的数量
    n_support = svm_model.n_support_


    # 获取所有类别的标签
    classes = svm_model.classes_

    # 获取支持向量的索引
    sv_indices = svm_model.support_
    # 获取所有支持向量
    support_vectors = con_x_w2[sv_indices]

    # 获取每个类别的拉格朗日乘子值
    dual_coefficients = np.abs(svm_model.dual_coef_)
    sup_dual = np.mean(dual_coefficients, axis=0)
    confidence_max_indices = []


    # 获取每个类别的支持向量的平均特征向量以及对应的标签
    support_vectors_by_class = []
    mean_support_vectors = []
    support_labels = []
    for i in range(len(classes)):
        start = sum(n_support[:i])
        end = start + n_support[i]
        # support_vectors_by_class.append(support_vectors[start:end])
        # mean_support_vectors.append(torch.mean(support_vectors[start:end], axis=0).unsqueeze(0))
        support_labels.append(classes[i])

        class_sup_dual = sup_dual[start:end]  # 选择对应类别的置信度值
        # 找到置信度最大的支持向量的索引
        max_index = np.argmax(class_sup_dual)

        # 将索引保存到结果列表中
        confidence_max_indices.append(sv_indices[start:end][max_index].item())
        max_indice = sv_indices[start:end][max_index].item()
        mean_support_vectors.append((con_x_w2[max_indice]).unsqueeze(0))

    mean_support_vectors = torch.cat(mean_support_vectors)
    mean_support_vectors = nn.functional.normalize(mean_support_vectors, dim=1)

    return mean_support_vectors, support_labels, confidence_max_indices

def cal_svcl_loss_all(support_vectors, support_labels, pred_norm, target_norm, args, labels, criterion_triplet, support_vectors_all):

    n = pred_norm.size(0)
    dist = -torch.matmul(pred_norm, target_norm.t())

    idx = torch.arange(n)
    mask = idx.expand(n, n).eq(idx.expand(n, n).t())

    dist_svcl = -torch.matmul(pred_norm, support_vectors.t())
    with torch.no_grad():
        support_vectors_all = nn.functional.normalize(support_vectors_all, dim=1)
    dist_svcl_all = -torch.matmul(pred_norm, support_vectors_all.t())

    dist_ap, dist_an = [], []
    dist_ap1, dist_an1 = [], []
    for i in range(n):

        # dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))

        pos_label = labels[i]
        if support_vectors.size(0) < args.num_class:
            support_labels = np.array(support_labels)

            if int(pos_label) in support_labels:
                index_pos = np.nonzero(support_labels == int(pos_label))
                dist_ap.append(dist_svcl[i][index_pos])
            else:

                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        else:
            support_labels = np.array(support_labels)
            index_pos = np.nonzero(support_labels == pos_label.item())
            dist_ap.append(dist_svcl[i][index_pos])


        neg_class_ids = set(support_labels)
        if int(pos_label) in neg_class_ids:
            neg_class_ids.remove(int(pos_label))

        if len(neg_class_ids) == 0:

            select_num = random.randint(1, n - 1)
            dist_an.append(dist[i][mask[select_num]].max().unsqueeze(0))
        else:
            neg_class_id = sample(neg_class_ids, 1)
            support_labels_s = np.array(support_labels)
            index_neg = np.nonzero(support_labels_s == neg_class_id[0])
            dist_an.append(dist_svcl[i][index_neg])

        neg_class_id1 = sample(neg_class_ids, 1)
        dist_ap1.append(dist_svcl_all[i][pos_label].unsqueeze(0))
        dist_an1.append(dist_svcl_all[i][neg_class_id1].unsqueeze(0))

    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)
    y = torch.ones_like(dist_an)

    dist_ap1 = torch.cat(dist_ap1)
    dist_an1 = torch.cat(dist_an1)
    y1 = torch.ones_like(dist_an1)
    loss_triplet1 = criterion_triplet(dist_an1, args.gamma * dist_ap1, y1)

    loss_triplet = criterion_triplet(dist_an, args.gamma * dist_ap, y)

    return (loss_triplet + loss_triplet1) / 2.0


def cal_svcl_loss(support_vectors, support_labels, pred_norm, target_norm, args, labels, criterion_triplet):

    n = pred_norm.size(0)
    dist = -torch.matmul(pred_norm, target_norm.t())

    idx = torch.arange(n)
    mask = idx.expand(n, n).eq(idx.expand(n, n).t())

    dist_svcl = -torch.matmul(pred_norm, support_vectors.t())



    dist_ap, dist_an = [], []

    for i in range(n):

        pos_label = labels[i]
        if support_vectors.size(0) < args.num_class:
            support_labels = np.array(support_labels)

            if int(pos_label) in support_labels:
                index_pos = np.nonzero(support_labels == int(pos_label))
                dist_ap.append(dist_svcl[i][index_pos])
            else:

                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        else:
            support_labels = np.array(support_labels)
            index_pos = np.nonzero(support_labels == pos_label.item())
            dist_ap.append(dist_svcl[i][index_pos])


        neg_class_ids = set(support_labels)
        if int(pos_label) in neg_class_ids:
            neg_class_ids.remove(int(pos_label))

        if len(neg_class_ids) == 0:

            select_num = random.randint(1, n - 1)
            dist_an.append(dist[i][mask[select_num]].max().unsqueeze(0))
        else:
            neg_class_id = sample(neg_class_ids, 1)
            support_labels_s = np.array(support_labels)
            index_neg = np.nonzero(support_labels_s == neg_class_id[0])
            dist_an.append(dist_svcl[i][index_neg])


    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)
    y = torch.ones_like(dist_an)

    loss_triplet = criterion_triplet(dist_an, args.gamma * dist_ap, y)

    return loss_triplet



def linear_rampup(args, current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)



def update_support_vectors(support_vectors_mean, support_vectors, support_labels, update_ratio=0.05):
    if support_vectors.size(0) == support_vectors_mean.size(0):
        return support_vectors_mean * (1 - update_ratio) + support_vectors * update_ratio
    else:
        for i in range(len(support_labels)):
            support_vectors_mean[support_labels[i]] = support_vectors_mean[support_labels[i]] \
                                                      * (1 - update_ratio) + support_vectors[i] * update_ratio
        return support_vectors_mean

def svmfix_train_flex(args, epoch, net, net2, optimizer, labeled_trainloader,unlabeled_trainloader, criterion_triplet, classwise_acc, contrastive_criterion, support_vectors_all):

    net2.eval()  # Freeze one network and train the other
    net.train()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1

    ## Loss statistics
    loss_x = 0
    loss_u = 0
    loss_tri = 0
    loss_ucl = 0
    loss_pen = 0

    selected_label = torch.ones((len(unlabeled_trainloader.dataset),), dtype=torch.long, ) * -1
    selected_label = selected_label.cuda()

    selected_label_x = torch.ones((len(labeled_trainloader.dataset),), dtype=torch.long, ) * -1
    selected_label_x = selected_label_x.cuda()


    support_vectors_mean = support_vectors_all

    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x, index_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4, index = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4, index = unlabeled_train_iter.next()

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()


        with torch.no_grad():
            # Label co-guessing of unlabeled samples
            cl_u11, outputs_u11 = net(inputs_u)
            cl_u12, outputs_u12 = net(inputs_u2)
            cl_u21, outputs_u21 = net2(inputs_u)
            cl_u22, outputs_u22 = net2(inputs_u2)

            ## Pseudo-label
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21,
                                                                                                        dim=1) + torch.softmax(
                outputs_u22, dim=1)) / 4
            cl_u = (cl_u11 +cl_u12 + cl_u21 + cl_u22) / 4

            ptu = pu ** (1 / args.T)  ## Temparature Sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

            max_probs, label_u = torch.max(targets_u, dim=-1)


            if epoch >= args.epoch_start_svmfix:
                # 与各类别支持向量的余弦相似度
                cos_similarities = F.cosine_similarity(cl_u.unsqueeze(1),
                                                       support_vectors_mean.unsqueeze(0))

                # 找到最相似的中心向量对应的类别
                most_similar_classes = torch.argmax(cos_similarities, dim=1).to(torch.int)

                # 比较最相似类别与训练样本标签是否相同
                mask_svm = (most_similar_classes == label_u.to(torch.int))

                mask_cls = max_probs.ge(args.flex_threshold).to(torch.int)

                combined_mask = torch.tensor(mask_svm, dtype=bool) | torch.tensor(mask_cls, dtype=bool)

                mask_idx = (combined_mask == 1).nonzero(as_tuple=False).squeeze(1)

                # values = torch.masked_select(combined_mask, combined_mask)

                # 计算选择出的张量的总和
                # sum_values = torch.sum(values)
                # print("sum_values", sum_values)

                # print("sum_values", sum_values)
            else:
                mask = max_probs.ge(args.flex_threshold).float()
                mask_idx = (mask == 1).nonzero(as_tuple=False).squeeze(1)



            select = max_probs.ge(args.flex_threshold).long()
            if index[select == 1].nelement() != 0:
                selected_label[index[select == 1]] = label_u.long()[select == 1]

            # Label refinement
            _, outputs_x = net(inputs_x)
            _, outputs_x2 = net(inputs_x2)

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

            px_new = w_x * labels_x + (1 - w_x) * px
            ptx = px_new ** (1 / args.T)  # Temparature sharpening


            targets_x = ptx / ptx.sum(dim=1, keepdim=True)
            targets_x = targets_x.detach()
            max_probs_x, label_x = torch.max(targets_x, dim=-1)

            select_x = max_probs_x.ge(args.sample_threshold).long()
            if index_x[select_x == 1].nelement() != 0:
                selected_label_x[index_x[select_x == 1]] = label_x.long()[select_x == 1]

        # MixMatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        if len(mask_idx) > 1:
            all_inputs = torch.cat([inputs_x3, inputs_x4, inputs_u3[mask_idx], inputs_u4[mask_idx]], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u[mask_idx], targets_u[mask_idx]], dim=0)
        else:
            all_inputs = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        # Mixup
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        _, logits = net(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        probs_u = torch.softmax(logits_u, dim=1)


        Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_target[:batch_size * 2], dim=1))
        Lu = torch.mean((probs_u - mixed_target[batch_size * 2:]) ** 2)


        # no mixup
        # _, logits = net(all_inputs)
        # logits_x = logits[:batch_size * 2]
        # logits_u = logits[batch_size * 2:]
        #
        # probs_u = torch.softmax(logits_u, dim=1)
        #
        # Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * all_targets[:batch_size * 2], dim=1))
        # Lu = torch.mean((probs_u - all_targets[batch_size * 2:]) ** 2)

        lambda_u = linear_rampup(args, epoch + batch_idx / num_iter, args.warm_up)

        ## Regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        # probs_pen = torch.softmax(logits, dim=1)
        # negpen = 1.5 * torch.mean(torch.sum(probs_pen.log() * probs_pen, dim=1))


        if epoch >= args.epoch_start_svmfix:


            with torch.no_grad():

                con_x_w2, _ = net2(inputs_x)
                support_vectors, support_labels, support_vectors_index = \
                    cal_support_vectors_svm(con_x_w2,label_x)

                support_vectors_mean = update_support_vectors(support_vectors_mean, support_vectors, support_labels)

                if len(mask_idx) > 1:
                    con_u_w2, _ = net2(inputs_u[mask_idx])
                    con_u_w2 = nn.functional.normalize(con_u_w2, dim=1)

                con_x_w2, _ = net2(inputs_x)
                con_x_w2 = nn.functional.normalize(con_x_w2, dim=1)

            con_x_s1, _ = net(inputs_x3)
            con_x_s1 = F.normalize(con_x_s1, dim=1)


            # Lc_s = cal_svcl_loss(support_vectors, support_labels, con_x_s1, con_x_w2, args, label_x, criterion_triplet)
            #
            # Lc_u = cal_svcl_loss(support_vectors, support_labels, con_u_s1, con_u_w2, args, label_u, criterion_triplet)

            Lc_s = cal_svcl_loss_all(support_vectors, support_labels, con_x_s1, con_x_w2, args, label_x, criterion_triplet, support_vectors_mean)

            if len(mask_idx) > 1:
                con_u_s1, _ = net(inputs_u4[mask_idx])
                con_u_s1 = F.normalize(con_u_s1, dim=1)
                Lc_u = cal_svcl_loss_all(support_vectors, support_labels, con_u_s1, con_u_w2, args, label_u, criterion_triplet, support_vectors_mean)

                L_hs = (Lc_s + Lc_u) / 2.0
                # L_hs = Lc_s
            else:
                L_hs = Lc_s

                ## Total Loss
            # loss = Lx + lambda_u * Lu + penalty + (args.lambda_tri * L_hs + args.lambda_c * loss_simCLR) / 2.0
            # loss = Lx + lambda_u * Lu  + args.lambda_tri * L_hs + penalty  # + negpen

            loss = Lx + args.lambda_tri * L_hs + penalty
            loss_tri += L_hs.item()
        else:
            # Unsupervised Contrastive Loss
            f1, _ = net(inputs_u3)
            f2, _ = net(inputs_u4)
            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_simCLR = contrastive_criterion(features)

            # loss = Lx + lambda_u * Lu + args.lambda_c * loss_simCLR + penalty  # + negpen
            loss = Lx + args.lambda_c * loss_simCLR + penalty
            loss_ucl += loss_simCLR.item()
        # Accumulate Loss
        loss_x += Lx.item()
        loss_u += Lu.item()

        # loss_pen += negpen.item()

        # Compute gradient and Do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write(
            '%s:%.2f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Lb loss: %.2f  Ulb loss: %.4f Sim Loss:%.4f svmtri:%.4f'
            % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
               loss_x / (batch_idx + 1), loss_u / (batch_idx + 1), loss_ucl / (batch_idx + 1), loss_tri / (batch_idx + 1)))
        sys.stdout.flush()

    return support_vectors_mean


