import argparse
import copy
import math
import os
import os.path as osp
import pdb
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

import loss
import network
from data_list import ImageList, get_dataset_root
from loss import CrossEntropyLabelSmooth


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def data_load(args):
    ## prepare data
    dataset_root = get_dataset_root(args.dset)
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size

    if not isinstance(args.t, list):
        args.t = [args.t]

    if args.dset == "domainnet":
        txt_src = open(args.s_dset_path).readlines()
        te_txt = open(args.v_dset_path).readlines()
        all_txt_test = []
        for target_domain in args.t:  # args.t now contains a list of target domains
            test_dset_path = os.path.join(
                folder, args.dset, names[int(target_domain)] + "_test.txt"
            )
            txt_test = open(test_dset_path).readlines()
            all_txt_test.extend(txt_test)  # Aggregate data from all target domains

        txt_src = [folder + args.dset + "/" + s for s in txt_src]
        te_txt = [folder + args.dset + "/" + s for s in te_txt]
        all_txt_test = [folder + args.dset + "/" + s for s in all_txt_test]

        if args.trte == "val":
            tr_txt = txt_src
        else:
            tr_txt = txt_src
            te_txt = txt_src

        dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
        dset_loaders["source_tr"] = DataLoader(
            dsets["source_tr"],
            batch_size=train_bs,
            shuffle=True,
            num_workers=args.worker,
            drop_last=False,
        )
        dsets["source_te"] = ImageList(te_txt, transform=image_test())
        dset_loaders["source_te"] = DataLoader(
            dsets["source_te"],
            batch_size=train_bs,
            shuffle=True,
            num_workers=args.worker,
            drop_last=False,
        )
        dsets["test"] = ImageList(all_txt_test, transform=image_test())
        dset_loaders["test"] = DataLoader(
            dsets["test"],
            batch_size=train_bs * 2,
            shuffle=True,
            num_workers=args.worker,
            drop_last=False,
        )
    else:
        txt_src = open(args.s_dset_path).readlines()  # source domain의 list읽는 부분

        # txt_test = open(args.test_dset_path).readlines()
        all_txt_test = []
        for target_domain in args.t:  # args.t now contains a list of target domains
            test_dset_path = os.path.join(
                folder, args.dset, names[int(target_domain)] + "_list.txt"
            )
            txt_test = open(test_dset_path).readlines()
            all_txt_test.extend(txt_test)  # Aggregate data from all target domains

        # txt_src = [folder + args.dset + "/" + s for s in txt_src]
        # txt_test = [folder + args.dset + "/" + s for s in txt_test]

        if args.trte == "val":
            dsize = len(txt_src)
            tr_size = int(0.9 * dsize)
            tr_txt, te_txt = torch.utils.data.random_split(
                txt_src, [tr_size, dsize - tr_size]
            )
        else:
            te_txt = txt_src

        dsets["source_tr"] = ImageList(tr_txt, transform=image_train(), dataset_root=dataset_root)
        dset_loaders["source_tr"] = DataLoader(
            dsets["source_tr"],
            batch_size=train_bs,
            shuffle=True,
            num_workers=args.worker,
            drop_last=False,
        )
        dsets["source_te"] = ImageList(te_txt, transform=image_test(), dataset_root=dataset_root)
        dset_loaders["source_te"] = DataLoader(
            dsets["source_te"],
            batch_size=train_bs,
            shuffle=True,
            num_workers=args.worker,
            drop_last=False,
        )
        dsets["test"] = ImageList(all_txt_test, transform=image_test(), dataset_root=dataset_root)
        dset_loaders["test"] = DataLoader(
            dsets["test"],
            batch_size=train_bs * 2,
            shuffle=True,
            num_workers=args.worker,
            drop_last=False,
        )

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )
    mean_ent = torch.mean(Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = np.divide(matrix.diagonal(), matrix.sum(axis=1) + np.finfo(float).eps) * 100
        aacc = np.nanmean(acc)  # Use nanmean to ignore NaN values
        aa = [str(np.round(i, 2)) for i in acc]
        acc = " ".join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def train_source(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == "res":
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:4] == "conv":
        netF = network.ConvNeXtBase(conv_name=args.net).cuda()

    netB = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{"params": v, "lr": learning_rate * 0.1}]
    for k, v in netB.named_parameters():
        param_group += [{"params": v, "lr": learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{"params": v, "lr": learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    smax = 100
    # while iter_num < max_iter:
    for epoch in range(args.max_epoch):
        # print(f'epoch:{epoch}')
        iter_source = iter(dset_loaders["source_tr"])
        for batch_idx, (inputs_source, labels_source) in enumerate(iter_source):

            if inputs_source.size(0) == 1:
                continue

            iter_num += 1
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            feature_src = netB(netF(inputs_source))

            outputs_source = netC(feature_src)
            classifier_loss = CrossEntropyLabelSmooth(
                num_classes=args.class_num, epsilon=args.smooth
            )(outputs_source, labels_source)

            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset == "visda-2017":
                acc_s_te, acc_list = cal_acc(
                    dset_loaders["source_te"], netF, netB, netC, flag=True
                )
                log_str = (
                    "Task: {}, Iter:{}/{}; Accuracy = {:.2f}%".format(
                        args.name_src, iter_num, max_iter, acc_s_te
                    )
                    + "\n"
                    + acc_list
                )
            else:
                acc_s_te, acc_list = cal_acc(
                    dset_loaders["source_te"], netF, netB, netC, flag=True
                )
                log_str = (
                    "Task: {}, Iter:{}/{}; Accuracy = {:.2f}%".format(
                        args.name_src, iter_num, max_iter, acc_s_te
                    )
                    + "\n"
                    + acc_list
                )
            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str + "\n")

            # if acc_s_te >= acc_init:
            #     acc_init = acc_s_te
            #     best_netF = netF.state_dict()
            #     best_netB = netB.state_dict()
            #     best_netC = netC.state_dict()

            netF.train()
            netB.train()
            netC.train()

    netF.eval()
    netB.eval()
    netC.eval()
    acc_s_te, acc_list = cal_acc(dset_loaders["test"], netF, netB, netC, flag=True)

    log_str = (
        "Task: {}; Accuracy on target = {:.2f}%".format(args.name_src, acc_s_te)
        + "\n"
        + acc_list
    )
    args.out_file.write(log_str + "\n")
    args.out_file.flush()
    print(log_str + "\n")

    # torch.save(best_netF, osp.join(args.output_dir_src, f"source_F.pt"))
    # torch.save(best_netB, osp.join(args.output_dir_src, f"source_B.pt"))
    # torch.save(best_netC, osp.join(args.output_dir_src, f"source_C.pt"))

    torch.save(netF.state_dict(), osp.join(args.output_dir_src, f"source_F.pt"))
    torch.save(netB.state_dict(), osp.join(args.output_dir_src, f"source_B.pt"))
    torch.save(netC.state_dict(), osp.join(args.output_dir_src, f"source_C.pt"))

    return netF, netB, netC


def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == "res":
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:4] == "conv":
        netF = network.ConvNeXtBase(conv_name=args.net).cuda()

    netB = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()

    args.modelpath = args.output_dir_src + f"/source_F.pt"
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + f"/source_B.pt"
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + f"/source_C.pt"
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, acc_list = cal_acc(dset_loaders["test"], netF, netB, netC, flag=True)
    log_str = (
        "\nTraining: {}, Task: {}, Accuracy = {:.2f}%".format(args.trte, args.name, acc)
        + "\n"
        + acc_list
    )

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neighbors")
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="2", help="device id to run"
    )
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=str, nargs="+", help="target domains")
    parser.add_argument("--max_epoch", type=int, default=15, help="max iterations")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument("--dset", type=str, default="office-home")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--net", type=str, default="resnet50")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--smooth", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="weight1/source/")
    parser.add_argument("--da", type=str, default="uda")
    parser.add_argument("--trte", type=str, default="val", choices=["full", "val"])
    args = parser.parse_args()

    if args.dset == "office-home":
        names = ["Art", "Clipart", "Product", "Real_World"]
        args.class_num = 65
    if args.dset == "visda-2017":
        names = ["train", "validation"]
        args.class_num = 12
    if args.dset == "domainnet":
        names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        args.class_num = 345
    if args.dset == "domainnet-126":
        names = ["clipart", "painting", "real", "sketch"]
        args.class_num = 126

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = "./data/"
    if args.dset == "office-home":
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_list.txt"
    elif args.dset == "domainnet-126":
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_list.txt"
    elif args.dset == "domainnet":
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_train.txt"
        args.v_dset_path = folder + args.dset + "/" + names[args.s] + "_test.txt"

    args.output_dir_src = osp.join(
        args.output, args.da, args.dset, names[args.s][0].upper(), str(args.seed)
    )
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system("mkdir -p " + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, "log.txt"), "w")
    args.out_file.write(print_args(args) + "\n")
    args.out_file.flush()
    train_source(args)

    args.out_file = open(osp.join(args.output_dir_src, "log_test.txt"), "w")
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        args.name = names[args.s][0].upper() + names[args.t][0].upper()
        test_target(args)
