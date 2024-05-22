# -*- coding: utf-8 -*-
import os
import argparse
from random import sample,randint

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from sketch_model import SketchModel
from view_model import MVCNN
from classifier import Classifier
from view_dataset_reader import MultiViewDataSet
from am_softmax import AMSoftMaxLoss
from focal_am_loss import FocalAMSoftMaxLoss
from center_loss import CenterLoss

parser = argparse.ArgumentParser("Sketch_View Modality")
# dataset
parser.add_argument('--sketch-datadir', type=str, default='E:\\3d_retrieval\\Dataset\\Shrec_13\\12_views\\13_sketch_train_picture')
parser.add_argument('--val-sketch-datadir', type=str, default='E:\\3d_retrieval\\Dataset\\Shrec_13\\12_views\\13_sketch_test_picture')
parser.add_argument('--view-datadir', type=str, default='E:\\3d_retrieval\\Dataset\\Shrec_13\\12_views\\13_view_render_img')
parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 0)")
# optimization
parser.add_argument('--sketch-batch-size', type=int, default=2)
parser.add_argument('--view-batch-size', type=int, default=16)
parser.add_argument('--num-classes', type=int, default=90)
parser.add_argument('--lr-model', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=305)
parser.add_argument('--stepsize', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.9, help="learning rate decay")
parser.add_argument('--feat-dim', type=int, default=4096, help="feature size")
parser.add_argument('--alph', type=float, default=12, help="L2 alph")
# model
parser.add_argument('--model', type=str, choices=['alexnet', 'vgg16', 'vgg19','resnet50','inceptionresnetv2'], default='alexnet')
parser.add_argument('--pretrain', type=bool, choices=[True, False], default=True)
parser.add_argument('--uncer', type=bool, choices=[True, False], default=False)
# misc
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--save-model-freq', type=int, default=10)
parser.add_argument('--gpu', type=str, default='0,1,2,3')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model-dir', type=str,default='/data/david/sketch_uncertainty/RES')
parser.add_argument('--count', type=int, default=0)

args = parser.parse_args()
writer = SummaryWriter()


def get_data(sketch_datadir,val_sketch_datadir,view_datadir):
    """Image reading and image augmentation
       Args:
         traindir: path of the traing picture
    """
    image_transforms = transforms.Compose([
        transforms.Resize(299),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),  # Randomly change the brightness, contrast, and saturation of the image
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomCrop(224),
        # transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])])  # Imagenet standards

    val_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])])

    view_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])])

    val_sketch_data=datasets.ImageFolder(root=val_sketch_datadir, transform=val_transform)
    val_sketch_dataloaders=DataLoader(val_sketch_data,batch_size=args.sketch_batch_size,num_workers=args.workers)

    sketch_data = datasets.ImageFolder(root=sketch_datadir, transform=image_transforms)
    sketch_dataloaders = DataLoader(sketch_data, batch_size=args.sketch_batch_size, shuffle=True, num_workers=args.workers)

    view_data = MultiViewDataSet(view_datadir, transform=view_transform)
    view_dataloaders = DataLoader(view_data, batch_size=args.view_batch_size, shuffle=True, num_workers=args.workers)

    return sketch_dataloaders,val_sketch_dataloaders,view_dataloaders

def val(sketch_model,classifier,val_sketch_dataloader,use_gpu):
    with torch.no_grad():
        sketch_model.eval()
        classifier.eval()
        sketch_size = len(val_sketch_dataloader)
        sketch_dataloader_iter = iter(val_sketch_dataloader)
        total = 0.0
        correct = 0.0
        for batch_idx in range(sketch_size):
            sketch = next(sketch_dataloader_iter)
            sketch_data, sketch_labels = sketch
            if use_gpu:
                sketch_data, sketch_labels = sketch_data.cuda(), sketch_labels.cuda()

            sketch_features = sketch_model.forward(sketch_data)
            _, logits = classifier.forward(sketch_features)
            _, predicted = torch.max(logits.data, 1)
            total += sketch_labels.size(0)
            correct += (predicted == sketch_labels).sum()

        val_acc = correct.item() / total
        return val_acc

def train_sketch(sketch_model, classifier, criterion_soft, criterion_am,criterion_transfer,optimizer_model, sketch_dataloader, use_gpu):
    sketch_model.train()
    classifier.train()

    total = 0.0
    correct = 0.0

    sketch_size = len(sketch_dataloader)

    sketch_dataloader_iter = iter(sketch_dataloader)

    for batch_idx in range(sketch_size):
        sketch = next(sketch_dataloader_iter)
        sketch_data, sketch_labels = sketch

        if use_gpu:
            sketch_data, sketch_labels = sketch_data.cuda(), sketch_labels.cuda()

        sketch_features = sketch_model.forward(sketch_data)

        concat_feature = sketch_features
        concat_labels = sketch_labels

        feature, logits,weight = classifier.forward(concat_feature)
        cls_loss = criterion_am(logits, concat_labels)
        transfer_loss=criterion_transfer(feature,weight, concat_labels)
        loss = transfer_loss
        return
        _, predicted = torch.max(logits.data, 1)
        total += concat_labels.size(0)
        correct += (predicted == concat_labels).sum()
        avg_acc = correct.item() / total

        optimizer_model.zero_grad()
        loss.backward()

        optimizer_model.step()

        if (batch_idx + 1) % args.print_freq == 0:
            print("Iter [%d/%d] Total Loss: %.4f" % (batch_idx + 1, sketch_size, loss.item()))
            print("\tAverage Accuracy: %.4f" % (avg_acc))

        args.count += 1

        writer.add_scalar("Loss", loss.item(), args.count)
        writer.add_scalar("average accuracy", avg_acc, args.count)

    return avg_acc


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    best_acc = 0.0

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating model: {}".format(args.model))

    sketch_model = SketchModel(args.model, args.num_classes)
    sketch_model.cuda()
    classifier = Classifier(args.alph, args.feat_dim, args.num_classes)
    classifier.cuda()
    
#     classifier1 = torch.load(args.model_dir + '/'  +args.model+ '_baseline_view_classifier_80'  + '.pth')
#     class_centroid = nn.functional.normalize(classifier1["module.fc5.weight"], dim=0).permute(1,0)
    if use_gpu:
        sketch_model = nn.DataParallel(sketch_model).cuda()
        classifier = nn.DataParallel(classifier).cuda()
    
#     sketch_model.load_state_dict(torch.load(args.model_dir + '/' + args.model+'_best_baseline_sketch_model'  +  '.pth'))
#     classifier.load_state_dict(torch.load(args.model_dir + '/' + args.model+ '_best_baseline_sketch_classifier' + '.pth'))
    criterion_am = AMSoftMaxLoss()
    criterion_transfer= CenterLoss(num_classes=args.num_classes,feat_dim=128)
    criterion_soft = nn.CrossEntropyLoss()

    optimizer_model = torch.optim.SGD([{"params": sketch_model.parameters()},
                                       {"params": classifier.parameters(),"lr":args.lr_model*10}],
                                      lr=args.lr_model, momentum=0.9, weight_decay=1e-4)

    if args.stepsize > 0:
#         scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_model, T_0=10, T_mult= 1,eta_min=1e-6, last_epoch=-1)

    sketch_trainloader,val_sketch_dataloader, view_trainloader = get_data(args.sketch_datadir,args.val_sketch_datadir, args.view_datadir)
    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        print("++++++++++++++++++++++++++")
        # save model

        avg_acc = train_sketch(sketch_model, classifier, criterion_soft, criterion_am,criterion_transfer,
                        optimizer_model, sketch_trainloader, use_gpu)
        return
        val_acc=val(sketch_model,classifier,val_sketch_dataloader,use_gpu)
        print("\tVal Accuracy: %.4f" % (val_acc))
        print("\tBest Val Accuracy: %.4f" % (best_acc))
#         if epoch > 60 and epoch % args.save_model_freq == 0:
#             torch.save(sketch_model.state_dict(),
#                        args.model_dir + '/' + args.model + '_baseline_sketch_model' + '_' + str(epoch) + '.pth')
#             torch.save(classifier.state_dict(),
#                        args.model_dir + '/' + args.model + '_baseline_sketch_classifier' + '_' + str(epoch) + '.pth')
        if epoch>20 and val_acc>best_acc:
            best_acc=val_acc
            torch.save(sketch_model.state_dict(),
                       args.model_dir + '/' +args.model+'_best_baseline_sketch_model' + '.pth')
            torch.save(classifier.state_dict(),
                       args.model_dir + '/' +args.model+ '_best_baseline_sketch_classifier' + '.pth')

        if args.stepsize > 0: scheduler.step()
    writer.close()

if __name__ == '__main__':
    main()