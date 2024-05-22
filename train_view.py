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
from transfer_loss import TransferLoss

parser = argparse.ArgumentParser("Sketch_View Modality")
# dataset
parser.add_argument('--sketch-datadir', type=str, default='/mnt/baseline/14_sketch_picture')
parser.add_argument('--val-sketch-datadir', type=str, default='/mnt/baseline/14_sketch_picture')
parser.add_argument('--view-datadir', type=str, default='/mnt/baseline/14_view_render_img')
parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 0)")
# optimization
parser.add_argument('--sketch-batch-size', type=int, default=14)
parser.add_argument('--view-batch-size', type=int, default=16)
parser.add_argument('--num-classes', type=int, default=171)
parser.add_argument('--lr-model', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=81)
parser.add_argument('--stepsize', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.9, help="learning rate decay")
parser.add_argument('--feat-dim', type=int, default=2048, help="feature size")
parser.add_argument('--alph', type=float, default=12, help="L2 alph")
# model
parser.add_argument('--model', type=str, choices=['alexnet', 'vgg16', 'vgg19','resnet50','inceptionresnetv2'], default='vgg19')
parser.add_argument('--pretrain', type=bool, choices=[True, False], default=True)
parser.add_argument('--uncer', type=bool, choices=[True, False], default=True)
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
        transforms.Resize(224),
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
        transforms.Resize(224),
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


def train_view(view_model,classifier,criterion_soft,criterion_am,criterion_transfer,class_centroid,optimizer_model,view_dataloader,use_gpu):
    view_model.train()
    classifier.train()

    total = 0.0
    correct = 0.0

    view_size = len(view_dataloader)

    view_dataloader_iter = iter(view_dataloader)

    for batch_idx in range(view_size):

        view = next(view_dataloader_iter)
        view_data,view_labels=view
        view_data = np.stack(view_data, axis=1)
        view_data = torch.from_numpy(view_data)
        if use_gpu:
            view_data,view_labels = view_data.cuda(),view_labels.cuda()

        view_features = view_model.forward(view_data)

        #print(concat_labels)
        #print(concat_feature[4:6])
        #print("___________________________________________")
        """
        if view_labels.shape[0] % 2 ==0:
            index = randint(0,30)
            concat_feature = torch.cat((concat_feature,concat_feature[index].view(1,-1)),dim=0)
            concat_labels = torch.cat((concat_labels,concat_labels[index].view(1,)),dim=0)
        if view_labels.shape[0] % 2 !=0:
            index = randint(0,25)
            concat_feature = torch.cat((concat_feature,concat_feature[index].view(1,-1)),dim=0)
            concat_labels = torch.cat((concat_labels,concat_labels[index].view(1,)),dim=0)
            concat_feature = concat_feature[0:24]
            concat_labels = concat_labels[0:24]"""

        #print(concat_labels.shape)
        #if args.model == 'alexnet':
#         concat_feature = concat_feature.view(-1,2,args.feat_dim).transpose(0, 1).contiguous().view(-1, args.feat_dim)
#         concat_labels = concat_labels.view(-1,2,).transpose(0, 1).contiguous().view(-1,)
        #elif args.model == 'resnet50':
            #concat_feature = concat_feature.view(-1,2,args.feat_dim).transpose(0, 1).contiguous().view(-1, args.feat_dim)
            #concat_labels = concat_labels.view(-1,2,).transpose(0, 1).contiguous().view(-1,)




        feature,logits = classifier.forward(view_features)
        cls_loss = criterion_am(logits, view_labels)
        transfer_loss=criterion_transfer(feature, view_labels,class_centroid)
        loss = transfer_loss


        _, predicted = torch.max(logits.data, 1)
        total += view_labels.size(0)
        correct += (predicted == view_labels).sum()
        avg_acc = correct.item() / total

        optimizer_model.zero_grad()
        loss.backward()

        optimizer_model.step()

        if (batch_idx + 1) % args.print_freq == 0:
            print("Iter [%d/%d] Total Loss: %.4f" % (batch_idx + 1, view_size, loss.item()))
#             print("\tAverage Accuracy: %.4f" % (avg_acc))

        args.count += 1

        writer.add_scalar("Loss", loss.item(), args.count)
#         writer.add_scalar("average accuracy", avg_acc, args.count)
    
    return avg_acc

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    best_acc=0

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")



    print("Creating model: {}".format(args.model))

    view_model = MVCNN(args.model,args.num_classes)
    view_model.cuda()

    #if args.model == 'alexnet':
    classifier = Classifier(args.alph, args.feat_dim, args.num_classes)
    classifier.cuda()
    classifier1 = torch.load(args.model_dir + '/'  +args.model+ '_best_baseline_sketch_classifier'  + '.pth')
    class_centroid = nn.functional.normalize(classifier1["module.fc5.weight"], dim=0).permute(1,0)
    #elif args.model == 'resnet50':
        #classifier = Classifier(args.alph,args.feat_dim, args.num_classes)
        #classifier.cuda()

    ignored_keys = ["L2Classifier.fc2","L2Classifier.fc4"]
    if use_gpu:
        view_model = nn.DataParallel(view_model).cuda()
        classifier = nn.DataParallel(classifier).cuda()

    #sketch_model.load_state_dict(torch.load(args.model_dir + '/' + 'softmax_sketch_model' + '_' +str(70) +  '.pth'))
    #view_model.load_state_dict(torch.load(args.model_dir + '/' + 'softmax_view_model' + '_' + str(70) + '.pth'))
    #classifier.load_state_dict(torch.load(args.model_dir + '/' + 'softmax_classifier' + '_' + str(70) + '.pth'))
    #classifier = torch.load(args.model_dir + '/' + 'softmax_classifier' + '_' + str(70) + '.pth')
    #state_dict = {k: v for k, v in classifier.items() if k in classifier.keys()}



    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    #print(pretrained_dict)
    # Cross Entropy Loss and Center Loss
    criterion_am = AMSoftMaxLoss()
    criterion_transfer= TransferLoss()
    criterion_soft = nn.CrossEntropyLoss()
    optimizer_model = torch.optim.SGD([{"params":view_model.parameters()},
                                       {"params":classifier.parameters(),"lr":args.lr_model*10}],
                                      lr=args.lr_model, momentum=0.9, weight_decay=1e-4)

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)
        #scheduler = lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=args.stepsize, eta_min=0, last_epoch=-1)

    sketch_trainloader,val_sketch_dataloader, view_trainloader = get_data(args.sketch_datadir,args.val_sketch_datadir, args.view_datadir)
    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        print("++++++++++++++++++++++++++")
        # save model

        avg_acc=train_view(view_model,classifier,criterion_soft,criterion_am,criterion_transfer,class_centroid,optimizer_model,view_trainloader,use_gpu)
        
        if epoch>60 and epoch % args.save_model_freq == 0:
            torch.save(view_model.state_dict(),
                       args.model_dir + '/' + args.model+'_baseline_view_model' + '_'+str(epoch)  + '.pth')
            torch.save(classifier.state_dict(),
                       args.model_dir + '/' + args.model+'_baseline_view_classifier' + '_'+str(epoch)  + '.pth')

        if args.stepsize > 0: scheduler.step()
    writer.close()


if __name__ == '__main__':
    main()