import os
import math
import argparse
import sys
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append('vfa/3.28.pretrain_vfa_backbone_res_swin/model')
from model.res101 import resnet101
from model.swin_v2_t import swin_v2_t

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    tb_writer = SummaryWriter(log_dir='runs/chromo_classification')
    if os.path.exists('./weights') is False:
        os.makedirs('./weights')

    data_transform = {
    'train':transforms.Compose([transforms.CenterCrop(120),
                                transforms.Resize((224,224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.99302036, 0.99302036, 0.99302036], std=[0.05381432, 0.05381432, 0.05381432])
                                ]),
    'val':transforms.Compose([
        transforms.CenterCrop(120),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.99302036, 0.99302036, 0.99302036], std=[0.05381432, 0.05381432, 0.05381432])

    ])}

    image_path = args.data_path

    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(image_path,'train'),transform=data_transform['train'])
    val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(image_path,'val'),transform=data_transform['val'])

    batch_size = args.batch_size
    num_worker = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(num_worker))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_worker)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,num_workers=num_worker)

    chromo_lst = train_dataset.class_to_idx

    cla_dict = dict((val,key) for key,val in chromo_lst.items())
    json_str = json.dumps(cla_dict,indent=4)

    with open('class_indices.json','w') as js:
        js.write(json_str)

    #

    if args.model == 0:
        model = resnet101()
    else:
        model = swin_v2_t()
    model.to(device)

    init_img = torch.zeros((1,3,224,224),device=device)
    tb_writer.add_graph(model,init_img)

    if os.path.exists(args.weights):
        weights_dict = torch.load(args.weights,map_location=device)
        load_weight = {k:v for k,v in weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weight,strict=False)
    else:
        print('not using pretrained-weights')

    if args.freeze_layers:
        print('freeze layers except fc layer')
        for name,parameter in model.named_parameters():
            if 'fc' not in name:
                parameter.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params,lr=args.lr,momentum=0.9,weight_decay=0.005)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0
    for epoch in range(args.epochs):
        #train
        model.train()
        loss = nn.CrossEntropyLoss()
        mean_loss = torch.zeros(1).to(device)
        optimizer.zero_grad()

        data_loader = tqdm(train_loader,file=sys.stdout)

        for step,train_data in enumerate(data_loader):
            train_img,train_label = train_data

            preds = model(train_img.to(device))
            l = loss(preds,train_label.to(device))
            l.backward()
            mean_loss = (mean_loss * step + l.detach()) / (step + 1)

            data_loader.desc = '[epoch {}] mean loss {}'.format(epoch, round(mean_loss.item(), 3))

            optimizer.step()
            optimizer.zero_grad()

        mean_loss = mean_loss.item()

        scheduler.step()

        #val
        model.eval()
        sum_num = torch.zeros(1).to(device)
        num_samples = len(val_loader.dataset)

        data_loader = tqdm(val_loader,desc='validation..',file=sys.stdout)
        for step,val_data in enumerate(data_loader):
            val_imgs,val_labels = val_data
            preds_val = model(val_imgs.to(device))
            preds_val = torch.max(preds_val,dim=1)[1]
            sum_num += torch.eq(preds_val,val_labels.to(device)).sum()

        acc = sum_num.item() / num_samples

        #add loss
        print('[epoch {}] accuracy: {} '.format(epoch+1,round(acc,3)))
        tags = ["train_loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(),'./weights/{}-{}.pth'.format(args.model,epoch+1))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-b','--batch_size',type=int,default=128)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--lrf',type=float,default=0.1)
    parser.add_argument('--weights', type=str, 
    default='./weights/model_base.pth',
                        help='initial weights path')
    parser.add_argument('-dp','--data_path',type=str,default='/home/kemosheng/code/zxx/vfa/data')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('-m','--model',type=int,default=0,help='0 res101,1 swin_v2_t')


    opt = parser.parse_args()

    main(opt)