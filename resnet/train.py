import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision
import torchvision.models as models

from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor

class ResNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        backbone = models.resnet101(weights="DEFAULT")
        in_features = backbone.fc.in_features
        out_features = 24 #type of chromo
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features,out_features)

    def configure_optimizers(self):
        optimizer_StepLR = torch.optim.SGD(self.parameters(),lr=0.005)
        step_lr = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_StepLR,step_size=10)
        return {'optimizer':optimizer_StepLR,
                'lr_scheduler': step_lr}
                                          
    
    def train_dataloader(self):
        transform = transforms.Compose([transforms.Resize(224,224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        dataset = torchvision.datasets.ImageFolder(root='/home/kemosheng/zera/vfa/data/train',
                                                   transform=transform
                                                   )
        train_loader = DataLoader(dataset=dataset,batch_size=64,shuffle=True)
        return train_loader
    
    def training_step(self,batch,batch_idx):
        x,y = batch
        preds = self.forward(x)
        loss = F.cross_entropy(preds,y)
        acc = (preds.argmax(dim=-1) == y).float().mean()

        self.log('train_loss',loss)
        self.log('train_acc',acc,on_step=False, on_epoch=True)
        return loss    
    
    def val_dataloader(self):
        transform = transforms.Compose([transforms.Resize(224,224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        dataset = torchvision.datasets.ImageFolder(root='/home/kemosheng/zera/vfa/data/val',
                                                    transform=transform
                                                    )
        val_loader = DataLoader(dataset=dataset,batch_size=64,shuffle=False)
        return val_loader
    
    def validation_step(self,batch,batch_idx):
        x,y = batch
        preds = self.forward(x).argmax(dim=-1)
        acc = (y == preds).float().mean()
        self.log('val_acc',acc) 

    def forward(self,x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representation = self.feature_extractor(x).flatten(1)
            # test = self.feature_extractor(x)    #test shape torch.Size([6, 2048, 1, 1])
        x = self.classifier(representation)
        return x

if __name__ == '__main__':
    print('start'+'\n')

    trainer = pl.Trainer(default_root_dir='/home/kemosheng/zera/vfa/resnet/checkpoint',
                         accelerator='gpu',devices=1,max_epochs=30,
                         callbacks=[ModelCheckpoint(mode='max',monitor='val_acc'),LearningRateMonitor('epoch')],
                         fast_dev_run=True)

    net = ResNet()
    trainer.fit(model=net)