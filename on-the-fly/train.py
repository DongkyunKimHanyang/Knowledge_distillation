import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import ONE_model, Resnet_110_model

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import argparse

def training(args,train_loader,test_loader,device, num_classes):
    if args.model_name == "baseline":
        model = Resnet_110_model(num_classes).to(device)
    if args.model_name == "ONE":
        model = ONE_model(num_classes).to(device)
    cls_loss_fn = nn.CrossEntropyLoss()
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    error_hist =[]
    for epoch in range(1, 301):
        if epoch == 150:
            optimizer.param_groups[0]['lr'] = 0.01
        if epoch == 225:
            optimizer.param_groups[0]['lr'] = 0.001

        train_progress_bar = tqdm(train_loader)
        model.train()
        for i,batch in enumerate(train_progress_bar):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            branch_logits, pred_logits = model(images)
            if args.model_name == "ONE":
                cls_loss = cls_loss_fn(branch_logits[:,0,:],labels) + cls_loss_fn(branch_logits[:,1,:],labels) + cls_loss_fn(branch_logits[:,2,:],labels) + cls_loss_fn(pred_logits,labels)
                student_logits = torch.log_softmax(branch_logits / 3,-1)
                teacher_logits = torch.softmax(pred_logits / 3, -1).detach()
                kl_loss = kl_loss_fn(student_logits[:,0,:],teacher_logits) + kl_loss_fn(student_logits[:,1,:],teacher_logits) + kl_loss_fn(student_logits[:,2,:],teacher_logits)
                total_loss = cls_loss + 9 * kl_loss
                train_progress_bar.set_description(f"Epoch {epoch}: Total_loss {total_loss:.6f} Cls_loss {cls_loss:.6f}, KL_loss {9 * kl_loss:.6f}")
            elif args.model_name == "baseline":
                total_loss = cls_loss_fn(pred_logits,labels)
                train_progress_bar.set_description(f"Epoch {epoch}: Total_loss {total_loss:.6f}")
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            test_progress_bar = tqdm(test_loader)
            correct = 0
            count = 0
            for i, batch in enumerate(test_progress_bar):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                branch_logits, pred_logits = model(images)
                y_hat = pred_logits.argmax(-1)
                correct += (y_hat == labels).sum().data.cpu().numpy()
                count += labels.size(0)

            error = 100 - (correct / count) * 100
            error_hist.append(error)
            print(f"Epoch {epoch}: Test top1 error {error}")

    plt.plot(np.array(error_hist))
    plt.savefig(f'./error_hist_{args.model_name}.png')
    np.save(f'./error_hist_{args.model_name}.npy',np.array(error_hist))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type =str, default="ONE", choices=["baseline","ONE"])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(32,padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    trainset = torchvision.datasets.CIFAR100(root='./data',train=True,download=True,transform=train_transforms)
    testset = torchvision.datasets.CIFAR100(root='./data',train=False,download=True,transform=test_transforms)

    train_loader = data.DataLoader(trainset,batch_size=128, shuffle=True, num_workers=0)
    test_loader = data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

    training(args,train_loader,test_loader,device,100)
