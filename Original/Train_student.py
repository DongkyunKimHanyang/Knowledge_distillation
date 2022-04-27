import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from models import Teacher, Student

import optuna
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_ds = datasets.MNIST('./data/',train=True,transform=transform,download=True)
    test_ds = datasets.MNIST('./data/', train=False,transform=transform,download=True)

    train_batch = DataLoader(train_ds,batch_size=256,shuffle=True)
    test_batch = DataLoader(test_ds, batch_size=256, shuffle=False)

    def objective(trial):
        student_hidden_size = trial.suggest_int("student_hidden_size",0,300,step=100)
        T = trial.suggest_int("T",2,10,step=2)
        calc_KD_loss = trial.suggest_categorical("calc_KD_loss",[False,True])

        teacher = Teacher().to(device)
        teacher.load_state_dict(torch.load('./result/teacher.pt'))
        teacher.eval()

        student = Student(student_hidden_size).to(device)

        CE_loss_fn = nn.CrossEntropyLoss()

        optimizer = optim.Adam(student.parameters())

        for epoch in range(1,21):
            student.train()
            train_loss = 0
            acc = 0
            for i,batch in enumerate(train_batch):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                teacher_logits = teacher(images).detach()
                student_logits = student(images)


                if calc_KD_loss == True:
                    # 논문에서는 2번째 텀에 T^2를 붙이하고 헀지만 안붙이는게 실험 상 더 좋았다
                    loss = 0.5* CE_loss_fn(student_logits, labels) + 0.5 *(-F.log_softmax(student_logits/T,-1) * F.softmax(teacher_logits/T,-1)).sum(1).mean()
                elif calc_KD_loss == False:
                    loss = CE_loss_fn(student_logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss = train_loss + (loss.item() - train_loss) / (i + 1)


        with torch.no_grad():
            student.eval()
            val_loss = 0
            val_acc = 0
            for i,batch in enumerate(test_batch):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                logits = student(images)

                val_acc = val_acc + ((logits.argmax(-1) == labels).sum().item()/len(images) - val_acc) / (i + 1)


        return val_acc


    search_space = {"student_hidden_size": [32,300],"T":[1,2,5,10],"calc_KD_loss":[False,True]}
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(objective,n_trials=16)
    df = study.trials_dataframe()
    df.to_csv('./result/tuning_df.csv')