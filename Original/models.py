import torch
import torch.nn as nn
import torch.nn.functional as F

class Teacher(nn.Module):
    def __init__(self):
        super(Teacher,self).__init__()

        self.fc1 = nn.Linear(28*28,1200)
        self.fc2 = nn.Linear(1200,1200)
        self.fc3 = nn.Linear(1200,10)
        self.dropout = nn.Dropout(0.8)

    def forward(self,x):
        out = x.view(-1,28*28)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

class Student(nn.Module):
    def __init__(self,hidden_size):
        super(Student,self).__init__()

        self.fc1 = nn.Linear(28*28,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,10)

    def forward(self,x):
        out = x.view(-1,28*28)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out
