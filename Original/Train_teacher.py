import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from models import Teacher

from tqdm import tqdm

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_ds = datasets.MNIST('./data/',train=True,transform=transform,download=True)
    test_ds = datasets.MNIST('./data/', train=False,transform=transform,download=True)

    train_batch = DataLoader(train_ds,batch_size=256,shuffle=True)
    test_batch = DataLoader(test_ds, batch_size=256, shuffle=False)

    teacher = Teacher().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters())

    for epoch in range(1,11):
        train_progress = tqdm(train_batch)
        teacher.train()
        train_loss = 0
        acc = 0
        for i,batch in enumerate(train_progress):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            logits = teacher(images)
            loss = loss_fn(logits,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + (loss.item() - train_loss) / (i + 1)
            acc = acc + ((logits.argmax(-1)==labels).sum().data.cpu().numpy() /len(images)  - acc) / (i + 1)

            train_progress.set_description(f"Epoch {epoch}: Train_loss {loss:.6f} Train_Accuracy {acc*100:.2f}%")


        with torch.no_grad():
            teacher.eval()
            test_progress = tqdm(test_batch)
            val_loss = 0
            val_acc = 0
            for i,batch in enumerate(test_progress):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                logits = teacher(images)
                loss = loss_fn(logits,labels)

                val_loss = val_loss + (loss.item() - val_loss) / (i + 1)
                val_acc = val_acc + ((logits.argmax(-1) == labels).sum().item()/len(images) - val_acc) / (i + 1)
            print(f"Epoch {epoch}: Test_loss {val_loss:.6f} Test_Accuracy {val_acc*100:.2f}%")


    torch.save(teacher.state_dict(),'./result/teacher.pt')






