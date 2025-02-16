import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from triplet_loader import TripletImageLoader
from triplet_model import EmbeddingModel, MyTransform


# Initialize the model, loss function, and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EmbeddingModel().to(device)
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

transform = MyTransform
dataset = TripletImageLoader('./data', transform=transform)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=16)

# Training loop
num_epochs = 20
for epoch in tqdm(range(num_epochs), desc='Epochs', unit='epoch'):
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader, 0), desc='Batches', unit='batch', total=len(dataloader)):
        img1, img2, img3 = data
        img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)

        optimizer.zero_grad()
        
        output1 = model(img1)
        output2 = model(img2)
        output3 = model(img3)

        loss = criterion(output1, output2, output3)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 10 == 9:    # print every 10 mini-batches
            tqdm.write(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.3f}')
            running_loss = 0.0

torch.save(model.state_dict(), 'triplet.pth')

print('Finished Training')
