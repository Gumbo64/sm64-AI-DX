
import torch.nn as nn
from torchvision import transforms

class EmbeddingModel(nn.Module):
    def __init__(self, embedding_size=64):
        super(EmbeddingModel, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 10 * 10, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_size),
        )

    def forward(self, x):
        return self.convnet(x)
    
MyTransform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((80, 80)),
    transforms.ToTensor(),
])


