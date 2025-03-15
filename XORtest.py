import numpy as np
import torch
from sklearn.datasets import make_swiss_roll
import pygame
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
np.set_printoptions(suppress=True)

data_amount = 15000
dim_amount = 50

pop_amount = 200
survival_amount = 180

D = np.random.randn(data_amount, dim_amount)
D = torch.tensor(D).float()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(dim_amount, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1)),


            # nn.Linear(dim_amount, 64),
            # nn.Tanh(),
            # nn.Linear(64, 64),
            # nn.Tanh(),
            # nn.Linear(64, 1),
        )
    
    def forward(self, x):
        return self.network(x)



population = [network() for _ in range(pop_amount)]

pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption('Press any key to switch dataset')
history = []
running = True
while running:
    values = [net(D) > 0 for net in population]
    M = torch.zeros((len(population), len(population)))
    values_tensor = torch.stack(values).float()
    for i in range(len(population)):
        for j in range(len(population)):
            M[i][j] = torch.sum(values_tensor[i] != values_tensor[j]).item()
    fitnesses = torch.sum(M, dim=1)
    grades = torch.argsort(fitnesses)
    p = torch.flip(fitnesses[grades], dims=[0]) / (pop_amount * data_amount)
    

    
    population = [population[grades[i]] if i < survival_amount else network() for i in range(pop_amount)]

    # D += torch.randn_like(D) * 0.1
    roll_amount = data_amount // 50
    D = torch.roll(D, roll_amount, 0)
    D[:roll_amount] = torch.randn(roll_amount, dim_amount)

    


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                D += torch.randn_like(D) * 0.01
                D = D / torch.norm(D, dim=1, keepdim=True)
                print("Tiny changes made to the dataset:")
            elif event.key == pygame.K_e:
                D += torch.randn_like(D) * 0.05
                D = D / torch.norm(D, dim=1, keepdim=True)
                print("Medium changes made to the dataset:")
            elif event.key == pygame.K_r:
                D = np.random.randn(data_amount, dim_amount)
                D = torch.tensor(D).float()
                print("New dataset generated:")

    # Keep history of average value of p[middle]
    middle_index = len(p) // 2
    history.append(p[middle_index].item())
    if len(history) > 10:
        history.pop(0)
    avg_p_middle = sum(history) / len(history)
    ##########################################



    screen.fill((0, 0, 0))
    max_p = max(p).item()
    for i, value in enumerate(p):
        height = int((value / max_p) * 300)
        pygame.draw.rect(screen, (255, 255, 255), (i * 4, 300 - height, 4, height))
    
    font = pygame.font.Font(None, 36)
    text = font.render(f'Avg P Middle: {avg_p_middle:.4f}', True, (0, 255, 0))
    screen.blit(text, (0, 10))

    pygame.display.flip()

pygame.quit()