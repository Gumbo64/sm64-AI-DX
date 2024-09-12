import torch
import torch.nn as nn
import numpy as np



src = torch.rand(batch_size, num_elements, input_dim)
out = transformer_encoder(src)

print(out.shape)
print(out)

out_single = torch.mean(out, dim=1)

print(out_single.shape)
print(out_single)



class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.critic = nn.Sequential(

        )
        self.actor = nn.Sequential(

        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(n_envs)], shared_memory=False)
agent = Agent(envs)
print("Number of parameters in the agent:", sum(p.numel() for p in transformer_encoder.parameters()))