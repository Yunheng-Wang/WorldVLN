import torch.nn as nn


def build_mlp(in_features, out_features, depth):
    modules = [nn.Linear(in_features, out_features)]
    for _ in range(1, depth):
        modules.append(nn.SiLU())
        modules.append(nn.Linear(out_features, out_features))
    return nn.Sequential(*modules)


def build_binary_mlp(in_features):
    min_hidden_size = 64
    hidden_size = max(in_features // 2, min_hidden_size)

    modules = [nn.Linear(in_features, hidden_size), nn.SiLU()]

    while hidden_size > min_hidden_size:
        next_hidden_size = max(hidden_size // 2, min_hidden_size)
        modules.append(nn.Linear(hidden_size, next_hidden_size))
        modules.append(nn.SiLU())
        hidden_size = next_hidden_size

    modules.append(nn.Linear(hidden_size, 1))  
    return nn.Sequential(*modules)