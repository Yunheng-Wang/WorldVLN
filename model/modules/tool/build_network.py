import torch.nn as nn


def build_mlp(in_features, out_features, depth):
    modules = [nn.Linear(in_features, out_features)]
    for _ in range(1, depth):
        modules.append(nn.SiLU())
        modules.append(nn.Linear(out_features, out_features))
    return nn.Sequential(*modules)