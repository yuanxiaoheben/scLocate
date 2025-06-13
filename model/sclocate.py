import torch
from torch import nn
import numpy as np

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)

        return x

class SCL(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.image_projection = ProjectionHead(embedding_dim=1024, projection_dim=configs.c_dim, dropout= configs.c_drop_rate)
        self.gene_projection = ProjectionHead(embedding_dim=768, projection_dim=configs.c_dim, dropout= configs.c_drop_rate)

        self.relu = nn.ReLU()
        self.fc = nn.Linear(configs.c_dim, configs.c_dim)
        self.relu2 = nn.ReLU()
        self.temperature = 1
    
    def forward(self, img_fea, gene_fea):
        st_gene = self.gene_projection(gene_fea)
        st_img = self.image_projection(img_fea)
        cos_smi = (st_gene @ st_img.T) / self.temperature
        # label = torch.eye(cos_smi.shape[0], cos_smi.shape[1]).cuda()
        return  cos_smi
