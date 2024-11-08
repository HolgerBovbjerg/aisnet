import torch
from torch import nn


class BiasConditioner(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.linear = nn.Linear(embedding_dim, input_dim)

    def forward(self, x, embedding):
        # Assumes x is (B, input_dim, T) and embedding is (B, embedding_dim)
        return x + self.linear(embedding).unsqueeze(1)


class ScaleConditioner(nn.Module):
    def __init__(self, embedding_dim, input_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.linear = nn.Linear(embedding_dim, input_dim)

    def forward(self, x, embedding):
        # Assumes x is (B, input_dim, T) and embedding is (B, embedding_dim)
        return x * self.linear(embedding).unsqueeze(1)


class FiLMGenerator(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim, 2 * input_dim)

    def forward(self, embedding):
        film_parameters = self.linear(embedding).view(embedding.size(0), 2, -1)
        beta = film_parameters[:, 0]
        gamma = film_parameters[:, 1]
        return beta, gamma


class FiLM(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.FiLM_generator = FiLMGenerator(embedding_dim=embedding_dim, input_dim=input_dim)

    def forward(self, x, embedding):
        # Assumes x is (B, input_dim, T) and embedding is (B, embedding_dim)
        beta, gamma = self.FiLM_generator(embedding=embedding)
        beta = beta.unsqueeze(1)
        gamma = gamma.unsqueeze(1)
        return gamma * x + beta


if __name__ == "__main__":
    gen = FiLMGenerator(embedding_dim=256, input_dim=64)

    emb = torch.ones(10, 256)
    x = torch.ones(10, 80, 64)

    beta, gamma = gen(emb)

    film = FiLM(embedding_dim=256, input_dim=64)

    out2 = film(x, emb)

    print("done")
