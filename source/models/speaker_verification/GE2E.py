import torch
from torch import nn
from torch.nn import functional as F


class GE2EModel(nn.Module):
    """LSTM GE2E model class. """

    def __init__(self, feature_extractor: torch.nn.Module, encoder: torch.nn.Module, hidden_dim: int):
        """GE2E model class initializer.
        Args:
            hidden_dim (int): Hidden size.
            encoder (int): Speech encoder network.
        """

        super().__init__()
        self.hidden_dim = hidden_dim

        # define the model layers
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, lengths):
        """GE2E model forward pass method."""
        x = self.feature_extractor(x)
        x = self.encoder(x, lengths)
        x = self.fc(x)
        x = self.relu(x)
        x = F.normalize(x, p=2, dim=-1)
        x = (x, lengths)
        return x

    def embed_utterance(self, features, window_length: int = 160, hop_length: int = 80, return_partials: bool = False):
        features_unfolded = features.unfold(size=window_length, step=hop_length, dimension=0).transpose(-1,
                                                                                                        -2)  # N x T x D
        partial_embeddings, _ = self(features_unfolded, x_lens=torch.ones(features_unfolded.size(0),
                                                                          device=features.device) * window_length)
        embedding = partial_embeddings.mean(dim=0)
        if return_partials:
            return embedding, partial_embeddings, features_unfolded
        return embedding

    def embed_speaker(self, features, **kwargs):
        partial_embeddings = torch.stack([self.embed_utterance(feature, **kwargs) for feature in features])
        mean_embedding = torch.mean(partial_embeddings, dim=0)
        return torch.nn.functional.normalize(mean_embedding, p=2.0, dim=0)
