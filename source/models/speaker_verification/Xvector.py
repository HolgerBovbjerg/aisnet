import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from source.nnet.encoder import ConformerEncoder


class XvectorModel(nn.Module):
    """
    x-vector model class.
    n_classes: int, number of speakers in training data
    Voxceleb2 dev = 5 994 speaker
    Librispeech 960h = 2 338 speakers
    """

    def __init__(self, feature_extractor: torch.nn.Module, encoder: torch.nn.Module, hidden_dim: int, n_classes: int = 5994):
        """Xvector model class initializer.
        Args:
            hidden_dim (int): Hidden size.
            encoder (int): Speech encoder network.
        """

        super().__init__()
        self.hidden_dim = hidden_dim

        # define the model layers
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, lengths):
        """Xvector model forward pass method."""
        x = self.feature_extractor(x)
        x = self.encoder(x, lengths)
        x = self.fc(x)
        return x

    def embed_utterance(self, features, window_size: int = 180, stride: int = 45, return_partials: bool = False):
        features_unfolded = features.unfold(size=window_size, step=stride, dimension=0).transpose(-1, -2)  # N x T x D
        partial_embeddings, _ = self(features_unfolded, x_lens=torch.ones(features_unfolded.size(0)) * window_size)
        embedding = partial_embeddings.mean(dim=0)
        if return_partials:
            return embedding, partial_embeddings, features_unfolded
        return embedding

    def embed_speaker(self, features, **kwargs):
        partial_embeddings = torch.stack([self.embed_utterance(feature, **kwargs) for feature in features])
        mean_embedding = torch.mean(partial_embeddings, dim=0)
        return torch.nn.functional.normalize(mean_embedding, p=2.0, dim=0)



if __name__ == "__main__":
    from source.nnet.utils import count_parameters
    from GE2E.loss import GE2ELoss
    model = XvectorModel(input_dim=40, hidden_dim=768, num_layers=3)
    loss_model = GE2ELoss()
    parameters = count_parameters(model)
    features = torch.ones(32, 100, 40)
    lengths = torch.ones(32, ) * 100

    out, lengths, hidden = model(x=features, x_lens=lengths, output_hidden=True)
    loss = loss_model(out.reshape(4, 8, out.size(-1)))
    print("done")
