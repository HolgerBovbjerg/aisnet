import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


class GE2ELoss(nn.Module):
    """https://github.com/cvqluu/GE2E-Loss/blob/master/ge2e.py"""

    def __init__(self, init_w=10.0, init_b=-5.0, loss_method='softmax'):
        """
        Implementation of the Generalized End-to-End loss defined in https://arxiv.org/abs/1710.10467 [1]

        Accepts an input of size (N, M, D)

            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector (e.g. d-vector)

        Args:
            - init_w (float): defines the initial value of w in Equation (5) of [1]
            - init_b (float): definies the initial value of b in Equation (5) of [1]
        """
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.loss_method = loss_method

        assert self.loss_method in ['softmax', 'contrast']

        if self.loss_method == 'softmax':
            self.embed_loss = self.embed_loss_softmax
        if self.loss_method == 'contrast':
            self.embed_loss = self.embed_loss_contrast

    def calc_new_centroids(self, embeddings, centroids, spkr, utt):
        '''
        Calculates the new centroids excluding the reference utterance
        '''
        excl = torch.cat((embeddings[spkr, :utt], embeddings[spkr, utt + 1:]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, embeddings, centroids):
        """
        Make the cosine similarity matrix with dims (N,M,N)
        """
        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(embeddings):
            cs_row = []
            for utt_idx, utterance in enumerate(speaker):
                new_centroids = self.calc_new_centroids(embeddings, centroids, spkr_idx, utt_idx)
                # vector based cosine similarity for speed
                cs_row.append(torch.clamp(
                    torch.mm(utterance.unsqueeze(1).transpose(0, 1), new_centroids.transpose(0, 1)) / (
                            torch.norm(utterance) * torch.norm(new_centroids, dim=1)), 1e-6))
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        """
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        """
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                L_row.append(-F.log_softmax(cos_sim_matrix[j, i], 0)[j])
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        """
        Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
        """
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j, i])
                excl_centroids_sigmoids = torch.cat((centroids_sigmoids[:j], centroids_sigmoids[j + 1:]))
                L_row.append(1. - torch.sigmoid(cos_sim_matrix[j, i, j]) + torch.max(excl_centroids_sigmoids))
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def forward(self, dvecs):
        """
        Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        """
        # Calculate centroids
        centroids = torch.mean(dvecs, 1)

        # Calculate the cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        L = self.embed_loss(dvecs, cos_sim_matrix)
        return L.sum()


"""PyTorch implementation of GE2E loss"""
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


class GE2ELoss2(nn.Module):
    """Implementation of the GE2E loss in https://arxiv.org/abs/1710.10467

    Accepts an input of size (N, M, D)

        where N is the number of classes (e.g., speakers in the batch)
        M is the number of examples per class (e.g., utterance per speaker),
        and D is the dimensionality of the embedding vector (e.g., d-vector dimensionality)

    Args:
        - init_w (float): the initial value of w in Equation (5)
        - init_b (float): the initial value of b in Equation (5)
    """

    def __init__(self, init_w=10.0, init_b=-5.0, loss_method="softmax"):
        super(GE2ELoss2, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor([init_w]))
        self.b = nn.Parameter(torch.FloatTensor([init_b]))
        self.loss_method = loss_method

        assert self.loss_method in ["softmax", "contrast"]

        if self.loss_method == "softmax":
            self.embed_loss = self.embed_loss_softmax
        if self.loss_method == "contrast":
            self.embed_loss = self.embed_loss_contrast

    def cosine_similarity(self, embeddings):
        """Calculate cosine similarity matrix of shape (N, M, N)."""
        n_spkr, n_uttr, d_embd = embeddings.size()

        embedding_expns = embeddings.unsqueeze(-1).expand(n_spkr, n_uttr, d_embd, n_spkr)
        embedding_expns = embedding_expns.transpose(2, 3)

        ctrds = embeddings.mean(dim=1).to(embeddings.device)
        ctrd_expns = ctrds.unsqueeze(0).expand(n_spkr * n_uttr, n_spkr, d_embd)
        ctrd_expns = ctrd_expns.reshape(-1, d_embd)

        embedding_rolls = torch.cat([embeddings[:, 1:, :], embeddings[:, :-1, :]], dim=1)
        embedding_excls = embedding_rolls.unfold(1, n_uttr - 1, 1)
        mean_excls = embedding_excls.mean(dim=-1).reshape(-1, d_embd)

        indices = _indices_to_replace(n_spkr, n_uttr).to(embeddings.device)
        ctrd_excls = ctrd_expns.index_copy(0, indices, mean_excls)
        ctrd_excls = ctrd_excls.view_as(embedding_expns)

        return F.cosine_similarity(embedding_expns, ctrd_excls, 3, 1e-6)

    def embed_loss_softmax(self, embeddings, cos_sim_matrix):
        """Calculate the loss on each embedding by taking softmax."""
        n_classes, n_examples, _ = embeddings.size()
        indices = _indices_to_replace(n_classes, n_examples).to(embeddings.device)
        losses = -F.log_softmax(cos_sim_matrix, 2)
        return losses.flatten().index_select(0, indices).view(n_classes, n_examples)

    def embed_loss_contrast(self, embeddings, cos_sim_matrix):
        """Calculate the loss on each embedding by contrast loss."""
        N, M, _ = embeddings.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j, i])
                excl_centroids_sigmoids = torch.cat(
                    (centroids_sigmoids[:j], centroids_sigmoids[j + 1:])
                )
                L_row.append(
                    1.0
                    - torch.sigmoid(cos_sim_matrix[j, i, j])
                    + torch.max(excl_centroids_sigmoids)
                )
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def forward(self, embeddings):
        """Calculate the GE2E loss for an input of dimensions (N, M, D)."""
        cos_sim_matrix = self.cosine_similarity(embeddings)
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        L = self.embed_loss(embeddings, cos_sim_matrix)
        return L.sum()


@lru_cache(maxsize=5)
def _indices_to_replace(n_spkr, n_uttr):
    indices = [
        (s * n_uttr + u) * n_spkr + s for s in range(n_spkr) for u in range(n_uttr)
    ]
    return torch.LongTensor(indices)


def equal_error_rate(similarity_matrix, n_speakers, n_utterances):
    # EER (not backpropagated)
    def inv_argmax(i):
        return np.eye(1, n_speakers, i, dtype=int)[0]

    ground_truth = np.repeat(np.arange(n_speakers), n_utterances)
    labels = np.array([inv_argmax(i) for i in ground_truth])

    similarity_matrix = similarity_matrix.reshape((n_speakers * n_utterances, n_speakers))
    preds = similarity_matrix.detach().cpu().numpy()

    # Snippet from https://yangcha.github.io/EER-ROC/
    fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


class GE2ELoss3(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, loss_device="cpu"):
        super().__init__()
        self.w = nn.Parameter(torch.FloatTensor([init_w]))
        self.b = nn.Parameter(torch.FloatTensor([init_b]))
        self.loss_device = loss_device
        self.loss_fn = nn.CrossEntropyLoss().to(self.loss_device)

    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, embedding_size)
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch, device=embeds.device)  # .to(self.loss_device)
        mask_matrix = ~torch.eye(speakers_per_batch, dtype=torch.bool, device=embeds.device)
        for j in range(speakers_per_batch):
            mask = torch.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)

        ## Even more vectorized version (slower maybe because of transpose)
        # sim_matrix2 = torch.zeros(speakers_per_batch, speakers_per_batch, utterances_per_speaker
        #                           ).to(self.loss_device)
        # eye = np.eye(speakers_per_batch, dtype=int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)

        sim_matrix = sim_matrix * self.w + self.b
        return sim_matrix

    def forward(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker,
                                         speakers_per_batch))
        ground_truth = torch.repeat_interleave(torch.arange(speakers_per_batch, device=sim_matrix.device), utterances_per_speaker)
        target = ground_truth.long()
        loss = self.loss_fn(sim_matrix, target)

        # EER (not backpropagated)
        with torch.no_grad():
            eer = equal_error_rate(sim_matrix, speakers_per_batch, utterances_per_speaker)

        return loss, eer
