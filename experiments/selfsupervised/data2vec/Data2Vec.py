"""Data2Vec module based on https://github.com/arxyzan/data2vec-pytorch"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from source.nnet.modules.EMA import EMA
from typing import Optional


class Data2Vec(nn.Module):
    """
    Data2Vec main module.
    """
    MODALITIES = ['vision', 'text', 'audio']

    def __init__(self,
                 encoder: nn.Module,
                 modality: str,
                 encoder_embedding_dim: int,
                 ema_decay: float,
                 ema_end_decay: float,
                 ema_anneal_end_step: int,
                 average_top_k_layers: int,
                 normalize_targets: bool = True,
                 input_projection: Optional[nn.Module] = None,
                 **kwargs):
        """
        :param encoder: transformer encoder
        :param modality: vision, audio or text
        :param encoder_embedding_dim: Embedding dimension of transformer encoder
        :param ema_decay: EMA model decay
        :param ema_end_decay: EMA model end decay
        :param ema_anneal_end_step: Number of annealing steps for EMA model decay
        :param average_top_k_layers: Number of encoder layers to use for Data2Vec target
        :param normalize_targets: Specifies whether Dat2Vec targets are normalized
        :param kwargs: keyword arguments
        """
        super().__init__()
        self.encoder = encoder
        assert modality in self.MODALITIES
        self.modality = modality
        self.embed_dim = encoder_embedding_dim
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_step = ema_anneal_end_step
        self.average_top_k_layers = average_top_k_layers
        self.normalize_targets = normalize_targets
        self.input_projection = input_projection
        self.__dict__.update(kwargs)
        self.ema = EMA(self.encoder)  # Instantiate teacher encoder
        self.regression_head = self._build_regression_head()  # Instantiate regression head to predict target

    def _build_regression_head(self):
        if self.modality == 'text':
            embed_dim = self.embed_dim
            curr_dim = embed_dim
            projections = []
            for i in range(self.cfg.model.head_layers - 1):
                next_dim = embed_dim * 2 if i == 0 else curr_dim
                projections.append(nn.Linear(curr_dim, next_dim))
                projections.append(nn.GELU())
                curr_dim = next_dim

            projections.append(nn.Linear(curr_dim, embed_dim))
            return nn.Sequential(*projections)

        if self.modality in ['audio', 'vision']:
            return nn.Linear(self.embed_dim, self.embed_dim)

    def ema_step(self):
        """
        Function which to step the EMA encoder
        """
        if self.ema_decay != self.ema_end_decay:
            if self.ema.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.ema.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.ema.num_updates,
                    self.ema_anneal_end_step,
                )
            self.ema.decay = decay
        if self.ema.decay < 1:
            self.ema.step(self.encoder)

    def forward(self, student_input, lengths, teacher_input=None, mask=None):
        """
        Data2Vec forward method.
        :param student_input: Input for student encoder.
        :param lengths: List of valid lengths in input.
        :param teacher_input: Input for teacher encoder.
        :param mask: mask for student input if input is not already masked.
        :return: Data2Vec model output prediction, target for student prediction and teacher target, respectively.
        """

        if self.input_projection:
            student_input = self.input_projection(student_input)
            teacher_input = self.input_projection(teacher_input)

        encoder_out, lengths, student_hidden_states = self.encoder(student_input, lengths, mask=mask,
                                                                   output_hidden_states=True)
        if teacher_input is None:
            return encoder_out, lengths
        prediction = student_hidden_states[-1]
        with torch.no_grad():
            self.ema.model.eval()

            _, _, teacher_hidden_states = self.ema.model(teacher_input, lengths, mask=None, output_hidden_states=True)

            target = teacher_hidden_states[-self.average_top_k_layers:]
            if self.modality in ['vision', 'text']:  # Follow the same layer normalization procedure for text and vision
                target = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in target]
                target = sum(target) / len(target)
                if self.normalize_targets:
                    target = F.layer_norm(target.float(), target.shape[-1:])

            elif self.modality == 'audio':  # Use instance normalization for audio
                target = [F.instance_norm(tl.float().transpose(1, 2)).transpose(1, 2) for tl in target]
                target = sum(target) / len(target)
                if self.normalize_targets:
                    target = F.instance_norm(target).transpose(1, 2).transpose(1, 2)

        prediction = prediction[mask]
        target = target[mask]

        prediction = self.regression_head(prediction)

        return prediction, target
