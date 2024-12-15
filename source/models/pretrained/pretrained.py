from typing import List, Optional, Dict, Union
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from s3prl.nn import S3PRLUpstream, Featurizer

from source.models.utils import freeze_model_parameters


def get_s3prl_upstream_model(model_name: str, freeze_model: bool = True, extra_config: Optional[dict] = None, refresh: bool = False) -> S3PRLUpstream:
    model = S3PRLUpstream(model_name, extra_conf=extra_config, refresh=refresh)
    if freeze_model:
        freeze_model_parameters(model)
    return model


def get_s3prl_featurizer(model: S3PRLUpstream, layer_selection: Optional[List[int]] = None, normalizer: bool = True) -> nn.Module:
    featurizer = Featurizer(model, layer_selections=layer_selection, normalize=normalizer)
    return featurizer


@dataclass
class S3PRLFeaturizerConfig:
    model_name: str  # Name of the S3PRL upstream model
    freeze_model: bool = True  # Whether to freeze upstream model parameters
    extra_config: Optional[Dict] = None  # Additional configuration for the upstream model
    layer_selection: Optional[List[int]] = None  # Layers to use for feature extraction
    normalizer: bool = True  # Whether to normalize the extracted features
    refresh: bool = False # Whether to force re-download of model even if a version already exist


class S3PRLFeaturizer(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze_model: bool = True,
        extra_config: Optional[Dict] = None,
        layer_selection: Optional[List[int]] = None,
        normalizer: bool = True,
        refresh: bool = False
    ):
        """
        Wrapper module to combine an S3PRL upstream model and a featurizer.

        Args:
            model_name (str): Name of the S3PRL upstream model.
            freeze_model (bool): Whether to freeze the upstream model parameters.
            extra_config (dict, optional): Additional configuration for the upstream model.
            layer_selection (list of int, optional): Layers to use for feature extraction.
            normalizer (bool): Whether to normalize the extracted features.
        """
        super().__init__()
        self.freeze_model = freeze_model
        self.model = get_s3prl_upstream_model(
            model_name=model_name,
            freeze_model=freeze_model,
            extra_config=extra_config,
            refresh=refresh
        )
        if self.freeze_model:
           self.model.eval()
        self.featurizer = get_s3prl_featurizer(
            model=self.model,
            layer_selection=layer_selection,
            normalizer=normalizer
        )

    def forward(self, wavs: torch.Tensor, wavs_len: torch.Tensor):
        """
        Forward pass through the upstream model and featurizer.

        Args:
            wavs (torch.Tensor): Input waveforms, shape (batch_size, max_length).
            wavs_len (torch.Tensor): Lengths of each waveform in the batch.

        Returns:
            torch.Tensor, torch.Tensor: Processed features and corresponding lengths.
        """
        with torch.set_grad_enabled(not self.freeze_model):
            if self.freeze_model:
                self.model.eval()
            all_hs, all_hs_len = self.model(wavs, wavs_len)
        hs, hs_len = self.featurizer(all_hs, all_hs_len)
        return hs, hs_len


class PretrainedModel(nn.Module):
    def __init__(self, source: str, config: Union[S3PRLFeaturizerConfig, Dict]):
        """
        A general wrapper to select and initialize a pretrained model wrapper based on the source.

        Args:
            source (str): The source of the pretrained model ("s3prl" or "local").
            config (Union[UpstreamFeaturizerWrapperConfig, Dict]): Configuration for the selected model wrapper.
        """
        super().__init__()

        if source == "s3prl":
            if isinstance(config, S3PRLFeaturizerConfig):
                self.wrapper = S3PRLFeaturizer(**asdict(config))
            else:
                self.wrapper = S3PRLFeaturizer(**config)
        elif source == "local":
            raise ValueError(f"Source: {source} is not yet implemented. Will be available in the future")
        else:
            raise ValueError(f"Unsupported source: {source}. Valid options are 's3prl' and 'local'.")

    def forward(self, wavs: torch.Tensor, wavs_len: torch.Tensor):
        """
        Forward pass through the selected featurizer wrapper.

        Args:
            wavs (torch.Tensor): Input waveforms, shape (batch_size, max_length).
            wavs_len (torch.Tensor): Lengths of each waveform in the batch.

        Returns:
            torch.Tensor, torch.Tensor: Processed features and corresponding lengths.
        """
        return self.wrapper(wavs, wavs_len)
