from functools import partial
from warnings import warn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict

from utils import add_colorbar


class CKA:
    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 model1_name: str = None,
                 model2_name: str = None,
                 model1_layers: List[str] = None,
                 model2_layers: List[str] = None,
                 device: str ='cpu'):
        """

        :param model1: (nn.Module) Neural Network 1
        :param model2: (nn.Module) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model1 = model1
        self.model2 = model2

        self.device = device

        self.model1_info = {}
        self.model2_info = {}

        if model1_name is None:
            self.model1_info['Name'] = model1.__repr__().split('(')[0]
        else:
            self.model1_info['Name'] = model1_name

        if model2_name is None:
            self.model2_info['Name'] = model2.__repr__().split('(')[0]
        else:
            self.model2_info['Name'] = model2_name

        if self.model1_info['Name'] == self.model2_info['Name']:
            warn(f"Both model have identical names - {self.model2_info['Name']}. " \
                 "It may cause confusion when interpreting the results. " \
                 "Consider giving unique names to the nnet :)")

        self.model1_info['Layers'] = []
        self.model2_info['Layers'] = []

        self.model1_features = {}
        self.model2_features = {}

        if len(list(model1.modules())) > 150 and model1_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model1_layers' parameter. Your CPU/GPU will thank you :)")

        self.model1_layers = model1_layers

        if len(list(model2.modules())) > 150 and model2_layers is None:
            warn("Model 2 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model2_layers' parameter. Your CPU/GPU will thank you :)")

        self.model2_layers = model2_layers

        self._insert_hooks()
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)

        self.model1.eval()
        self.model2.eval()

    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model1":
            self.model1_features[name] = out

        elif model == "model2":
            self.model2_features[name] = out

        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model1.named_modules():
            if self.model1_layers is not None:
                if name in self.model1_layers:
                    self.model1_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model1", name))
            else:
                self.model1_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model1", name))

        # Model 2
        for name, layer in self.model2.named_modules():
            if self.model2_layers is not None:
                if name in self.model2_layers:
                    self.model2_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model2", name))
            else:

                self.model2_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model2", name))

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def compare(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None) -> None:
        """
        Computes the feature similarity between the nnet on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """

        if dataloader2 is None:
            warn("Dataloader for Model 2 is not given. Using the same dataloader for both nnet.")
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        self.hsic_matrix = torch.zeros(N, M, 3)

        num_batches = min(len(dataloader1), len(dataloader1))

        for data, data2 in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):

            self.model1_features = {}
            self.model2_features = {}
            _ = self.model1(data[0].to(self.device), data[2].to(self.device), data[3].to(self.device))
            _ = self.model2(data2[0].to(self.device), data2[2].to(self.device), data2[3].to(self.device))

            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                X = feat1.flatten(1)
                K = X @ X.t()
                K.fill_diagonal_(0.0)
                self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                for j, (name2, feat2) in enumerate(self.model2_features.items()):
                    Y = feat2.flatten(1)
                    L = Y @ Y.t()
                    L.fill_diagonal_(0)
                    assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                    self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                    self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())

        assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"

    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": self.model1_info['Name'],
            "model2_name": self.model2_info['Name'],
            "CKA": self.hsic_matrix,
            "model1_layers": self.model1_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],
            "dataset2_name": self.model2_info['Dataset'],

        }

    def plot_results(self,
                     save_path: str = None,
                     title: str = None):
        fig, ax = plt.subplots()
        im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma')
        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        plt.show()


if __name__ == "__main__":
    from pathlib import Path
    from copy import deepcopy

    import torch
    import matplotlib.pyplot as plt

    from common.feature_extraction import LogMelFeatureExtractor
    from common.model_loader import get_model, load_pretrained_model
    from PVAD.data_loader_SC import get_data_set, get_data_loader
    from common.augment import get_augmentor

    repo_root = Path.cwd().parent
    data_root = Path("/Users/JG96XG/Desktop/data_sets/")
    baseline_checkpoints = list((repo_root / Path("nnet/pretrained/100h/PVAD1_SC_baseline_100h/")).rglob("*/best.pt"))
    finetune_checkpoints = list(
        (repo_root / Path("nnet/pretrained/100h/PVAD1_SC_APC_finetune_100h/")).rglob("*/best.pt"))
    dn_apc_finetune_checkpoints = list(
        (repo_root / Path("nnet/pretrained/100h/PVAD1_SC_DenoisingAPC_finetune_100h_noisy_no_cafe/")).rglob(
            "*/best.pt"))

    feature_extractor = LogMelFeatureExtractor()

    model_config = {'name': 'PVAD1_SC',
                    'settings': {'input_dim': 40,
                                 'hidden_dim': 64,
                                 'num_layers': 2,
                                 'out_dim': 2
                                 }
                    }
    model = get_model(model_config)
    baseline_model = load_pretrained_model(deepcopy(model), checkpoint_path=baseline_checkpoints[0], map_location="cpu")
    finetuned_model = load_pretrained_model(deepcopy(model), checkpoint_path=finetune_checkpoints[0],
                                            map_location="cpu")
    dn_apc_finetuned_model = load_pretrained_model(deepcopy(model), checkpoint_path=dn_apc_finetune_checkpoints[0],
                                                   map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_config = {'sample_rate': 16000,
                     'n_mels': 40,
                     'n_fft': 400,
                     'window_length': 400,
                     'hop_length': 160,
                     'stacked_consecutive_features': 1,
                     'subsample_factor': 1}
    feature_extractor = LogMelFeatureExtractor(**feature_config)
    test_dataset_clean = get_data_set(name="librispeech_concat",
                                      root=str(data_root) + "/LibriSpeechConcat/",
                                      splits=["test-clean"],
                                      use_waveforms=True,
                                      feature_extractor=feature_extractor)

    noise_root = Path("/Users/JG96XG/Desktop/data_sets/noise_files/kolbek_slt2016")

    SNR = 20
    add_cafe_noise = get_augmentor(name="noise", noise_paths=str(noise_root) + "/caf/", snr_db_min=SNR, snr_db_max=SNR)
    test_dataset_cafe_noise = get_data_set(name="librispeech_concat",
                                           root=str(data_root) + "/LibriSpeechConcat/",
                                           splits=["test-clean"],
                                           use_waveforms=True,
                                           feature_extractor=feature_extractor,
                                           augmentor=add_cafe_noise)

    # %%
    batch_size = 8
    test_loader_clean = get_data_loader(test_dataset_clean, batch_size=batch_size, shuffle=False)
    test_loader_cafe_noise = get_data_loader(test_dataset_cafe_noise, batch_size=batch_size, shuffle=False)

    # cka = CKA(model1, model2,
    #         model1_name="ResNet18", model2_name="ResNet34",
    #         device='cuda')
    #
    # cka.compare(dataloader)
    #
    # cka.plot_results(save_path="../assets/resnet_compare.png")

    # ===============================================================
    # model1 = resnet50(pretrained=True)
    # model2 = wide_resnet50_2(pretrained=True)

    cka = CKA(baseline_model, dn_apc_finetuned_model,
              model1_name="Baseline", model2_name="DN-APC Pretrained",
              device='cpu')

    cka.compare(test_loader_cafe_noise)

    cka.plot_results(save_path="../assets/resnet-resnet_compare.png")