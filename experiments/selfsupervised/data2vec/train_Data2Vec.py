"""Script for training Data2Vec model"""
from argparse import ArgumentParser
from common.config_parser import get_config
import os
import time
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

from common.augment import get_composed_augmentations, SpecAugment
from data2vec.masking import AudioMaskingGenerator
from data2vec.Data2Vec import Data2Vec
from data2vec.data2vec_utils.trainer import train, evaluate
from data2vec.data2vec_utils.data_loader import build_data2vec_datapipe, pad_collate_features, \
    pad_collate_features_clean_noisy
from common.feature_extraction import LogMelFeatureExtractor
from common.misc import seed_everything, count_parameters, calc_step, log
from common.optimizer import get_optimizer
from common.scheduler import get_scheduler
from common.model_loader import get_model


def training_pipeline(config):
    """
    Initiates and executes all the steps involved with Data2Vec training
    :param config: Data2Vec configuration
    """
    config["exp"]["save_dir"] = os.path.join(config["exp"]["exp_dir"], config["exp"]["exp_name"])
    os.makedirs(config["exp"]["save_dir"], exist_ok=True)
    config_str = yaml.dump(config)
    print("Using settings:\n", config_str)

    with open(os.path.join(config["exp"]["save_dir"], "settings.txt"), "w+", encoding="utf8") as settings_file:
        settings_file.write(config_str)

    device = config["exp"]["device"]

    # feature extractor
    feature_extractor = LogMelFeatureExtractor(**config["hparams"]["audio"])

    # Augmentor
    wav_augmentor = None
    if config["hparams"]["augment"]["waveform"]:
        wav_augmentor = get_composed_augmentations(config["hparams"]["augment"]["waveform"])

    # feature augmentation
    spec_augmentor = None
    if config["hparams"]["augment"]["spectrogram"]:
        spec_augmentor = SpecAugment(**config["hparams"]["augment"]["spectrogram"]["specaugment"])

    # data
    if config["data"]["clean_and_noisy"]:
        collate_fn = pad_collate_features_clean_noisy
    else:
        collate_fn = pad_collate_features

    train_datapipe = build_data2vec_datapipe(data_sets=config["data"]["train_data"],
                                             feature_extractor=feature_extractor,
                                             augmentor=wav_augmentor,
                                             load_from_tar=config["data"]["load_from_tar"],
                                             load_from=config["data"]["load_from"],
                                             buffer_size=config["data"]["buffer_size"],
                                             batch_size=config["hparams"]["batch_size"],
                                             clean_and_noisy=config["data"]["clean_and_noisy"],
                                             segment_max_size=config["data"]["segment_max_size"],
                                             max_token_count=config["data"]["max_token_count"],
                                             min_length=config["data"]["min_length"])
    train_loader = torch.utils.data.DataLoader(dataset=train_datapipe,
                                               batch_size=1,
                                               collate_fn=collate_fn,
                                               num_workers=config["exp"]["n_workers"],
                                               shuffle=True)
    validation_datapipe = build_data2vec_datapipe(data_sets=config["data"]["validation_data"],
                                                  feature_extractor=feature_extractor,
                                                  augmentor=wav_augmentor,
                                                  load_from_tar=config["data"]["load_from_tar"],
                                                  load_from=config["data"]["load_from"],
                                                  buffer_size=config["data"]["buffer_size"],
                                                  batch_size=config["hparams"]["batch_size"],
                                                  clean_and_noisy=config["data"]["clean_and_noisy"],
                                                  segment_max_size=config["data"]["segment_max_size"],
                                                  max_token_count=config["data"]["max_token_count"],
                                                  min_length=config["data"]["min_length"])
    validation_loader = torch.utils.data.DataLoader(dataset=validation_datapipe,
                                                    batch_size=1,
                                                    collate_fn=collate_fn,
                                                    num_workers=config["exp"]["n_workers"],
                                                    shuffle=False)
    test_datapipe = build_data2vec_datapipe(data_sets=config["data"]["test_data"],
                                            feature_extractor=feature_extractor,
                                            augmentor=wav_augmentor,
                                            load_from_tar=config["data"]["load_from_tar"],
                                            load_from=config["data"]["load_from"],
                                            buffer_size=config["data"]["buffer_size"],
                                            batch_size=config["hparams"]["batch_size"],
                                            clean_and_noisy=config["data"]["clean_and_noisy"],
                                            segment_max_size=config["data"]["segment_max_size"],
                                            max_token_count=config["data"]["max_token_count"],
                                            min_length=config["data"]["min_length"])
    test_loader = torch.utils.data.DataLoader(dataset=test_datapipe,
                                              batch_size=1,
                                              collate_fn=collate_fn,
                                              num_workers=config["exp"]["n_workers"],
                                              shuffle=False)

    # create mask generator
    mask_generator = AudioMaskingGenerator(mask_probability=config["hparams"]["model"]["data2vec"]["mask_probability"],
                                           mask_length=config["hparams"]["model"]["data2vec"]["mask_length"],
                                           attention_mask=None,
                                           min_masks=config["hparams"]["model"]["data2vec"]["min_masks"])

    # create model to use as encoder in Data2Vec
    input_projection = nn.Sequential(
        nn.Linear(config["hparams"]["model"]["input_dim"], config["hparams"]["model"]["hidden_dim"]),
        nn.Dropout(p=config["hparams"]["model"]["input_dropout"])
    )
    encoder = get_model(config["hparams"]["model"]["encoder"])
    # Create Data2Vec model
    data2vec = Data2Vec(encoder=encoder,
                        input_projection=input_projection,
                        **config["hparams"]["model"]["data2vec"])

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        data2vec.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint {args.ckpt}.")
    model = data2vec.to(device)
    print(f"Created model with {count_parameters(model)} parameters.")

    # Loss
    # criterion = nn.SmoothL1Loss(reduction="none", beta=config["hparams"]["loss_beta"])
    criterion = nn.MSELoss(reduction="none")

    # Optimizer
    optimizer = get_optimizer(model, config["hparams"]["optimizer"])

    # Learning rate scheduler
    schedulers = {
        "warmup": None,
        "scheduler": None
    }

    total_iters = len(train_loader) * max(1, (config["hparams"]["n_epochs"]))
    schedulers["scheduler"] = get_scheduler(optimizer, config["hparams"]["scheduler"]["scheduler_type"],
                                            total_iters, **config["hparams"]["scheduler"]["scheduler_kwargs"])

    #####################################
    # Training Run
    #####################################

    print("Initiating training.")
    train(model, mask_generator, optimizer, criterion, train_loader, validation_loader, schedulers, config)

    #####################################
    # Final Test
    #####################################
    final_step = calc_step(config["hparams"]["n_epochs"] + 1, len(train_loader), len(train_loader) - 1)

    # evaluating the final state (last.pth)
    test_loss, test_target_var, test_prediction_var = evaluate(model, mask_generator, criterion, test_loader,
                                                               config["hparams"]["device"])
    log_dict = {
        "test_loss_last": test_loss,
        "test_target_var_last": test_target_var,
        "test_prediction_var_last": test_prediction_var,
    }
    log(log_dict, final_step, config)

    # evaluating the best validation state (best.pth)
    ckpt = torch.load(os.path.join(config["exp"]["save_dir"], "best.pth"))
    model.load_state_dict(ckpt["model_state_dict"])
    print("Best ckpt loaded.")

    test_loss, test_target_var, test_prediction_var = evaluate(model, mask_generator, criterion, test_loader,
                                                               config["hparams"]["device"])
    log_dict = {
        "test_loss_best": test_loss,
        "test_target_var_best": test_target_var,
        "test_prediction_var_best": test_prediction_var,
    }
    log(log_dict, final_step, config)


def main(arguments):
    """
    Calls training pipeline and sets up wandb logging if used
    :param args: input arguments
    """
    config = get_config(arguments.conf)
    if arguments.seed:
        config["hparams"]["seed"] = arguments.seed
    seed_everything(config["hparams"]["seed"])
    if arguments.id == "time":
        config["exp"]["exp_name"] = config["exp"]["exp_name"] + "_" + time.strftime("%Y%m%d-%H%M%S")
    elif arguments.id:
        config["exp"]["exp_name"] = config["exp"]["exp_name"] + "_" + arguments.id

    if config["exp"]["wandb"]:
        if config["exp"]["wandb_api_key"] is not None:
            with open(config["exp"]["wandb_api_key"], "r") as f:
                os.environ["WANDB_API_KEY"] = f.read()

        elif os.environ.get("WANDB_API_KEY", False):
            print("Found API key from env variable.")

        else:
            wandb.login()

        with wandb.init(project=config["exp"]["proj_name"],
                        name=config["exp"]["exp_name"],
                        config=config["hparams"],
                        group=config["exp"]["group_name"]):
            training_pipeline(config)

    else:
        training_pipeline(config)


if __name__ == "__main__":
    parser = ArgumentParser("Script for pretraining model with Autoregressive Predictive Coding.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    parser.add_argument("--ckpt", type=str, required=False, help="Path to checkpoint file.", default=None)
    parser.add_argument("--id", type=str, required=False, help="Additional experiment id. If 'time' is passed the "
                                                               "current time will be used", default=None)
    parser.add_argument("--seed", type=int, required=False, help="Optional random seed (overrules config file).",
                        default=None)

    args = parser.parse_args()

    main(arguments=args)
