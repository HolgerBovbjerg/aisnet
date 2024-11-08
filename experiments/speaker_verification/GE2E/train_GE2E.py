from argparse import ArgumentParser
from math import ceil
import os
import time

import torch
import yaml
import wandb
import numpy as np
from tqdm import tqdm

from common.augment import get_composed_augmentations, SpecAugment
from common.misc import seed_everything, count_parameters, freeze_model_parameters, log, save_model
from common.feature_extraction import LogMelFeatureExtractor
from common.model_loader import get_model, load_pretrained_model
from common.metrics import compute_eer, compute_error_rates, compute_min_dcf
from common.optimizer import get_optimizer
from common.scheduler import get_scheduler
from common.config_parser import get_config
from source.models.speaker_verification.GE2E import build_ge2e_datapipe, get_ge2e_loader
from xvector.data_loader import build_voxceleb1_test_datapipe, voxceleb1_test_collate, get_data_loader
from source.models.speaker_verification.GE2E import GE2ELoss3


def load_pretrained_lstm(model: torch.nn.Module, checkpoint_path: str = "", map_location: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_state_dict = model.state_dict()
    checkpoint_model_state_dict = checkpoint["model_state_dict"]
    model_state_dict.update(checkpoint_model_state_dict)

    model.load_state_dict(model_state_dict)
    print(f"Loaded checkpoint {checkpoint_path}.")
    return model


def train_one_batch(model, features, lengths, criterion, spec_augmentor, optimizer, scheduler, device, config):
    n_speakers = config["hparams"]["batch_size"]
    n_utterances = config["hparams"]["n_utterances"]

    if spec_augmentor:
        for i, feature in enumerate(features):
            features[i, :lengths[i]] = spec_augmentor(feature[:lengths[i]])

    features, lengths = features.to(device), lengths.to(device)

    out, lengths = model(features, lengths)
    out = out.reshape(n_speakers, n_utterances, out.size(-1))

    loss, eer = criterion(out)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(criterion.parameters()),
        max_norm=3.0,
        norm_type=2.0,
    )

    model.fc.weight.grad *= 0.5
    model.fc.bias.grad *= 0.5
    criterion.w.grad *= 0.01
    criterion.b.grad *= 0.01

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    return loss.item(), eer


@torch.no_grad()
def evaluate(model, dataloader, device):
    batch_step = 0
    start_time = time.time()
    labels_list = []
    similarity_list = []

    model.eval()
    for batch in tqdm(dataloader):
        features1, lengths1, features2, lengths2, labels = batch

        features1, lengths1, features2, lengths2, labels = features1.to(device), lengths1.to(device), \
            features2.to(device), lengths2.to(device), labels.to(device)

        enrollment = torch.stack([model.embed_utterance(features[:length])
                                  for features, length in zip(features1, lengths1)])
        test = torch.stack([model.embed_utterance(features[:length])
                            for features, length in zip(features2, lengths2)])

        similarity = torch.nn.functional.cosine_similarity(enrollment, test, dim=-1)
        similarity[similarity < 0.] = 0.  # Set negative cosine distances to zero

        labels_list.extend(labels.tolist())
        similarity_list.extend(similarity.tolist())

        batch_step += 1

    eer, eer_threshold = compute_eer(labels_list, similarity_list)
    fprs, fnrs, thresholds = compute_error_rates(similarity_list, labels_list)
    min_cdf, min_cdf_threshold = compute_min_dcf(fprs=fprs, fnrs=fnrs, thresholds=thresholds)

    avg_eer = eer
    avg_min_cdf = min_cdf
    evaluation_time = time.time() - start_time

    model.train()

    return avg_eer, avg_min_cdf, evaluation_time


def test(model_path, dataloader, device, config):
    print(f"Testing {model_path}")
    model = get_model(config["hparams"]["model"]["encoder"])
    model = load_pretrained_model(model, model_path)
    model = model.to(device)
    avg_eer, avg_min_cdf, test_time = evaluate(model, dataloader, device)
    return avg_eer, avg_min_cdf, test_time


def training_pipeline(config):
    """
    Initiates and executes all the steps involved with model training and testing
    :param config: Experiment configuration
    """
    config["exp"]["save_dir"] = os.path.join(config["exp"]["exp_dir"], config["exp"]["exp_name"])
    os.makedirs(config["exp"]["save_dir"], exist_ok=True)
    config_str = yaml.dump(config)
    print("Using settings:\n", config_str)

    with open(os.path.join(config["exp"]["save_dir"], "settings.txt"), "w+", encoding="utf8") as settings_file:
        settings_file.write(config_str)

    # experiment
    device = config["exp"]["device"]
    epochs = config["hparams"]["n_epochs"]
    gradient_accumulation_steps = config["hparams"]["loss"]["accumulation_steps"]
    log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")
    best_model_path = os.path.join(config["exp"]["save_dir"], "best.pt")
    last_model_path = os.path.join(config["exp"]["save_dir"], "last.pt")

    # feature extractor
    if config["data"]["load_from"] == "raw":
        feature_extractor = LogMelFeatureExtractor(**config["hparams"]["audio"])
    else:
        feature_extractor = None

    # waveform augmentation
    wav_augmentor = None
    if config["hparams"]["augment"]["waveform"]:
        wav_augmentor = get_composed_augmentations(config["hparams"]["augment"]["waveform"])

    # feature augmentation 
    spec_augmentor = None
    if config["hparams"]["augment"]["spectrogram"]:
        spec_augmentor = SpecAugment(**config["hparams"]["augment"]["spectrogram"]["specaugment"])

    train_datapipe = build_ge2e_datapipe(data_sets=config["data"]["train_data"],
                                         n_utterances=config["hparams"]["n_utterances"],
                                         batch_size=config["hparams"]["batch_size"],
                                         buffer_size=config["data"]["buffer_size"],
                                         load_from=config["data"]["load_from"],
                                         shuffle=True,
                                         augmentor=wav_augmentor,
                                         feature_extractor=feature_extractor,
                                         chunk_size=config["data"]["chunk_size"],
                                         load_from_tar=config["data"]["load_from_tar"])
    train_loader = get_ge2e_loader(datapipe=train_datapipe,
                                   reading_service=config["exp"]["reading_service"],
                                   batch_size=None,  # config["hparams"]["batch_size"],
                                   num_workers=config["exp"]["n_workers"],
                                   pin_memory=config["exp"]["pin_memory"],
                                   shuffle=True,
                                   chunk_size=config["data"]["chunk_size"])
    validation_datapipe = build_voxceleb1_test_datapipe(voxceleb1_root=config["data"]["voxceleb1_root"],
                                                        feature_extractor=feature_extractor,
                                                        augmentor=None,  # wav_augmentor,
                                                        batch_size=config["data"]["validation_data"]["batch_size"],
                                                        test_pair_file_path=config["data"]["validation_data"][
                                                            "test_pair_file_path"])
    validation_loader = get_data_loader(dataset=validation_datapipe,
                                        batch_size=None,
                                        num_workers=config["exp"]["n_workers"],
                                        pin_memory=config["exp"]["pin_memory"],
                                        shuffle=False,
                                        collate_fn=voxceleb1_test_collate)
    test_datapipe = build_voxceleb1_test_datapipe(voxceleb1_root=config["data"]["voxceleb1_root"],
                                                  feature_extractor=feature_extractor,
                                                  augmentor=None,  # wav_augmentor,
                                                  batch_size=config["data"]["test_data"]["batch_size"],
                                                  test_pair_file_path=config["data"]["test_data"][
                                                      "test_pair_file_path"])
    test_loader = get_data_loader(dataset=test_datapipe,
                                  batch_size=None,
                                  num_workers=config["exp"]["n_workers"],
                                  pin_memory=config["exp"]["pin_memory"],
                                  shuffle=False,
                                  collate_fn=voxceleb1_test_collate)

    # model
    model = get_model(config["hparams"]["model"]["encoder"])
    print(f"Created model with {count_parameters(model)} parameters.")

    if "checkpoint" in config["hparams"]["model"]:
        model.encoder = load_pretrained_model(model.encoder,
                                              checkpoint_path=config["hparams"]["model"]["checkpoint"][
                                                  "checkpoint_path"],
                                              map_location="cpu")
        if config["hparams"]["model"]["checkpoint"]["freeze"]:
            freeze_model_parameters(model.encoder)
            print(f"{count_parameters(model.encoder)} parameters frozen. "
                  f"{count_parameters(model, trainable=True)} trainable parameters.")

    model = model.to(device)

    # loss
    criterion = GE2ELoss3().to(device)

    # optimizer
    optimizer = get_optimizer([model, criterion], config["hparams"]["optimizer"])

    # lr scheduler
    scheduler = None
    if config["hparams"]["scheduler"]["scheduler_type"] is not None:
        if config["hparams"]["scheduler"]["steps_per_epoch"]:
            total_iters = config["hparams"]["scheduler"]["steps_per_epoch"] * max(1, (config["hparams"]["n_epochs"]))
            scheduler = get_scheduler(optimizer,
                                      scheduler_type=config["hparams"]["scheduler"]["scheduler_type"],
                                      t_max=total_iters,
                                      **config["hparams"]["scheduler"]["scheduler_kwargs"])
        else:
            total_iters = ceil(len(train_loader) / config["hparams"]["loss"]["accumulation_steps"])
            total_iters = total_iters * max(1, (config["hparams"]["n_epochs"]))

            scheduler = get_scheduler(optimizer,
                                      scheduler_type=config["hparams"]["scheduler"]["scheduler_type"],
                                      t_max=total_iters,
                                      **config["hparams"]["scheduler"]["scheduler_kwargs"])

    #####################################
    # Training Run
    #####################################

    print(f"Initiating training on {device=}.")
    step = 0
    best_score = np.inf

    n_speakers = config["hparams"]["batch_size"]
    n_utterances = config["hparams"]["n_utterances"]

    for epoch in range(epochs):
        batch_step = 0
        train_epoch_loss = 0.
        train_epoch_eer = 0.

        epoch_start_time = time.time()
        model.train()
        batch_start_time = time.time()
        for batch in train_loader:
            data_load_time = time.time() - batch_start_time
            features, lengths, speaker_ids, audio_paths, segment_ids = batch

            if features.size(0) != (n_speakers * n_utterances):
                continue  # skip non-full batches

            if spec_augmentor:
                for i, feature in enumerate(features):
                    features[i, :lengths[i]] = spec_augmentor(feature[:lengths[i]])

            features, lengths = features.to(device), lengths.to(device)

            train_batch_loss, train_batch_eer = train_one_batch(model, features, lengths, criterion, spec_augmentor,
                                                                optimizer, scheduler, device, config)
            train_epoch_loss += train_batch_loss
            train_epoch_eer += train_batch_eer

            log_dict = {"epoch": epoch,
                        "batch": batch_step,
                        "train_loss": train_batch_loss,
                        "train_eer": train_batch_eer,
                        "data_load_time": data_load_time,
                        "time_per_batch": time.time() - batch_start_time,
                        "lr": optimizer.param_groups[0]["lr"]}

            if (step % config["exp"]["val_freq"]) == 0 and (step > 0) and (config["exp"]["val_freq"] > 0):
                avg_val_eer, avg_val_min_cdf, validation_time = evaluate(model=model,
                                                                         dataloader=validation_loader,
                                                                         device=device)
                validation_dict = {"avg_validation_eer": avg_val_eer,
                                   "avg_validation_min_cdf": avg_val_min_cdf,
                                   "validation_time": validation_time}
                log_dict.update(validation_dict)

                if avg_val_eer <= best_score:
                    best_score = avg_val_eer
                    save_model(epoch, avg_val_eer, best_model_path, model, optimizer, log_file, criterion=criterion)

            log(log_dict, step, config)

            batch_step += 1
            step += 1
            batch_start_time = time.time()

        log_dict = {"epoch": epoch,
                    "batch": batch_step,
                    "avg_train_loss": train_epoch_loss / batch_step,
                    "avg_train_eer": train_epoch_eer / batch_step,
                    "time_per_epoch": time.time() - epoch_start_time}

        if (epoch + 1) == epochs:
            avg_val_eer, avg_val_min_cdf, validation_time = evaluate(model, validation_loader, device)
            validation_dict = {"avg_validation_eer": avg_val_eer,
                            "avg_validation_min_cdf": avg_val_min_cdf,
                            "validation_time": validation_time}
            log_dict.update(validation_dict)
            if avg_val_eer <= best_score:
                best_score = avg_val_eer
                save_model(epoch, avg_val_eer, best_model_path, model, optimizer, log_file, criterion=criterion)

            save_model(epoch, avg_val_eer, last_model_path, model, optimizer, log_file, criterion=criterion)

        log(log_dict, step, config)


    print("Finished training.\n")

    #####################################
    # Test Run
    #####################################
    print(f"Initiating test on {device=}.")

    avg_test_eer_last, avg_test_min_cdf_last, test_time_last = test(last_model_path, test_loader, device,
                                                                    config)
    avg_test_eer_best, avg_test_min_cdf_best, test_time_best = test(best_model_path, test_loader, device,
                                                                    config)
    log_dict = {"avg_eer_last": avg_test_eer_last,
                "avg_test_min_cdf_last": avg_test_min_cdf_last,
                "test_time_last": test_time_last,
                "avg_test_eer_best": avg_test_eer_best,
                "avg_test_min_cdf_best": avg_test_min_cdf_best,
                "test_time_best": test_time_best}
    log(log_dict, step, config)


def main(arguments):
    """
    Calls training pipeline and sets up wandb logging if used
    """
    config = get_config(arguments.conf)
    if args.seed:
        config["hparams"]["seed"] = args.seed
    seed_everything(config["hparams"]["seed"])
    if args.id == "time":
        config["exp"]["exp_name"] = config["exp"]["exp_name"] + "_" + time.strftime("%Y%m%d-%H%M%S")
    elif args.id:
        config["exp"]["exp_name"] = config["exp"]["exp_name"] + "_" + args.id

    if config["exp"]["wandb"]:
        if config["exp"]["wandb_api_key"] is not None:
            with open(config["exp"]["wandb_api_key"], "r", encoding="utf8") as file:
                os.environ["WANDB_API_KEY"] = file.read()

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
    parser = ArgumentParser("Training and evaluation script for GE2E nnet.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    parser.add_argument("--id", type=str, required=False, help="Optional experiment identifier.", default=None)
    parser.add_argument("--seed", type=int, required=False, help="Optional random seed (overrules config file).",
                        default=None)

    args = parser.parse_args()

    main(arguments=args)
