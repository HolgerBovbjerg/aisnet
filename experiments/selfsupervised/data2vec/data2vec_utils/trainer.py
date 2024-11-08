"""Functions for training Data2Vec model"""
import math
import os
import time
from typing import Callable, Any, Tuple

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data2vec.masking import AudioMaskingGenerator
from data2vec.Data2Vec import Data2Vec
from common.misc import log, save_model


def compute_var(y: torch.Tensor):
    """
    Function for computing standard deviation of target
    :param y:
    :return: standard deviation of y
    """
    y = y.view(-1, y.size(-1))
    return torch.sqrt(y.var(dim=0) + 1e-6).mean()


def variance_loss(z_a: torch.Tensor, z_b: torch.Tensor):
    """
    Variance loss as defined in VICReg
    :param z_a: prediction
    :param z_b: target
    :return: variance loss
    """
    std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-6).mean()
    std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-6).mean()
    var_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))
    return var_loss


def train_single_batch(model: Data2Vec, student_input: torch.Tensor, teacher_input: torch.Tensor, lengths, mask: torch.Tensor, optimizer: optim.Optimizer,
                       criterion: Callable, device: torch.device) -> Tuple[Any, Any, Any]:
    """
    Performs single training step for Data2Vec model
    :param model: Data2Vec model
    :param data: input data
    :param mask: mask for student input
    :param optimizer: torch optimizer
    :param criterion: torch criterion
    :param device: device for model and optimizer
    :return: loss, target variance and prediction variance
    """
    optimizer.zero_grad()
    predictions, targets = model(student_input=student_input, lengths=lengths, teacher_input=teacher_input, mask=mask)
    scale = math.sqrt(predictions.size(dim=-1))

    loss = criterion(predictions.float(), targets.float()).sum(dim=-1).sum().div(scale)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        target_var = compute_var(targets.float())
        prediction_var = compute_var(predictions.float())
    return loss.item(), target_var.item(), prediction_var.item()


@torch.no_grad()
def evaluate(model: Data2Vec, mask_generator: AudioMaskingGenerator, criterion: Callable, dataloader: DataLoader,
             device: torch.device) -> Tuple[float, float, float]:
    """
    Evaluates Data2Vec model
    :param model: Data2Vec model
    :param mask_generator: mask generator for student input mask
    :param criterion: torch criterion
    :param dataloader: torch dataloader
    :param device: device for model and optimizer
    :return: loss, target variance and prediction variance
    """

    model.eval()  # Set model in evaluation mode
    # Initialise metrics
    running_loss = 0.0
    running_target_var = 0.0
    running_prediction_var = 0.0
    batch_step = 0

    for batch_idx, data in tqdm(enumerate(dataloader)):
        if len(data) < 3:
            student_input = data[0]
            teacher_input = data[0]
            lengths = data[1]
        else:
            student_input = data[0]
            teacher_input = data[1]
            lengths = data[2]

        batch_size = student_input.size(dim=0)
        audio_length = student_input.size(dim=1)

        student_input = student_input.to(device)
        teacher_input = teacher_input.to(device)
        lengths = lengths.to(device)
        mask = mask_generator(shape=(batch_size, audio_length)).to(device)
        predictions, targets = model(student_input=student_input, lengths=lengths, teacher_input=teacher_input, mask=mask)

        scale = math.sqrt(predictions.size(dim=-1))
        loss = criterion(predictions.float(), targets.float()).sum(dim=-1).sum().div(scale)
        target_var = compute_var(targets.float())
        prediction_var = compute_var(predictions.float())
        running_loss += loss.item()
        running_target_var += target_var.item()
        running_prediction_var += prediction_var.item()

        batch_step = batch_idx

    avg_loss = running_loss / (batch_step + 1)
    avg_target_var = running_target_var / (batch_step + 1)
    avg_prediction_var = running_prediction_var / (batch_step + 1)

    model.train()  # Set model in training mode
    return avg_loss, avg_target_var, avg_prediction_var


def train(model: Data2Vec, mask_generator, optimizer: optim.Optimizer, criterion: Callable, train_loader: DataLoader,
          validation_loader: DataLoader, schedulers: dict, config: dict):
    """
    Trains Data2Vec model
    :param model: Data2Vec model
    :param mask_generator: student input mask generator
    :param optimizer: torch optimizer
    :param criterion: torch criterion
    :param train_loader: training set loader
    :param validation_loader: validation set loader
    :param schedulers: learning rate scheduler
    :param config: model and training config
    """

    step = 0
    best_avg_loss = 0.0
    device = config["exp"]["device"]
    log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")

    ############################
    # start training
    ############################
    model.train()

    for epoch in range(config["hparams"]["n_epochs"]):
        t0 = time.time()
        running_loss = 0.0
        running_target_var = 0.
        running_prediction_var = 0.
        batch_step = 0
        batch_start_time = time.time()

        for batch_index, data, in enumerate(train_loader):
            data_load_time = time.time() - batch_start_time
            if len(data) < 3:
                student_input = data[0]
                teacher_input = data[0]
                lengths = data[1]
            else:
                student_input = data[0]
                teacher_input = data[1]
                lengths = data[2]

            batch_size = student_input.size(dim=0)
            audio_length = student_input.size(dim=1)

            ####################
            # optimization step
            ####################
            student_input = student_input.to(device)
            teacher_input = teacher_input.to(device)
            lengths = lengths.to(device)
            mask = mask_generator(shape=(batch_size, audio_length)).to(device)
            loss, target_var, prediction_var = train_single_batch(model=model, student_input=student_input,
                                                                  teacher_input=teacher_input, lengths=lengths,
                                                                  mask=mask, optimizer=optimizer, criterion=criterion,
                                                                  device=device)
            model.ema_step()
            running_loss += loss
            running_target_var += target_var
            running_prediction_var += prediction_var

            if not step % config["exp"]["log_freq"]:
                log_dict = {"epoch": epoch,
                            "loss": loss,
                            "lr": optimizer.param_groups[0]["lr"],
                            "target_var": target_var,
                            "prediction_var": prediction_var,
                            "time_per_batch": time.time() - batch_start_time,
                            "data_load_time": data_load_time}
                log(log_dict, step, config)

            if schedulers["warmup"] is not None and epoch < config["hparams"]["scheduler"]["n_warmup"]:
                schedulers["warmup"].step()

            elif schedulers["scheduler"] is not None:
                schedulers["scheduler"].step()

            batch_step = batch_index
            step += 1
            batch_start_time = time.time()

        #######################
        # epoch complete
        #######################

        log_dict = {"epoch": epoch,
                    "time_per_epoch": time.time() - t0,
                    "avg_train_target_var": running_target_var / (batch_step + 1),
                    "avg_train_prediction_var": running_prediction_var / (batch_step + 1),
                    "avg_train_loss": running_loss / (batch_step + 1)}
        log(log_dict, step, config)

        if not epoch % config["exp"]["val_freq"]:
            avg_val_loss, avg_val_target_var, avg_val_prediction_var = evaluate(model, mask_generator, criterion,
                                                                                validation_loader, device)
            log_dict = {"epoch": epoch,
                        "val_loss": avg_val_loss,
                        "avg_val_target_var": avg_val_target_var,
                        "avg_val_prediction_var": avg_val_prediction_var}
            log(log_dict, step, config)

            # save best validation checkpoint
            if avg_val_loss < best_avg_loss or epoch == config["exp"]["val_freq"]:
                best_avg_loss = avg_val_loss
                save_path = os.path.join(config["exp"]["save_dir"], "best.pth")
                save_model(epoch, avg_val_loss, save_path, model, optimizer, log_file)
                save_path = os.path.join(config["exp"]["save_dir"], "best_encoder.pth")
                save_model(epoch, avg_val_loss, save_path, model.encoder, optimizer, log_file)

    ###########################
    # training complete
    ###########################

    avg_val_loss, avg_val_target_var, avg_val_prediction_var = evaluate(model, mask_generator, criterion,
                                                                        validation_loader,
                                                                        device)
    log_dict = {"epoch": epoch, "val_loss": avg_val_loss,
                "avg_val_target_var": avg_val_target_var, "avg_val_prediction_var": avg_val_prediction_var}

    log(log_dict, step, config)

    # save final checkpoint
    save_path = os.path.join(config["exp"]["save_dir"], "last.pth")
    save_model(epoch, avg_val_loss, save_path, model, optimizer, log_file)
    save_path = os.path.join(config["exp"]["save_dir"], "last_encoder.pth")
    save_model(epoch, avg_val_loss, save_path, model.encoder, optimizer, log_file)
