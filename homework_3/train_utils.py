from datetime import datetime
from os import makedirs
from os.path import join
from typing import Callable

import torch
from torchvision import utils
from tqdm.auto import trange, tqdm

CHECKPOINT_DIR = "checkpoints"
SAMPLES_DIR = "samples"


def train_loop(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    z_noise: torch.Tensor,
    train_step_func: Callable,
    device: torch.device,
    *,
    n_epochs: int = 10,
    lr: float = 1e-3,
    decay_gamma: float = 0.95,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_gamma)

    timestamp = datetime.now().strftime("%m_%d__%H_%M_%S")
    ckpt_path = join(CHECKPOINT_DIR, timestamp, model.__class__.__name__.lower())
    samples_path = join(SAMPLES_DIR, timestamp, model.__class__.__name__.lower())
    makedirs(ckpt_path)
    makedirs(samples_path)

    model.eval()
    with torch.no_grad():
        sampled_images = model.sample(z_noise).detach().cpu()
    utils.save_image(sampled_images, join(samples_path, f"epoch_0.png"), nrow=5, padding=5)

    losses = []
    with trange(n_epochs, desc="Epochs") as epoch_pbar:
        for e in epoch_pbar:
            model.train()
            for images in tqdm(dataloader, desc="Training", leave=False):
                images = images.to(device)
                loss = train_step_func(model, images)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                losses.append(loss.item())
                epoch_pbar.set_postfix({"loss": losses[-1]})

            torch.save(model.state_dict(), join(ckpt_path, f"epoch_{e + 1}.ckpt"))

            model.eval()
            with torch.no_grad():
                sampled_images = model.sample(z_noise).detach().cpu()
            utils.save_image(sampled_images, join(samples_path, f"epoch_{e + 1}.png"), nrow=5, padding=5)

            scheduler.step()

    return losses
