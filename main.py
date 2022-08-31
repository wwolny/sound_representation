import os
import sys
import torch
import yaml
import wandb
import logging
import argparse
import numpy as np
from typing import List
from dataclasses import asdict
import matplotlib.pyplot as plt
from torchaudio.datasets import LJSPEECH
from torchmetrics import SignalNoiseRatio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from sound_repr.dataset import Samples, CustomDataset
from sound_repr.utils import Sine, LSD
from cfg import MainConfig

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=os.environ.get("LOGLEVEL", "INFO"),
)


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    return parser.parse_args(args)


def main(config: MainConfig):
    logger.info("Setting seed {0}".format(config.seed))
    torch.manual_seed(config.seed)

    logger.info("Process dataset...")
    if config.name == "LJSPEECH":
        dataset = LJSPEECH('datasets', download=False)
    elif config.name == "GTZAN":
        # dataset = GTZAN('drive/MyDrive/MgrData', download=True)
        dataset = CustomDataset("datasets/small_gtzan/wavs")
    elif config.name == "ESC50":
        dataset = CustomDataset("datasets/small_rsc_50/wavs")
    elif config.name == "UrbanSound":
        dataset = CustomDataset("datasets/small_urban_sound/wavs")
    else:
        logger.error("Wrong dataset name")
        return

    config.SR = dataset[0][1]

    melspec = MelSpectrogram(
        sample_rate=config.SR,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels
    )
    amp2db = AmplitudeToDB()

    samples = Samples(dataset, SR=config.SR, frame_len=config.frame_len)

    logger.info("Running in mode: {0}".format(config.mode))
    for id_ in range(config.samples):
        # Initialize wandb
        if config.wandb:
            logger.info("Initializing  WandB...")
            run = wandb.init(
                project="sound_representation",
                entity="wwolny",
                config={**asdict(config), **{"sample_id": id_}},
                reinit=True,
            )

        logger.info(f"Process sample index: {id_}.")
        data = samples[id_]
        x_numpy = np.array(list(map(lambda el: [el], range(len(data)))))
        x_numpy = (x_numpy / len(x_numpy)) * 2 * config.t_scale_max - config.t_scale_max
        x = torch.from_numpy(x_numpy).float()

        y_numpy = np.array(list(map(lambda el: [el], data)))
        y_numpy = (y_numpy / max(max(y_numpy), -1 * min(y_numpy)))
        y = torch.from_numpy(y_numpy).float()

        if config.mode == "nerf":
            L = config.L
            indices = torch.arange(0, len(x_numpy),
                                   dtype=torch.float).unsqueeze(-1)
            indices = -1 + (1 + 1) * indices / (len(x_numpy) - 1)
            indices = indices.repeat(1, 2 * L)
            for nerf_l in range(L):
                arg = 2 ** nerf_l * torch.pi * indices[:, 2 * nerf_l]
                indices[:, 2 * nerf_l] = torch.sin(arg)
                indices[:, 2 * nerf_l + 1] = torch.cos(arg)
            x = indices

            model = torch.nn.Sequential(
                *build_network(
                    2 * L,
                    1,
                    config.network,
                    torch.nn.ReLU(),
                    config.bias,
                    torch.nn.ReLU(),
                )
            )
        elif config.mode == "siren":
            module_lst = build_network(
                1,
                1,
                config.network,
                Sine(config.hidden_omega),
                config.bias,
                Sine(config.omega),
            )
            with torch.no_grad():
                for id_mod, mod in enumerate(module_lst):
                    if isinstance(mod, torch.nn.Linear):
                        if id_mod == 0:
                            mod.weight.uniform_(-1 / mod.in_features, 1 / mod.in_features)
                            if config.siren_bias_init:
                                mod.bias.uniform_(-1 / mod.in_features, 1 / mod.in_features)
                        else:
                            mod.weight.uniform_(
                                -np.sqrt(6 / mod.in_features) / config.hidden_omega,
                                np.sqrt(6 / mod.in_features) / config.hidden_omega
                            )
                            if config.siren_bias_init:
                                mod.bias.uniform_(
                                    -np.sqrt(6 / mod.in_features) / config.hidden_omega,
                                    np.sqrt(6 / mod.in_features) / config.hidden_omega
                                )
            model = torch.nn.Sequential(
                *module_lst
            )

        elif config.mode == "default":
            model = torch.nn.Sequential(
                *build_network(
                    1,
                    1,
                    config.network,
                    torch.nn.ReLU(),
                    config.bias,
                    torch.nn.ReLU(),
                )
            )
        else:
            exit()

        if config.wandb:
            run.watch(model)

        loss_fn = getattr(torch.nn, config.loss_fn)()
        optimizer = getattr(torch.optim, config.optimizer)(
            model.parameters(), lr=config.lr)

        plot_spectrogram(
            sample=y.detach().numpy(),
            mel_spec=melspec,
            amp2db=amp2db
        )
        if config.wandb:
            run.log({"original": plt}, commit=False)
        else:
            plt.show()
        plt.close()

        # train
        batch_size = config.batch_size
        loss_fn_list = []
        loss_lst = {}
        for el in config.loss_lst:
            loss_lst[el] = getattr(torch.nn, el)()
        if config.SNR:
            loss_lst["SNR"] = SignalNoiseRatio()
        for i in range(config.epochs):
            if config.batched:
                permutation = torch.randperm(x.size()[0])
                for j in range(0, x.size()[0], batch_size):
                    optimizer.zero_grad()
                    indices = permutation[j:min(x.size()[0], j + batch_size)]
                    batch_x, batch_y = x[indices], y[indices]

                    y_pred = model(batch_x)
                    loss = loss_fn(y_pred, batch_y)

                    loss.backward()
                    optimizer.step()
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                loss_fn_list.append(loss.item())
                if config.wandb:
                    for el in loss_lst.keys():
                        run.log({el: loss_lst[el](y_pred, y).item(), 'epoch': i}, commit=False)
            else:
                optimizer.zero_grad()

                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                loss_fn_list.append(loss.item())
                if config.wandb:
                    for el in loss_lst.keys():
                        run.log({el: loss_lst[el](y_pred, y).item(), 'epoch': i}, commit=False)

                loss.backward()
                optimizer.step()

            if config.plot and i % config.log_img == 0:
                y_pred = model(x)
                plt.plot(y)
                plt.plot(y_pred.detach().numpy())
                if config.wandb:
                    run.log({"wave": plt}, commit=False)
                else:
                    plt.show()
                plt.close()
                plot_spectrogram(
                    sample=y_pred.detach().numpy(),
                    mel_spec=melspec,
                    amp2db=amp2db
                )
                if config.wandb:
                    run.log({"spectrogram": plt}, commit=False)
                else:
                    plt.show()
                plt.close()
            if config.wandb:
                run.log({'loss': loss.item(), 'epoch': i})
        logger.info(
            "Sample {0} final loss equals: {1}".format(id_, loss_fn_list[-1]))
        if config.wandb:
            run.finish()


def build_network(
        input_size: int,
        output: int,
        network: List[int],
        activation,
        bias: bool,
        first_acitvation,
):
    module_list = [
        torch.nn.Linear(input_size, network[0], bias=bias),
        first_acitvation
    ]
    for module_id in range(1, len(network)):
        module_list.append(
            torch.nn.Linear(network[module_id - 1], network[module_id], bias=bias))
        module_list.append(activation)
    module_list.append(torch.nn.Linear(network[-1], output, bias=bias))
    return module_list


def plot_spectrogram(sample, mel_spec, amp2db):
    sample = np.hstack(sample)
    sample = torch.Tensor(sample)
    spectrogram = mel_spec(sample)
    spectrogram = amp2db(spectrogram)
    plt.imshow(spectrogram)
    plt.gca().invert_yaxis()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [ms]')


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])

    # Load config file
    logger.info("Loading config...")
    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    main_config = MainConfig.from_dict(config)
    main(main_config)
