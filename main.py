import os
import sys
import torch
import yaml
import wandb
import logging
import argparse
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from torchaudio.datasets import LJSPEECH
from torchmetrics import SignalNoiseRatio

from sound_repr.dataset import Samples
from sound_repr.utils import Sine, LSD

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=os.environ.get("LOGLEVEL", "INFO"),
)


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    return parser.parse_args(args)


def main():
    args = parse_arguments(sys.argv[1:])

    # Load config file
    logger.info("Loading config...")
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    logger.info("Setting seed {0}".format(config["seed"]))
    torch.manual_seed(config["seed"])

    logger.info("Process dataset...")
    dataset = LJSPEECH('datasets', download=False)

    samples = Samples(dataset, SR=config["SR"], frame_len=config["frame_len"])

    logger.info("Running in mode: {0}".format(config["mode"]))
    for id_ in range(config["samples"]):
        # Initialize wandb
        if config["wandb"]:
            logger.info("Initializing  WandB...")
            run = wandb.init(
                project="sound_representation",
                entity="wwolny",
                config={**config, **{"sample_id": id_}},
                reinit=True,
            )

        logger.info(f"Process sample index: {id_}.")
        data = samples[id_]
        x_numpy = np.array(list(map(lambda el: [el], range(len(data)))))
        if config["mode"] == "siren":
            x_numpy = (x_numpy / len(x_numpy)) * 600 - 300
        elif config["mode"] == "nerf":
            x_numpy = (x_numpy / len(x_numpy)) * 2 - 1
        x = torch.from_numpy(x_numpy).float()

        y_numpy = np.array(list(map(lambda el: [el], data)))
        y = torch.from_numpy(y_numpy).float()

        if config["mode"] == "nerf":
            L = config["L"]
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
                    config["network"],
                    torch.nn.ReLU(),
                    config["bias"]
                )
            )
        elif config["mode"] == "siren":
            module_lst = build_network(
                1,
                1,
                config["network"],
                Sine(),
                config["bias"]
            )
            with torch.no_grad():
                for id_, mod in enumerate(module_lst):
                    if isinstance(mod, torch.nn.Linear):
                        if id_ == 0:
                            mod.weight.uniform_(-1 / mod.in_features, 1 / mod.in_features)
                            mod.bias.uniform_(-1 / mod.in_features, 1 / mod.in_features)
                        else:
                            mod.weight.uniform_(-np.sqrt(6 / mod.in_features) / 30., np.sqrt(6 / mod.in_features) / 30.)
                            mod.bias.uniform_(-np.sqrt(6 / mod.in_features) / 30., np.sqrt(6 / mod.in_features) / 30.)
            model = torch.nn.Sequential(
                *module_lst
            )
        elif config["mode"] == "default":
            model = torch.nn.Sequential(
                *build_network(
                    1,
                    1,
                    config["network"],
                    torch.nn.ReLU(),
                    config["bias"]
                )
            )
        else:
            exit()

        if config["wandb"]:
            run.watch(model)

        loss_fn = getattr(torch.nn, config["loss_fn"])()
        optimizer = getattr(torch.optim, config["optimizer"])(
            model.parameters(), lr=config["lr"])

        # train
        batch_size = config["batch_size"]
        loss_fn_list = []
        loss_lst = {}
        for el in config["loss_lst"]:
            loss_lst[el] = getattr(torch.nn, el)()
        if config["SNR"]:
            loss_lst["SNR"] = SignalNoiseRatio()
        for i in range(config["epochs"]):
            if config["batched"]:
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
                if config["wandb"]:
                    for el in loss_lst.keys():
                        run.log({el: loss_lst[el](y_pred, y).item(), 'epoch': i})
            else:
                optimizer.zero_grad()

                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                loss_fn_list.append(loss.item())
                if config["wandb"]:
                    for el in loss_lst.keys():
                        run.log({el: loss_lst[el](y_pred, y).item(), 'epoch': i})

                loss.backward()
                optimizer.step()

            if config["plot"] and i % config["log_img"] == 0:
                y_pred = model(x)
                plt.plot(y)
                plt.plot(y_pred.detach().numpy())
                if config["wandb"]:
                    run.log({"chart": plt})
                else:
                    plt.show()
            if config["wandb"]:
                run.log({'loss': loss.item(), 'epoch': i})
        logger.info(
            "Sample {0} final loss equals: {1}".format(id_, loss_fn_list[-1]))
        if config["wandb"]:
            run.finish()


def build_network(
        input_size: int,
        output: int,
        network: List[int],
        activation,
        bias: bool,
):
    module_list = [
        torch.nn.Linear(input_size, network[0], bias=bias),
        activation
    ]
    for module_id in range(1, len(network)):
        module_list.append(
            torch.nn.Linear(network[module_id - 1], network[module_id], bias=bias))
        module_list.append(activation)
    module_list.append(torch.nn.Linear(network[-1], output, bias=bias))
    return module_list

if __name__ == '__main__':
    main()
