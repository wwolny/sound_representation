from main import main
from cfg import MainConfig

if __name__ == '__main__':
    samples = 10
    epochs = 301
    frame_len = 2
    lr = 0.0001
    network = [256, 256, 256, 256, 256]

    def_cfg = MainConfig.from_dict({
        "samples": samples,
        "epochs": epochs,
        "mode": "default",
        "network": network,
        "t_scale_max": 1,
        "frame_len": frame_len,
        "lr": lr,
        "name": "GTZAN",
    })
    main(def_cfg)

    nerf_cfg = MainConfig.from_dict({
        "samples": samples,
        "epochs": epochs,
        "mode": "nerf",
        "network": network,
        "t_scale_max": 1,
        "L": 16,
        "name": "GTZAN",
    })
    main(nerf_cfg)

    siren_cfg = MainConfig.from_dict({
        "samples": samples,
        "epochs": epochs,
        "mode": "siren",
        "network": network,
        "t_scale_max": 300,
        "omega": 30,
        "hidden_omega": 100,
        "lr": 0.0001,
        "name": "GTZAN",
    })
    main(siren_cfg)

    def_cfg = MainConfig.from_dict({
        "samples": samples,
        "epochs": epochs,
        "mode": "default",
        "network": network,
        "t_scale_max": 1,
        "frame_len": frame_len,
        "lr": lr,
        "name": "ESC50",
    })
    main(def_cfg)

    nerf_cfg = MainConfig.from_dict({
        "samples": samples,
        "epochs": epochs,
        "mode": "nerf",
        "network": network,
        "t_scale_max": 1,
        "L": 16,
        "name": "ESC50",
    })
    main(nerf_cfg)

    siren_cfg = MainConfig.from_dict({
        "samples": samples,
        "epochs": epochs,
        "mode": "siren",
        "network": network,
        "t_scale_max": 300,
        "omega": 30,
        "hidden_omega": 100,
        "lr": 0.0001,
        "name": "ESC50",
    })
    main(siren_cfg)
