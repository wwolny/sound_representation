from main import main
from sound_repr.cfg.main_config import MainConfig

main_cfg = MainConfig()

models = ["default", "nerf", "siren"]

# Nerf test różnych L
test_l = [1, 16, 32, 64]

# wielkość sieci default
network_def = [[512, 512, 512, 512, 512]]

# wielkość sieci nerf
network_nerf = [[256, 256, 256, 256, 256]]

# wielkość sieci siren
network_sir = [[256, 256, 256, 256, 256]]

# scale_input
scale_def = [1]
scale_nerf = [1]
scale_siren = [1, 300, 600]

# omega
omega_0 = [1, 30, 50, 100, 1000]
omega_hidden = [1, 30, 50, 100, 1000]

if __name__ == "__main__":
    samples = 10
    epochs = 301
    frame_len = 1
    lr = 0.001

    # model = "default"
    # nets = network_def
    # scale = scale_def
    # def_cfg = MainConfig.from_dict({
    #     "samples": samples,
    #     "epochs": epochs,
    #     "mode": model,
    #     "network": nets[0],
    #     "t_scale_max": scale[0],
    #     "frame_len": frame_len,
    #     "lr": lr,
    # })
    # main(def_cfg)

    model = "nerf"
    nets = network_nerf
    scale = scale_nerf
    L_s = test_l
    for net in nets:
        for test_l in test_l:
            nerf_cfg = MainConfig.from_dict(
                {
                    "samples": samples,
                    "epochs": epochs,
                    "mode": model,
                    "network": net,
                    "t_scale_max": scale[0],
                    "L": test_l,
                }
            )
            main(nerf_cfg)

    model = "siren"
    nets = network_sir
    scale = scale_siren
    for net in nets:
        for sca in scale:
            for o_0 in omega_0:
                for h_o in omega_hidden:
                    siren_cfg = MainConfig.from_dict(
                        {
                            "samples": samples,
                            "epochs": epochs,
                            "mode": model,
                            "network": net,
                            "t_scale_max": sca,
                            "omega": o_0,
                            "hidden_omega": h_o,
                            "lr": 0.0001,
                        }
                    )
                    main(siren_cfg)
