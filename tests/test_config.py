from sound_repr.cfg.main_config import MainConfig


def test_main_config_default():
    main_config = MainConfig()
    assert main_config.seed == 0
    assert main_config.name == "LJSPEECH"
    assert main_config.batch_size == 64


def test_config_from_dict():
    main_config = MainConfig()
    assert main_config.seed == 0
    test_dict = {"seed": 10}
    test_config = MainConfig.from_dict(test_dict)
    assert test_config.seed == 10
