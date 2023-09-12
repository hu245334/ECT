import torch.nn as nn

from models.ECT import ECT


def get_generator(model_config):
    generator_name = model_config["g_name"]
    if generator_name == "ECT":
        model_g = ECT()
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    return nn.DataParallel(model_g)


def get_nets(model_config):
    return get_generator(model_config)
