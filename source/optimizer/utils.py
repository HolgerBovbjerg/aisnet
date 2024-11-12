from torch import nn, optim


def get_optimizer(net: nn.Module, opt_config: dict) -> optim.Optimizer:
    """Creates optimizer based on config.
    Args:
        net (nn.Module): Model instance.
        opt_config (dict): Dict containing optimizer settings.
    Raises:
        ValueError: Unsupported optimizer type.
    Returns:
        optim.Optimizer: Optimizer instance.
    """

    # Extract parameters that require gradients
    parameters = [
        p for model in (net if isinstance(net, list) else [net])
        for p in model.parameters() if p.requires_grad
    ]

    optimizer_classes = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "SGD": optim.SGD
    }

    opt_type = opt_config.get("opt_type")

    if opt_type in optimizer_classes:
        return optimizer_classes[opt_type](parameters, **opt_config["opt_kwargs"])

    raise ValueError(f'Unsupported optimizer {opt_type}')
