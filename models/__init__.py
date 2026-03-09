from .detr import build


def build_model(args, token_bank):
    return build(args, token_bank)
