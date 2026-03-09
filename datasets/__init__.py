from .eeg import UnifiedEEGDataset, load_token_bank


def build_dataset(split, args, token_bank):
    return UnifiedEEGDataset(args=args, split=split, token_bank=token_bank)
