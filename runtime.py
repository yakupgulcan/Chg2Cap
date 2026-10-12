"""Runtime records and portable checkpoint utilities."""
import json
import os
import platform
import subprocess
from pathlib import Path

import torch
import torchvision


MODEL_ARGUMENTS = ('network', 'encoder_dim', 'feat_size', 'n_heads', 'n_layers',
                   'decoder_n_layers', 'hidden_dim', 'attention_dim', 'feature_dim',
                   'max_length', 'dropout', 'data_name', 'fine_tune_encoder')


def command_output(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_info():
    return {
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
        'cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'nvidia_smi': command_output(['nvidia-smi']),
        'java': command_output(['java', '-version']),
        'git_commit': command_output(['git', 'rev-parse', 'HEAD']),
        'git_status': command_output(['git', 'status', '--short']),
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_state(path):
    # State dictionaries only; do not deserialize legacy pickled model objects.
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def validate_checkpoint(checkpoint, args, vocab):
    if checkpoint.get('vocab', vocab) != vocab:
        raise ValueError('Checkpoint vocabulary differs from the prepared vocabulary.')
    config = checkpoint.get('config', {})
    for name in MODEL_ARGUMENTS:
        if name in config and hasattr(args, name) and config[name] != getattr(args, name):
            raise ValueError(f'Checkpoint requires --{name} {config[name]!r}.')
