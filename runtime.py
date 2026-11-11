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
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
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


def run_logged(command, log_path, cwd=None):
    """Display merged stdout/stderr and persist it; stop the child on interruption.

    The display gets the raw output, so progress bars update in place. The log gets
    only the final state of each line: a progress bar is one line instead of thousands.
    """
    import codecs
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pending = ''

    def final_state(line):
        return line.rstrip('\r').rsplit('\r', 1)[-1]

    with log_path.open('a', encoding='utf-8', newline='') as log:
        log.write(f"\nStarting: {command!r}\n")
        log.flush()
        # Binary reads keep carriage returns; text mode would turn them into newlines.
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            try:
                decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                while chunk := process.stdout.read1(8192):
                    output = decoder.decode(chunk)
                    print(output, end='', flush=True)
                    *lines, pending = (pending + output).split('\n')
                    log.writelines(final_state(line) + '\n' for line in lines)
                    log.flush()
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            finally:
                if final_state(pending):
                    log.write(final_state(pending) + '\n')
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_archive(source, destination, log_path):
    """Copy a local/Drive ZIP or download a share URL; returns (path, sha256)."""
    import shutil
    import sys
    source, destination = str(source), Path(destination)
    if Path(source).is_file():
        shutil.copy2(source, destination)
    elif source.startswith(('https://', 'http://')):
        gdown = [sys.executable, '-u', '-m', 'gdown']
        # Older gdown needs --fuzzy for share links; newer versions parse them and reject the flag.
        if '--fuzzy' in subprocess.check_output(gdown + ['--help'], text=True):
            gdown.append('--fuzzy')
        run_logged(gdown + [source, '-O', str(destination)], log_path)
    else:
        raise FileNotFoundError(f'Expected an existing ZIP path or download URL: {source}')
    return destination, sha256(destination)


def extract_raw_dataset(archive, destination):
    """Extract a LEVIR-CC-layout ZIP; returns the folder holding the captions file and images/."""
    import zipfile
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(destination)
    candidates = [p.parent for p in Path(destination).rglob('LevirCCcaptions.json')
                  if (p.parent / 'images').is_dir()]
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected one LevirCCcaptions.json next to images/; found {len(candidates)}.')
    return candidates[0]


def start_run(run_dir, config, repo_dir='.'):
    """Create the run folder and its single record, run.json."""
    import sys
    from datetime import datetime, timezone

    def git(*arguments):
        return subprocess.check_output(['git', *arguments], cwd=repo_dir, text=True).strip()

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    record = {
        'run_id': run_dir.name, 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'code': {'repo_url': git('remote', 'get-url', 'origin'), 'commit': git('rev-parse', 'HEAD'),
                 'uncommitted_changes': bool(git('status', '--porcelain'))},
        'config': config,
        'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'torchvision': str(torchvision.__version__), 'cuda_build': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }
    write_json(run_dir / 'run.json', record)
    (run_dir / 'packages.txt').write_text(
        subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    return record


def update_run(run_dir, **fields):
    """Merge fields into run.json and return the updated record."""
    path = Path(run_dir) / 'run.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    record.update(fields)
    write_json(path, record)
    return record


def training_summary(checkpoint_dir):
    """Best-validation epoch and epoch count from the training_epoch_*.json records."""
    records = [json.loads(path.read_text(encoding='utf-8'))
               for path in sorted(Path(checkpoint_dir).glob('training_epoch_*.json'))]
    best = max(records, key=lambda r: r['validation']['Bleu_4'])
    return {'epochs_completed': records[-1]['epoch'] + 1, 'best_epoch': best['epoch'] + 1,
            'best_validation': best['validation']}


def validate_checkpoint(checkpoint, args, vocab):
    if checkpoint.get('vocab', vocab) != vocab:
        raise ValueError('Checkpoint vocabulary differs from the prepared vocabulary.')
    config = checkpoint.get('config', {})
    for name in MODEL_ARGUMENTS:
        if name in config and hasattr(args, name) and config[name] != getattr(args, name):
            raise ValueError(f'Checkpoint requires --{name} {config[name]!r}.')
