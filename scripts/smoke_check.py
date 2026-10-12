"""Exercise real GPU training, optional resume and test on small split lists."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='./datasets/LEVIR_CC')
    parser.add_argument('--list_path', default='./data/LEVIR_CC')
    parser.add_argument('--output_dir', default='./runs/smoke')
    parser.add_argument('--resume_check', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source = Path(args.list_path).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a new smoke output directory: ' + str(output))
    lists = output / 'lists'
    lists.mkdir(parents=True)
    for split in ('train', 'val', 'test'):
        rows = (source / (split + '.txt')).read_text().splitlines()
        assert len(rows) >= 4, f'{split} needs at least four rows'
        (lists / (split + '.txt')).write_text('\n'.join(rows[:4]) + '\n')
    (lists / 'vocab.json').write_bytes((source / 'vocab.json').read_bytes())
    common = ['--data_folder', str(Path(args.dataset_root).resolve() / 'images'),
              '--list_path', str(lists), '--token_folder', str(source / 'tokens'), '--workers', '0']
    checkpoints = output / 'checkpoints'
    def run(script, options):
        subprocess.run([sys.executable, '-u', str(project / script), *common, *options], cwd=project, check=True)
    run('train.py', ['--num_epochs', '1', '--train_batchsize', '2', '--print_freq', '1',
                     '--savepath', str(checkpoints)])
    if args.resume_check:
        run('train.py', ['--num_epochs', '2', '--train_batchsize', '2', '--print_freq', '1',
                         '--savepath', str(checkpoints), '--checkpoint', str(checkpoints / 'last.pth')])
    run('test.py', ['--savepath', str(checkpoints), '--checkpoint', 'best.pth',
                    '--results_dir', str(output / 'results'), '--print_freq', '1'])
    metrics = json.loads((output / 'results/metrics.json').read_text())
    assert metrics['pairs'] == 4
    print('PASS: small real-data training, checkpoint reload and evaluation' +
          (' including epoch resume.' if args.resume_check else '.'))


if __name__ == '__main__':
    main()
