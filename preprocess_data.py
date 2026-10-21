import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', type = str, default = 'Dubai_CC', help= 'the name of the dataset')
#parser.add_argument('--word_count_threshold', default=0, type=int)
parser.add_argument('--dataset', choices=['LEVIR_CC', 'Dubai_CC'], default='LEVIR_CC')
parser.add_argument('--dataset_root', default=None, help='Root containing captions and images')
parser.add_argument('--output_dir', default=None, help='Prepared token/list directory')
parser.add_argument('--word_count_threshold', default=5, type=int)

SPECIAL_TOKENS = {
  '<NULL>': 0,
  '<UNK>': 1,
  '<START>': 2,
  '<END>': 3,
}

def main(args):
    root = Path(args.dataset_root or ('./datasets/' + args.dataset))
    output = Path(args.output_dir or ('./data/' + args.dataset))
    if args.dataset == 'LEVIR_CC':
        sources = [(root / 'LevirCCcaptions.json', None)]
    else:
        captions = root / 'DubaiCC500impair/datasetDubaiCCPublic/description_jsontr_te_val'
        split_map = {'Train': 'train', 'Validation': 'val', 'Test': 'test'}
        sources = [(path, split_map[path.name.split('_')[0]]) for path in sorted(captions.glob('*.json'))]
    if not sources:
        raise ValueError('No caption JSON files found.')
    all_tokens = []
    lists = {'train': [], 'val': [], 'test': []}
    max_length = 0
    for source, source_split in sources:
        with source.open(encoding='utf-8') as stream:
            rows = json.load(stream)['images']
        for row in rows:
            name = row['filename']
            if Path(name).name != name:
                raise ValueError('Caption filename must be a basename: ' + name)
            split = source_split or name.split('_')[0]
            if split not in lists:
                raise ValueError('Unknown split: ' + split)
            if args.dataset == 'LEVIR_CC':
                for phase in ('A', 'B'):
                    image = root / 'images' / split / phase / name
                    if not image.is_file():
                        raise FileNotFoundError(image)
            tokens = []
            for sentence in row['sentences']:
                if not sentence['raw'].strip():
                    raise ValueError('Empty caption for ' + name)
                caption = tokenize(sentence['raw'], punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
                tokens.append(caption)
                max_length = max(max_length, len(caption))
            if not tokens:
                raise ValueError('No captions for ' + name)
            all_tokens.append((name, tokens))
            lists[split].extend([name + '-' + str(i) for i in range(len(tokens))] if split == 'train' else [name])
    if len({name for name, _ in all_tokens}) != len(all_tokens):
        raise ValueError('Duplicate image filenames in captions.')
    vocab = build_vocab(all_tokens, args.word_count_threshold)
    (output / 'tokens').mkdir(parents=True, exist_ok=True)
    for name, tokens in all_tokens:
        (output / 'tokens' / (Path(name).stem + '.txt')).write_text(json.dumps(tokens), encoding='utf-8')
    # Rewrite lists rather than appending: repeated preprocessing does not duplicate rows.
    for split, names in lists.items():
        (output / (split + '.txt')).write_text(''.join(name + '\n' for name in names), encoding='utf-8')
    (output / 'vocab.json').write_text(json.dumps(vocab), encoding='utf-8')
    manifest = {'dataset': args.dataset, 'word_count_threshold': args.word_count_threshold,
                'max_length': max_length, 'vocab_size': len(vocab),
                'list_rows': {split: len(names) for split, names in lists.items()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


def tokenize(s, delim=' ',add_start_token=True, 
    add_end_token=True,punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    for q in tokens:
        if q == '':
            tokens.remove(q)
    if tokens[0] == '':
        tokens.remove(tokens[0])
    if tokens[-1] == '':
        tokens.remove(tokens[-1])
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens

def build_vocab(sequences, min_token_count=1):#Calculate the number of independent words and tokenize vocab
    token_to_count = {}
    for it in sequences:
        for seq in it[1]:
            for token in seq:
                if token not in token_to_count:
                    token_to_count[token] = 0
                token_to_count[token] += 1

    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if token in token_to_idx.keys():
            continue
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx

def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
