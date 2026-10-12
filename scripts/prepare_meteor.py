"""Download the METEOR resource and verify its SHA-256."""
import hashlib
from pathlib import Path
import urllib.request

URL = 'https://raw.githubusercontent.com/salaniz/pycocoevalcap/master/meteor/data/paraphrase-en.gz'
SHA256 = 'c147ac7d2c91f2fbb3ad31e4b352235061eb83145e0434daf217ee9ca5975f48'


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def main():
    target = Path(__file__).resolve().parents[1] / 'eval_func/meteor/data/paraphrase-en.gz'
    if target.is_file() and digest(target) == SHA256:
        print('METEOR resource verified:', target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.download')
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if digest(temporary) != SHA256:
            raise RuntimeError('METEOR resource checksum mismatch; existing file was preserved.')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print('METEOR resource downloaded and verified:', target)


if __name__ == '__main__':
    main()
