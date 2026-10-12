"""Verified checkpoint uploads with permanent replacement in one Drive folder."""
import hashlib
from pathlib import Path
import uuid


def drive_service():
    import google.auth
    from googleapiclient.discovery import build
    credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive'])
    return build('drive', 'v3', credentials=credentials, cache_discovery=False)


def checksum(path):
    digest = hashlib.md5()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class DriveCheckpoints:
    def __init__(self, folder_id, service=None):
        self.folder_id = folder_id
        self.service = service if service is not None else drive_service()
        folder = self.service.files().get(fileId=folder_id, fields='id,mimeType,trashed').execute()
        if folder['mimeType'] != 'application/vnd.google-apps.folder' or folder.get('trashed'):
            raise ValueError('Checkpoint destination must be an active Drive folder.')

    def files(self, name):
        if name not in ('last.pth', 'best.pth'):
            raise ValueError('Only last.pth and best.pth are managed.')
        result, token = [], None
        while True:
            response = self.service.files().list(
                q=f"'{self.folder_id}' in parents and name = '{name}' and trashed = false",
                fields='nextPageToken,files(id,name,size,md5Checksum,parents)',
                pageToken=token, pageSize=100).execute()
            result.extend(response.get('files', []))
            token = response.get('nextPageToken')
            if not token:
                return result

    def upload(self, path):
        from googleapiclient.http import MediaFileUpload
        path = Path(path)
        previous = self.files(path.name)
        request = self.service.files().create(
            body={'name': path.name + '.upload-' + uuid.uuid4().hex,
                  'parents': [self.folder_id]},
            media_body=MediaFileUpload(str(path), resumable=True, chunksize=64 * 1024 * 1024),
            fields='id,size,md5Checksum')
        uploaded = None
        while uploaded is None:
            _, uploaded = request.next_chunk(num_retries=3)
        if int(uploaded.get('size', -1)) != path.stat().st_size or uploaded.get('md5Checksum') != checksum(path):
            raise RuntimeError('Drive upload verification failed; previous checkpoints were retained.')
        self.service.files().update(fileId=uploaded['id'], body={'name': path.name}, fields='id').execute()
        for old in previous:
            current = self.service.files().get(fileId=old['id'], fields='name,parents,trashed').execute()
            if current['name'] != path.name or self.folder_id not in current.get('parents', []) or current.get('trashed'):
                raise RuntimeError('Previous checkpoint changed; automatic deletion stopped.')
            # Delete only the captured previous checkpoint IDs, after verifying the new upload.
            self.service.files().delete(fileId=old['id']).execute()
        print(f'Drive checkpoint saved: {path.name}', flush=True)

    def download(self, name, destination):
        from googleapiclient.http import MediaIoBaseDownload
        matches = self.files(name)
        if len(matches) != 1:
            raise RuntimeError(f'Expected one {name} in the checkpoint folder; found {len(matches)}.')
        item = matches[0]
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.download')
        try:
            with temporary.open('wb') as stream:
                transfer = MediaIoBaseDownload(stream, self.service.files().get_media(fileId=item['id']),
                                              chunksize=64 * 1024 * 1024)
                done = False
                while not done:
                    _, done = transfer.next_chunk(num_retries=3)
            if temporary.stat().st_size != int(item['size']) or checksum(temporary) != item['md5Checksum']:
                raise RuntimeError('Downloaded checkpoint failed verification.')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
