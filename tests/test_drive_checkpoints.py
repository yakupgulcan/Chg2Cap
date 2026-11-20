import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

from drive_checkpoints import DriveCheckpoints, checksum


class DriveCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'last.pth'
        self.path.write_bytes(b'checkpoint contents')
        self.service = MagicMock()
        self.files = self.service.files.return_value
        self.files.get.return_value.execute.return_value = {
            'mimeType': 'application/vnd.google-apps.folder',
        }
        self.store = DriveCheckpoints('run-folder', self.service)
        self.files.list.return_value.execute.return_value = {'files': [{'id': 'previous'}]}
        self.files.get.return_value.execute.return_value = {
            'name': self.path.name, 'parents': ['run-folder'], 'trashed': False,
        }
        self.files.create.return_value.next_chunk.return_value = (None, {
            'id': 'uploaded', 'size': str(self.path.stat().st_size),
            'md5Checksum': checksum(self.path),
        })
        # No Google dependency or network access is needed for these tests.
        googleapi = ModuleType('googleapiclient')
        http = ModuleType('googleapiclient.http')
        http.MediaFileUpload = MagicMock()
        self.modules = patch.dict('sys.modules', {'googleapiclient': googleapi,
                                                  'googleapiclient.http': http})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_verified_upload_deletes_only_captured_previous_id(self):
        self.store.upload(self.path)
        self.files.update.assert_called_once_with(
            fileId='uploaded', body={'name': self.path.name}, fields='id')
        self.files.delete.assert_called_once_with(fileId='previous')

    def test_checksum_failure_retains_previous_checkpoint(self):
        self.files.create.return_value.next_chunk.return_value[1]['md5Checksum'] = 'wrong'
        with self.assertRaisesRegex(RuntimeError, 'verification failed'):
            self.store.upload(self.path)
        self.files.delete.assert_not_called()
        self.files.update.assert_not_called()

    def test_upload_failure_retains_previous_checkpoint(self):
        self.files.create.return_value.next_chunk.side_effect = OSError('upload failed')
        with patch('drive_checkpoints.time.sleep'), self.assertRaises(OSError):
            self.store.upload(self.path)
        self.files.delete.assert_not_called()

    def test_network_error_during_a_chunk_resumes_the_upload(self):
        chunk = self.files.create.return_value.next_chunk
        chunk.side_effect = [TimeoutError('The read operation timed out'), (None, None), chunk.return_value]
        with patch('drive_checkpoints.time.sleep') as sleep:
            self.store.upload(self.path)
        self.assertEqual(chunk.call_count, 3)
        sleep.assert_called_once_with(1)
        self.files.delete.assert_called_once_with(fileId='previous')

    def test_changed_previous_file_is_not_deleted(self):
        self.files.get.return_value.execute.return_value['parents'] = ['other-folder']
        with self.assertRaisesRegex(RuntimeError, 'automatic deletion stopped'):
            self.store.upload(self.path)
        self.files.delete.assert_not_called()

    def test_delete_failure_is_reported(self):
        self.files.delete.return_value.execute.side_effect = OSError('delete failed')
        with self.assertRaises(OSError):
            self.store.upload(self.path)

    def test_unmanaged_filename_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.files('unrelated.pth')
        self.files.list.assert_not_called()


if __name__ == '__main__':
    unittest.main()
