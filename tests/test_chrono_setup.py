"""First-use setup boundaries; scientific inference is exercised separately."""
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'echoforge' / 'chrono_tracker.py'
spec = importlib.util.spec_from_file_location('setup_component', SCRIPT)
component = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = component
spec.loader.exec_module(component)


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_verified_download_is_atomic_and_reusable_offline(self):
        target = self.root / 'model'
        payload = b'trusted checkpoint'
        digest = hashlib.sha256(payload).hexdigest()
        with patch('urllib.request.urlopen', return_value=io.BytesIO(payload)):
            component._download('https://models.invalid/file', target, digest)
        self.assertEqual(target.read_bytes(), payload)
        with patch('urllib.request.urlopen', side_effect=AssertionError('must use cache')):
            component._download('https://models.invalid/file', target, digest)
        self.assertFalse(list(self.root.glob('*.part')))

    def test_corrupt_download_never_replaces_existing_file(self):
        target = self.root / 'model'
        target.write_bytes(b'old')
        with patch('urllib.request.urlopen', return_value=io.BytesIO(b'corrupt')):
            with self.assertRaises(component.ComponentError) as error:
                component._download('https://models.invalid/file', target, '0' * 64)
        self.assertEqual(error.exception.code, 'download_integrity_error')
        self.assertEqual(target.read_bytes(), b'old')
        self.assertFalse(list(self.root.glob('*.part')))

    def test_interrupted_transfer_retries_and_recovers(self):
        payload = b'complete'
        with patch('urllib.request.urlopen', side_effect=[OSError('disconnect'), io.BytesIO(payload)]) as download:
            component._download('https://models.invalid/file', self.root / 'model', hashlib.sha256(payload).hexdigest())
        self.assertEqual(download.call_count, 2)
        self.assertEqual((self.root / 'model').read_bytes(), payload)

    def test_cancellation_never_publishes_a_download(self):
        with patch('urllib.request.urlopen', side_effect=AssertionError('cancel before network')):
            with self.assertRaises(component.ComponentError) as error:
                component._download('https://models.invalid/file', self.root / 'model', '0' * 64, cancelled=lambda: True)
        self.assertEqual(error.exception.code, 'cancelled')
        self.assertFalse((self.root / 'model').exists())

    def test_explicit_deployment_configuration_is_respected(self):
        with patch.dict(os.environ, {'CHRONO_PYTHON': sys.executable}, clear=True), \
                patch.object(component, '_download', side_effect=AssertionError('no automatic setup')):
            self.assertEqual(component._automatic_config().python, sys.executable)

    def test_complete_setup_uses_private_paths_and_works_without_network(self):
        runtime = self.root / component.RUNTIME_VERSION
        (runtime / 'bin').mkdir(parents=True)
        (runtime / 'bin' / 'python').symlink_to(sys.executable)
        identity = dict(version=component.RUNTIME_VERSION, python=component.PYTHON_VERSION,
                        requirements_sha256=hashlib.sha256(component.RUNTIME_REQUIREMENTS.encode()).hexdigest(),
                        installer_sha256=component.UV_SHA256)
        component._json(runtime / 'ready.json', identity)
        model = self.root / 'models' / 'chrono_union' / 'weight.pt'
        model.parent.mkdir(parents=True)
        model.write_bytes(b'weights')
        with patch.dict(os.environ, {'CHRONO_CACHE_DIR': str(self.root)}, clear=True), \
                patch.object(component, 'ASSET_PINS', {'weights_dir': [('weight.pt', hashlib.sha256(b'weights').hexdigest())]}), \
                patch('urllib.request.urlopen', side_effect=AssertionError('offline cache')), \
                patch.object(component, '_setup_command', side_effect=AssertionError('already installed')):
            config = component._automatic_config()
        self.assertEqual(config.python, str(runtime / 'bin' / 'python'))
        self.assertEqual(config.weights_dir, str(model.parent))
        self.assertEqual(config.device, 'cpu')

    def test_invalid_input_does_not_download_runtime(self):
        path = self.root / 'bad.txt'
        path.write_text('invalid')
        with patch.object(component, '_automatic_config', side_effect=AssertionError('invalid input')):
            with self.assertRaises(component.ComponentError) as error:
                component.run_tracking(path, output_dir=self.root / 'output')
        self.assertEqual(error.exception.code, 'invalid_input')

    def test_host_python_paths_do_not_enter_worker(self):
        config = component.Config('/models/a', '/models/b', '/models/c', '/models/d')
        with patch.dict(os.environ, {'PYTHONPATH': '/host/packages', 'PYTHONHOME': '/host/python'}):
            environment = config.environment()
        self.assertNotIn('PYTHONPATH', environment)
        self.assertNotIn('PYTHONHOME', environment)
        self.assertEqual(environment['PYTHONNOUSERSITE'], '1')
        self.assertEqual(environment['PATH'].split(os.pathsep)[0], str(Path(config.python).parent))

    def test_installer_keeps_lock_if_host_is_killed(self):
        import fcntl
        code = '''
import fcntl, importlib.util, os, pathlib, sys
spec=importlib.util.spec_from_file_location('lock_test',sys.argv[1])
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
root=pathlib.Path(sys.argv[2])
with (root/'setup.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 child="import os,pathlib,time;pathlib.Path('child.pid').write_text(str(os.getpid()));time.sleep(30)"
 m._setup_command([sys.executable,'-c',child],root,dict(os.environ),None,None,'test',lock_fd=lock.fileno())
'''
        parent = subprocess.Popen([sys.executable, '-c', code, str(SCRIPT), str(self.root)])
        child = None
        try:
            deadline = time.monotonic() + 10
            while not (self.root / 'child.pid').exists() and time.monotonic() < deadline:
                time.sleep(.02)
            child = int((self.root / 'child.pid').read_text())
            parent.kill()
            parent.wait(timeout=5)
            with (self.root / 'setup.lock').open('a') as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            if child is not None:
                os.killpg(child, signal.SIGKILL)
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
