import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT.parent / 'echoforge' / 'chrono_tracker.py'
spec = importlib.util.spec_from_file_location('component_under_test', SCRIPT)
component = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = component
spec.loader.exec_module(component)

# A real child executable exercises supervision without invoking scientific work.
FAKE = r'''
import hashlib,json,os,pathlib,sys,time
p=pathlib.Path(sys.argv[-1]);r=json.loads(p.read_text());out=pathlib.Path(r['output_dir'])
mode=pathlib.Path(r['input_path']).read_text()
if mode=='crash' and r['attempt']==1: os._exit(9)
if mode=='crash_forever': os._exit(9)
if mode=='hang':
 pathlib.Path(r['input_path']+'.pid').write_text(str(os.getpid()))
 time.sleep(60)
if mode=='wait_for_supervisor':
 release=pathlib.Path(r['input_path']+'.release')
 while not release.exists(): time.sleep(.02)
if mode=='child':
 import subprocess
 child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'])
 pathlib.Path(r['input_path']+'.pid').write_text(str(child.pid))
 time.sleep(.2)
receipt=dict(request_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),status='complete',cache_key=r['cache_key'],component_sha256=r['component_sha256'])
if mode=='fail': receipt.update(status='failed',code='test_deterministic',message='Fixture failure')
else:
 (out/'descriptor.json').write_text(json.dumps(dict(status='complete',num_frames=3)))
 (out/'viewer.html').write_text('<html>fixture</html>')
 receipt['artifacts']=[dict(name=str(f.relative_to(out)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())for f in sorted(out.rglob('*')) if f.is_file()]
 if mode=='tamper':(out/'viewer.html').write_text('altered')
pathlib.Path(r['receipt_path']).write_text(json.dumps(receipt))
'''


class ComponentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'cine.avi';self.source.write_text('success')
        self.output=self.root/'output';self.output.mkdir();(self.output/'EF.json').write_text('unchanged')
        self.fake=self.root/'worker.py';self.fake.write_text(FAKE)
        self.config=component.Config(str(self.root),str(self.root),str(self.root),str(self.root/'model'),timeout_seconds=5,idle_seconds=4)

    def run_fixture(self,mode='success',**kwargs):
        self.source.write_text(mode)
        with patch.object(component,'__file__',str(self.fake)),patch.object(component,'_assets',return_value=[]):
            return component.run_tracking(self.source,output_dir=self.output,config=kwargs.pop('config',self.config),public_base_url='/api/jobs/abc/files',**kwargs)

    def test_cold_import_is_stdlib_only(self):
        code="import runpy,sys;runpy.run_path(sys.argv[1],run_name='portable_test');assert not any(n in sys.modules for n in ('torch','cv2','numpy','tensorflow','onnxruntime'))"
        subprocess.run([sys.executable,'-S','-c',code,str(SCRIPT)],cwd='/tmp',check=True)

    def test_engine_byte_identity(self):
        import hashlib
        self.assertEqual(hashlib.sha256(component.ENGINE_SOURCE.encode()).hexdigest(),component.ENGINE_SHA256)

    def test_complete_is_atomic_and_leaves_host_artifacts(self):
        result=self.run_fixture();self.assertEqual(result['status'],'complete')
        self.assertEqual((self.output/'EF.json').read_text(),'unchanged')
        self.assertTrue(result['viewer_url'].startswith('/api/jobs/abc/files/chrono/'))
        self.assertTrue((self.output/'chrono'/result['cache_key']/'receipt.json').is_file())

    def test_authenticated_cache_skips_new_worker(self):
        result=self.run_fixture()
        with patch.object(component,'__file__',str(self.fake)),patch.object(component,'_assets',return_value=[]),patch.object(component.subprocess,'Popen',side_effect=AssertionError('must reuse')):
            self.assertEqual(component.run_tracking(self.source,output_dir=self.output,config=self.config,public_base_url='/api/jobs/abc/files'),result)

    def test_tampered_cache_not_reused_or_overwritten(self):
        result=self.run_fixture();p=self.output/'chrono'/result['cache_key']/'viewer.html';p.write_text('changed')
        with self.assertRaises(component.ComponentError) as ctx:self.run_fixture()
        self.assertEqual(ctx.exception.code,'invalid_cached_result');self.assertEqual(p.read_text(),'changed')

    def test_native_crash_retries_once(self):
        self.assertEqual(self.run_fixture('crash')['status'],'complete')

    def test_native_crash_has_bounded_attempts(self):
        with self.assertRaises(component.ComponentError) as ctx:self.run_fixture('crash_forever')
        self.assertEqual(ctx.exception.code,'worker_crashed')

    def test_deterministic_failure_not_published(self):
        with self.assertRaises(component.ComponentError) as ctx:self.run_fixture('fail')
        self.assertEqual(ctx.exception.code,'test_deterministic')
        self.assertFalse(list((self.output/'chrono').glob('*/receipt.json')))

    def test_changed_worker_artifact_rejected(self):
        with self.assertRaises(component.ComponentError) as ctx:self.run_fixture('tamper')
        self.assertEqual(ctx.exception.code,'invalid_worker_receipt')

    def test_deadline_terminates_authenticated_worker(self):
        for total, idle in ((.4, 0), (0, .4)):
            with self.subTest(total=total, idle=idle):
                config = dataclasses.replace(self.config, timeout_seconds=total, idle_seconds=idle)
                with self.assertRaises(component.ComponentError) as error:
                    self.run_fixture('hang', config=config)
                self.assertEqual(error.exception.code, 'processing_timeout')
                pid = int(self.source.with_suffix('.avi.pid').read_text())
                self.assertIsNone(component.birth(pid))

    def test_default_deadlines_allow_long_silent_tracking(self):
        release = self.source.with_suffix('.avi.release')
        with patch.dict(os.environ, {}, clear=True):
            default = component.Config.from_environment()
        config = dataclasses.replace(self.config, timeout_seconds=default.timeout_seconds,
                                     idle_seconds=default.idle_seconds)
        samples = 0

        def fast_forward():
            nonlocal samples
            samples += 1
            if samples == 1:
                return time.monotonic()
            release.touch(exist_ok=True)
            return time.monotonic() + 7200

        with patch.object(component, 'time', SimpleNamespace(monotonic=fast_forward, sleep=time.sleep)):
            self.assertEqual(self.run_fixture('wait_for_supervisor', config=config)['status'], 'complete')
        self.assertTrue(release.is_file())

    def test_budget_opt_in_is_forwarded_to_worker(self):
        variables = {'CHRONO_TIMEOUT_SECONDS': '2.5', 'CHRONO_IDLE_SECONDS': '1',
                     'CHRONO_MAX_DECODE_BYTES': '700', 'CHRONO_WORKER_ADDRESS_MB': '4096'}
        with patch.dict(os.environ, {}, clear=True):
            defaults = component.Config.from_environment()
            self.assertEqual(defaults.environment()['STUDIO_MAX_DECODE_BYTES'], '0')
            self.assertEqual(defaults.address_mb, 0)
            with patch.dict(os.environ, variables):
                config = component.Config.from_environment()
                config.validate()
                self.assertEqual(config.environment()['STUDIO_MAX_DECODE_BYTES'], '700')
        self.assertEqual((config.timeout_seconds, config.idle_seconds, config.address_mb), (2.5, 1, 4096))

    def test_default_decode_budget_accepts_large_native_payload(self):
        io = component._io()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(io._decode_byte_limit(), 0)
            io._require_decode_budget(1024 ** 3, io._decode_byte_limit())
            with patch.dict(os.environ, {'STUDIO_MAX_DECODE_BYTES': '536870912'}):
                with self.assertRaisesRegex(ValueError, 'no frames were trimmed'):
                    io._require_decode_budget(1024 ** 3, io._decode_byte_limit())

    def test_metadata_probe_waits_for_normal_completion(self):
        io = component._io()
        with patch('shutil.which', return_value='/usr/bin/ffprobe'), \
                patch('subprocess.run', return_value=SimpleNamespace(stdout='{"streams":[{"nb_frames":"42"}]}')) as run:
            self.assertEqual(io._declared_video_frame_count('cine.avi'), 42)
            self.assertNotIn('timeout', run.call_args.kwargs)

    def test_native_descendants_are_cleaned(self):
        self.run_fixture('child');pid=int(Path(str(self.source)+'.pid').read_text())
        try:state=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
        except OSError:state='gone'
        self.assertIn(state,('gone','Z'))

    def test_cancelled_job_never_publishes(self):
        with self.assertRaises(component.ComponentError) as ctx:self.run_fixture(cancelled=lambda:True)
        self.assertEqual(ctx.exception.code,'cancelled')

    def test_failure_adapter_does_not_raise(self):
        with patch.object(component,'run_tracking',side_effect=component.ComponentError('models_unavailable','Missing weights')):
            self.assertEqual(component.attach_tracking(self.source,output_dir=self.output)['status'],'unavailable')

    def test_nonfinite_deadline_rejected(self):
        with self.assertRaises(component.ComponentError):dataclasses.replace(self.config,timeout_seconds=float('nan')).validate()

    def test_worker_environment_does_not_change_parent(self):
        before=dict(os.environ);env=self.config.environment();self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(dict(os.environ),before)

    def test_trusted_asset_mutation_rejected_before_inference(self):
        from hashlib import sha256
        p=self.root/'weight';p.write_text('altered')
        with patch.object(component,'ASSET_PINS',{'weights_dir':[('weight',sha256(b'trusted').hexdigest())]}):
            with self.assertRaises(component.ComponentError) as ctx:component._assets(self.config)
        self.assertEqual(ctx.exception.code,'model_integrity_error')

    def test_bad_public_url_rejected(self):
        result=self.run_fixture();p=self.output/'chrono'/result['cache_key']
        with self.assertRaises(component.ComponentError):component._descriptor(p,result['cache_key'],'https://other.invalid')


if __name__=='__main__':unittest.main(verbosity=2)
