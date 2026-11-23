import importlib.util,sys,tempfile,json,hashlib,copy
from pathlib import Path
from unittest import TestCase,main
from unittest.mock import patch
import numpy as np
import time
HERE=Path(__file__).resolve().parent
p=HERE.parent/'echoforge'/'chrono_tracker.py';s=importlib.util.spec_from_file_location('execution_component',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
class ReuseTests(TestCase):
 def fixture(self,root,component_sha256=None):
  staging=root/'new/output/chrono/.stage';staging.mkdir(parents=True)
  identity=dict(version=m.COMPONENT_VERSION,component_sha256=m._sha(m.__file__),input_sha256='input',input_bytes=19,config={'threads':1},assets=[])
  oldid=dict(identity,component_sha256=component_sha256 or identity['component_sha256'])
  key=hashlib.sha256(json.dumps(oldid,sort_keys=True).encode()).hexdigest();old=root/'old/output/chrono'/key;old.mkdir(parents=True)
  (old/'tracking.json').write_text('{"test":1}');(old/'frames').mkdir();(old/'frames/000000.png').write_bytes(b'original pixels')
  receipt=dict(status='complete',cache_key=key,component_sha256=oldid['component_sha256'],input_sha256='input',engine_sha256=m.ENGINE_SHA256,shared_recovery_version=62,recovery_kernel_revision=9,scientific_source_sha256={n:d for n,(_,d) in m.REPAIR_SOURCES.items()},artifacts=m._manifest(old),acquired_frames=1)
  (old/'receipt.json').write_text(json.dumps(receipt))
  return dict(identity,output_dir=str(staging),attempt=1),staging,old,receipt
 def invoke(self,request,receipt):
  with patch.object(m,'_assets',return_value=[]):return m._reuse_completed_upload(request,None,receipt)
 def test_identical_complete_result_reused_with_independent_files(self):
  with tempfile.TemporaryDirectory() as tmp:
   req,staging,old,_=self.fixture(Path(tmp));receipt={};self.assertTrue(self.invoke(req,receipt))
   self.assertEqual((staging/'frames/000000.png').read_bytes(),b'original pixels')
   (staging/'tracking.json').write_text('new-job mutation');self.assertEqual((old/'tracking.json').read_text(),'{"test":1}')
 def test_different_input_is_not_reused(self):
  with tempfile.TemporaryDirectory() as tmp:
   req,_,_,_=self.fixture(Path(tmp));req['input_sha256']='different';self.assertFalse(self.invoke(req,{}))
 def test_previous_source_selection_is_not_reused(self):
  with tempfile.TemporaryDirectory() as tmp:
   previous='a497c67d3eb5a6f370d6f3466ae7fdcb03bef4cfc7e2f7bd1496cacdb95d92c0'
   req,_,_,_=self.fixture(Path(tmp),component_sha256=previous)
   self.assertFalse(self.invoke(req,{}))
 def test_corrupt_prior_data_is_not_reused(self):
  with tempfile.TemporaryDirectory() as tmp:
   req,_,old,_=self.fixture(Path(tmp));(old/'tracking.json').write_text('changed');self.assertFalse(self.invoke(req,{}))
 def test_different_scientific_source_is_not_reused(self):
  with tempfile.TemporaryDirectory() as tmp:
   req,_,old,r=self.fixture(Path(tmp));r['scientific_source_sha256']['recovery']='wrong';(old/'receipt.json').write_text(json.dumps(r));self.assertFalse(self.invoke(req,{}))
 def test_linked_prior_artifact_falls_back_to_fresh_inference(self):
  with tempfile.TemporaryDirectory() as tmp:
   req,_,old,_=self.fixture(Path(tmp));(old/'linked').symlink_to(old/'tracking.json');self.assertFalse(self.invoke(req,{}))
 def test_no_new_cross_job_retention_tree(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);req,_,_,_=self.fixture(root);self.invoke(req,{});self.assertEqual({p.name for p in root.iterdir()},{'old','new'})
 def test_memory_cache_is_input_bound_and_cannot_be_mutated_by_caller(self):
  from types import SimpleNamespace
  count={'hash':0,'model':0}
  def digest(p):count['hash']+=1;return hashlib.sha256(Path(p).read_bytes()).hexdigest()
  def model(frame,*args,**kwargs):count['model']+=1;return np.asarray(frame).copy(),{'unchanged':True}
  e=SimpleNamespace(_file_sha256=digest,_conditional_fields_v55=model,predict_cavity_mask=model,predict_wall_probability=model)
  m._install_worker_execution_cache(e)
  image=np.zeros((4,4),np.uint8);v=e._conditional_fields_v55(image);v[0][0,0]=5
  self.assertEqual(int(e._conditional_fields_v55(image)[0][0,0]),0);self.assertEqual(count['model'],1)
  image[0,0]=1;e._conditional_fields_v55(image);self.assertEqual(count['model'],2)
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'model';p.write_bytes(b'old');time.sleep(1.02);self.assertEqual(e._file_sha256(p),e._file_sha256(p));self.assertEqual(count['hash'],1)
   p.write_bytes(b'new');self.assertEqual(e._file_sha256(p),hashlib.sha256(b'new').hexdigest());self.assertEqual(count['hash'],2)
if __name__=='__main__':main()
