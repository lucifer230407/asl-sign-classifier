"""
Run this ONCE after training to patch the batch_shape bug in the .h5 file.
This makes the model loadable with tf_keras without errors.
"""
import h5py
import json
import shutil
import os

BASE  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "asl_model.h5")
FIXED_PATH = os.path.join(BASE, "model", "asl_model_fixed.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")

shutil.copy(MODEL_PATH, FIXED_PATH)
print(f"Copied model to {FIXED_PATH}")

with h5py.File(FIXED_PATH, 'r+') as f:
    cfg = f.attrs['model_config']
    if isinstance(cfg, bytes):
        cfg = cfg.decode('utf-8')

    config = json.loads(cfg)

    def fix(obj):
        if isinstance(obj, dict):
            if 'batch_shape' in obj:
                obj['batch_input_shape'] = obj.pop('batch_shape')
            for v in obj.values():
                fix(v)
        elif isinstance(obj, list):
            for i in obj:
                fix(i)

    fix(config)
    f.attrs['model_config'] = json.dumps(config).encode('utf-8')

print(f"✅ Fixed model saved to {FIXED_PATH}")
