"""Record exact operator-built libraries without loading or qualifying them."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--native-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
names = ['libbs_microflow_kkt.so', 'libbs_microflow_wu.so']
result = {'schema': 'prefill-microflow-native-v1', 'libraries': {}}
for name in names:
    result['libraries'][name] = hashlib.sha256((a.native_dir / name).read_bytes()).hexdigest()
with a.output.open('x', encoding='utf8', newline='\n') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
