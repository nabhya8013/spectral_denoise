import json
import datetime
import os

def log_audit(action, src_file, result_file, description, audit_dir='audit_logs'):
    os.makedirs(audit_dir, exist_ok=True)
    entry = {
        'timestamp': str(datetime.datetime.now()),
        'action': action,
        'source': src_file,
        'result': result_file,
        'description': description
    }
    fname = f"{audit_dir}/{os.path.basename(result_file)}.audit.json"
    with open(fname, 'a') as f:
        f.write(json.dumps(entry) + "\n")

