import json, csv, os, time, math
from pathlib import Path

class ExperimentLogger:
    def __init__(self, root_dir, model_name):
        ts = time.strftime('%Y%m%d_%H%M%S')
        self.dir = Path(root_dir) / 'results' / model_name / ts
        self.dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.dir / 'history.csv'
        self._init_history()
    def _init_history(self):
        with open(self.history_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['iteration','h','grad_norm','step_size','weights'])
    def log_history_row(self, rec):
        with open(self.history_file, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([rec['iteration'], rec['h(w)'], rec['grad_norm'], rec['step_size'], ' '.join(map(str, rec['weights']))])
    def save_config(self, cfg: dict):
        with open(self.dir / 'config.json','w') as f:
            json.dump(cfg, f, indent=2)
    def save_json(self, name, data):
        with open(self.dir / f'{name}.json','w') as f:
            json.dump(data, f, indent=2)
    def path(self, name):
        return str(self.dir / name)
