# LDNet and CLDNet configurations

`ldnet/{illinois,texas}.json` and `cldnet/{illinois,texas}.json` describe the released LDNet and CLDNet
checkpoints; `fno/run_config.json` is the Texas FNO run (see `code/fno/README.md`) and `vae_convlstm/texas.json`
the Texas VAE–ConvLSTM (see `code/vae_convlstm/README.md`). Run the LDNet/CLDNet
`inference` command arrays from the repository root. All paths
are repository relative. The Illinois full-grid commands use `data/sims_30`.

The architecture and inference flags were checked against the saved `dyn`, `rec`, and
Fourier `B` states. Illinois epoch 539 uses three output channels (`h`, `hu`, `hv`),
200 latent states, and Fourier size 32. Texas (CLDNet epoch 489, LDNet epoch 549) uses
30 latent states and Fourier size 10. CLDNet appends static inputs; Texas uses rain normalization.

The original training commands, seeds, validation selection, and schedulers are
unknown. A supplied H200 command was a template, not the command that produced the
Illinois epoch 539 weights. Each Illinois JSON therefore has a separate
`training.proposed_command` for a new run, with `--all-vars`, Fourier size 32, and
`splits/illinois_split.json`. That split contains 90 training events (including 106),
events 107–109 for both validation and test, and held-out 2013 event 120. Because
validation and test share events, their metrics are not an independent test. The proposed run
writes under `outputs/training/`, away from the deposited checkpoints. Its results
would be a new experiment, not a reproduction of the original checkpoint training.

From the repository root, `sbatch configs/illinois_h200.sbatch ldnet` or
`sbatch configs/illinois_h200.sbatch cldnet` launches the updated H200 template.

Run an array as a command, for example:

```bash
python3 - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

config = json.loads(Path('configs/cldnet/illinois.json').read_text())
command = config['inference']['reduced_test_command'].copy()
command[0] = sys.executable
subprocess.run(command, check=True)
PY
```
