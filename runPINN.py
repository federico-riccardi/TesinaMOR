#!/usr/bin/env python3

import yaml
from yaml.loader import SafeLoader
import subprocess

# Open the file and load the file
with open('Configurazioni.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

for iter in data['iterations']:
    for l in data['coeff']:
        for p in data['points']:
            subprocess.run(["./PINN.py", "--coeff1", str(l[1]), "--coeff2", str(l[2]), "--coeff3", str(l[3]), "--coeff4", str(l[4]), "--iterations", str(iter), "--points", str(p)])

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("\n")

