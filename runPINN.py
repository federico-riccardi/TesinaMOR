#!/usr/bin/env python3

import yaml
from yaml.loader import SafeLoader
import subprocess

# Open the file and load the file
with open('Configurazioni.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

for iter in data['iterations']:
    for l in data['lam']:
        subprocess.run(["./PINN.py", "--lam", str(l), "--iterations", str(iter)])

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
