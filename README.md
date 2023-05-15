## toMesh.sh

N.B. Alla riga 18 bisogna sostituire al posto di /root/TesinaMOR il vostro path assoluto della directory (per 4 volte).
So che è fastidioso, controllate prima di usarlo.

Se dà errore BLAS provare con questo `sudo apt-get install libopenblas-dev`.

Per mandare da terminale andare nella cartella TesinaMOR attraverso il comando `cd` e scrivere `./toMesh.sh` o `.\toMesh.sh` a seconda del sistema operativo.

Se dà errore "Permission denied" dirlo a Debora (può cambiare i permessi solo chi crea il file, `comando chmod g+x nomefile`)

## PINN.py

Per mandare da terminale andare nella cartella TesinaMOR attraverso il comando `cd` e scrivere `./PINN.py` o `.\PINN.py` a seconda del sistema operativo.

Potrebbe servire installare `scipy`, `matplotlib` o altro per far funzionare `gedim`.

Mandare DOPO aver mandato `toMesh.sh` o comunque quando si ha `CppToPython`.

ToInsert: controllo sull'esistenza di `CppToPython`.