## toMesh.sh

N.B. Alla riga 18 bisogna sostituire al posto di /root/TesinaMOR il vostro path assoluto della directory (per 4 volte).
So che è fastidioso, controllate prima di usarlo.

Se dà errore BLAS provare con questo `sudo apt-get install libopenblas-dev`.

Per mandare da terminale andare nella cartella TesinaMOR attraverso il comando `cd` e scrivere `./toMesh.sh` o `.\toMesh.sh` a seconda del sistema operativo.

Se dà errore "Permission denied" provare `chmod u+x ./toMesh.sh` (può cambiare i permessi solo chi crea il file di solito, ma tentar non nuoce, `comando chmod g+x nomefile`).