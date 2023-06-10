## toMesh.sh

N.B. Alla riga 18 bisogna sostituire al posto di /root/TesinaMOR il vostro path assoluto della directory (per 4 volte).
So che è fastidioso, controllate prima di usarlo.

Se dà errore BLAS provare con questo `sudo apt-get install libopenblas-dev`.

Per mandare da terminale andare nella cartella TesinaMOR attraverso il comando `cd` e scrivere `./toMesh.sh` o `.\toMesh.sh` a seconda del sistema operativo.

Se dà errore "Permission denied" provare `chmod u+x ./toMesh.sh` (può cambiare i permessi solo chi crea il file di solito, ma tentar non nuoce, `comando chmod g+x nomefile`).

## PINN.py

Per mandare da terminale andare nella cartella TesinaMOR attraverso il comando `cd` e scrivere `./PINN.py` o `.\PINN.py` a seconda del sistema operativo.

Potrebbe servire installare `scipy`, `matplotlib` o altro per far funzionare `gedim`.

Mandare DOPO aver mandato `toMesh.sh` o comunque quando si ha `CppToPython`.

### coeff

Nella loss, parametri da associare al contributo dei 4 lati - suppongo il contributo della PDE unitario.

### Funzione cut-off dato di Neumann non omogeneo

La funzione cut-off s(x) è costruita a partire dal polinomio p(x) che soddisfa le seguenti condizioni:
p(0) = 0, p'(0) = 0, p(1) = 1, p'(1) = 0.
Risolvendo si ottiene p(x) = x^2(-2x + 3). s è poi ottenuta dilatando p, ovvero s(x) = p(x/delta), dove delta è il punto in cui s diventa 1.

## plots.py

File Dash che serve per creare i grafici interattivi. Prende i risultati dalla cartella `results` (che non viene pushata perché è nel file `.gitignore`), creata dal file `PINN.py`.

La cartella `results` ha la seguente struttura `results` -> `iterations` -> `coeff` -> `foto.jpg` e `loss.csv`.

Il file va mandato quando esiste la cartella `results`, compare la scritta `Dash is running on http://127.0.0.1:8050/`, copiare il link sul browser e lì appare il grafico.

## PINN_funct.py

Come PINN.py ma è una funzione: prende in input `iterations` numero di iterazioni, `coeff` discusso prima, `n_points` numero di punti e `delta` parametro della funzione cutoff.

## PINNvsFEM

Calcola soluzione FEM e PINN con parametri presi dal file `Configurazioni.yaml` e calcola l'errore in norma $L_2$ e $L_{inf}$ a parità di iterazioni e coefficienti al variare di $[\mu_1, \, \mu_2]$. Usato per scegliere la quaterna di coefficienti migliore.
Salva i confronti in `results_plot1` -> `iterations` -> `coeff` -> `error.csv`.

## PINNvsFEM

Calcola soluzione FEM e PINN con parametri presi dal file `Configurazioni.yaml` e calcola l'errore in norma $L_2$ e $L_{inf}$ a parità di coefficienti e $[\mu_1, \, \mu_2]$ al variare delle iterazioni. Usato per scegliere il numero di iterazioni migliore.
Salva i confronti in `results_plot2` -> `coeff` -> $[\mu_1, \, \mu_2]$ -> `error.csv`.

CAMBIARE NOME CARTELLA PINNvsFEM E PINNvsFEM_IT