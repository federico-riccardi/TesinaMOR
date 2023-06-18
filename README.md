## Configurazioni.yaml

File dove sono inseriti i parametri di iterazioni, coefficienti e numero di punti per i PINN. Viene letto da altri file.

## PINN_funct.py

Prende in input  `iterations` numero di iterazioni, `coeff` coefficienti e `n_points` numero di punti da Configurazioni.yaml e il parametro per la funzione cutoff e restituisce la tabella con gli MSE per ogni epoch e la rete.

## Greedy_funct.py

Prende in input `problemData`, `lib`, il numero di punti nel training set, la tolleranza e il numero massimo di basi e restituisce la matrice con le basi, matrici e vettore forzante del problema ridotto.

## FEM_funct.py

Prende in input `problemData` e `lib` e restituisce matrici e termine noto del sistema FEM.

## PINNvsFEM.py

Calcola soluzione FEM e PINN con parametri presi dal file `Configurazioni.yaml` e calcola l'errore in norma $L^2$ e $L^{\infty}$ e seminorma $H^1$ a parità di iterazioni e coefficienti al variare di $[\mu_1, \, \mu_2]$. Usato per scegliere la quaterna di coefficienti migliore.
Salva i confronti in `results_plot1` -> `iterations` -> `coeff` -> `error.csv`.

## PINNvsFEM_it.py

Calcola soluzione FEM e PINN con parametri presi dal file `Configurazioni.yaml` e calcola l'errore in norma $L^2$ e $L^{\infty}$ e seminorma $H^1$ a parità di coefficienti e $[\mu_1, \, \mu_2]$ al variare delle iterazioni. Usato per scegliere il numero di iterazioni migliore.
Salva i confronti in `results_plot2` -> `coeff` -> $[\mu_1, \, \mu_2]$ -> `error.csv`.

## GREEDYvsFEM.py

Calcola la soluzione greedy e FEM e calcola l'errore in norma $L^2$ e $L^{\infty}$ e seminorma $H^1$ al variare di `M` numero di punti nel training set. Usato per scegliere il numero di punti nel training set migliore.
Salva i confronti in `result_plot_Greedy` -> `M` -> `error.csv`.

## plot_Greedy.py 

Legge i risultati nella cartella `result_plot_Greedy` e crea tre grafici Dash con le coppie testate di $[\mu_1, \, \mu_2]$ sull'asse delle ascisse e gli errori in norma $L^2$ e $L^{\infty}$ e seminorma $H^1$ sull'asse delle ordinate.

## plot1.py 

Legge i risultati nella cartella `result_plot1` e crea tre grafici Dash con le coppie testate di $[\mu_1, \, \mu_2]$ sull'asse delle ascisse e gli errori in norma $L^2$ e $L^{\infty}$ e seminorma $H^1$ sull'asse delle ordinate.

## plot2.py 

Legge i risultati nella cartella `result_plot2` e crea tre grafici Dash con le iterazioni sull'asse delle ascisse e gli errori in norma $L^2$ e $L^{\infty}$ e seminorma $H^1$ sull'asse delle ordinate.

## run_all_err.py

Calcola la soluzione PINN, FEM e Greedy e fa test online su un training set scelto random per calcolare media e varianza dell'errore commesso da PINN e Greedy in norma $L^2$ e $L^{\infty}$ e seminorma $H^1$.

## runall_funct.py

Calcola la soluzione PINN, FEM e Greedy e calcola i tempi impiegati nelle fasi offline e online. Per Greedy e PINN stampa a schermo i tempi offline e online, lo speedup della fase online e quello complessivo fase online + fase offline. 
Salva i grafici delle soluzioni qui calcolate.
