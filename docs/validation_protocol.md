# Protocollo obbligatorio di validazione dei backtest

Ogni risultato quantitativo di questo progetto è **PROVVISORIO** finché non supera tutti i controlli seguenti. Un controllo fallito deve interrompere il calcolo: non è consentito pubblicare o commentare metriche parziali come se fossero valide.

## 1. Integrità temporale

- timestamp non nulli e convertibili;
- ordinamento stabile esplicito per timestamp prima di qualsiasi calcolo;
- verifica programmatica `is_monotonic_increasing` dopo l’ordinamento;
- conteggio di righe riordinate e timestamp duplicati nel rapporto di audit;
- timezone e conversione dell’orario di sessione dichiarate esplicitamente;
- primo e ultimo timestamp della sessione verificati.

## 2. Sequenza degli eventi

- entrata, conferma, invalidazione, stop e target confrontati tramite timestamp;
- target valido solo se temporalmente precedente allo stop;
- trailing aggiornato esclusivamente con osservazioni già avvenute;
- uscita di fine sessione ricavata dall’ultimo timestamp cronologico, non dall’ultima riga fisica;
- test sintetici obbligatori per i casi target-prima-stop, stop-prima-target, stesso timestamp ed EOD.

## 3. Causalità

- ogni feature deve utilizzare solamente dati disponibili prima dell’entrata;
- indicatori rolling devono essere spostati quando la barra corrente non era ancora completata;
- train, validazione e test devono essere separati temporalmente;
- il campione OOS non può essere utilizzato per scegliere parametri o filtri.

## 4. Controlli di coerenza

- numero segnali, esclusioni, conferme e trade riconciliato riga per riga;
- almeno dieci giornate controllate manualmente contro i tick ordinati;
- confronto con un’implementazione indipendente o con casi sintetici noti;
- costi, slippage, fill e regole sui gap dichiarati;
- risultati separati per anno e sottoperiodo.

## 5. Robustezza statistica

- benchmark casuale/placebo con medesimi giorni, orari e direzione;
- sensibilità a costi e slippage;
- correzione per il numero di parametri provati;
- intervalli di confidenza mediante bootstrap temporale;
- configurazione congelata prima del test OOS.

## 6. Rapporto minimo prima della consegna

Ogni consegna deve includere:

- esito dei controlli di integrità;
- periodo e numero di sessioni disponibili;
- numero di segnali e trade;
- regole esatte e costi;
- distinzione tra in-sample, validazione e OOS;
- eventuali limitazioni ancora aperte.

La frase “verificato tick-by-tick” può essere usata solamente quando questo protocollo è stato completato e il relativo rapporto è stato salvato.
