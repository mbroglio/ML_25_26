# Analisi Esplorativa

## Descrizione del dataset

Il dataset utilizzato è il **Diabetes Binary Health Indicators (BRFSS 2015)**, composto da osservazioni binarie/ordinali che descrivono indicatori di salute individuali. La variabile target è `Diabetes_binary`, che indica la presenza (1) o assenza (0) di diabete.

- Le feature sono state caricate tramite `ucimlrepo` (ID=891) e, in caso di problemi di rete, da un file CSV locale.
- Per la fase di modellazione sono state considerate esclusivamente le feature numeriche (`int64` e `float64`), rimuovendo l’eventuale colonna identificativa `ID` e la colonna target `Diabetes_binary`.
- Il numero totale di feature numeriche considerate è pari a **21** (salvato nella variabile `FEATURES_NUMBER`).

Questa scelta di includere tutte le feature numeriche è motivata dal fatto che il dominio applicativo (indicatori di salute) suggerisce che molte variabili possano portare informazione rilevante e che non ci fossero, a priori, motivi forti per escludere alcune feature senza un’analisi quantitativa preliminare.

## Distribuzione delle classi

È stata calcolata la **distribuzione percentuale della variabile target** `Diabetes_binary`.

- Dal conteggio normalizzato emerge un **forte sbilanciamento** tra classe negativa (assenza di diabete) e classe positiva (presenza di diabete), con la classe 0 nettamente maggioritaria.
- Questo aspetto è cruciale perché i modelli di classificazione standard tendono a privilegiare la classe più frequente, massimizzando l’accuratezza complessiva ma penalizzando il **recall della classe positiva**, che in ambito clinico è più importante (perdere un paziente malato è più grave che etichettare come malato un paziente sano).

Per questo motivo, tutte le successive scelte modellistiche (undersampling, class weights, Focal Loss, utilizzo di metriche come ROC-AUC) sono state pensate in funzione della **gestione dello sbilanciamento**.

## Preprocessing: Standardizzazione

Le feature numeriche sono state standardizzate mediante `StandardScaler`:
- Per ciascuna feature viene sottratta la media e divisa per la deviazione standard, producendo `scaled_data`.

**Motivazioni della scelta:**
- Molti modelli usati (in particolare **reti neurali**, **SVM** e **PCA**) sono sensibili alla scala delle feature.
- La standardizzazione rende le feature confrontabili e impedisce che variabili con varianza maggiore dominino la funzione obiettivo.
- Per i modelli lineari a margine (SVM) e per i metodi basati su distanza, una scala omogenea è essenziale per ottenere frontiere di decisione significative.

## Train/Test Split

Dopo la standardizzazione, è stato eseguito uno split train/test con `train_test_split`:
- `test_size = 0.3` (30% dei dati per il test)
- `random_state = 42` per garantire **riproducibilità**

**Motivazioni dei parametri:**
- Una quota di test pari al 30% è un compromesso diffuso che consente di avere un numero sufficiente di campioni sia per l’addestramento sia per la valutazione.
- La fissazione del seme (`random_state=42`) consente di rendere ripetibile l’esperimento, requisito fondamentale in un progetto di Machine Learning documentato.

## PCA: Analisi delle Componenti Principali

È stata applicata una **PCA sui dati standardizzati** (`scaled_data`) per analizzare la struttura di correlazione tra le feature e valutare se fosse possibile una riduzione dimensionale efficace.

1. È stato inizialmente addestrato un oggetto `PCA()` senza specificare `n_components` per ottenere:
   - `explained_variance_ratio_` per ogni componente
   - la curva della varianza spiegata per componente
2. È stato calcolato il numero minimo di componenti necessario a spiegare almeno l’**80% della varianza totale**:
   - `n_components = (cumulative_variance < 0.8).sum() + 1`
   - Il risultato richiede **circa 14 componenti su 21** per raggiungere l’80% di varianza spiegata.

**Motivazioni della soglia di varianza (80%):**
- Una soglia intorno all’80–90% è uno standard de facto nei problemi di riduzione dimensionale, perché consente spesso di ridurre in modo significativo la dimensionalità mantenendo la maggior parte dell’informazione.
- Nel nostro caso, il fatto che servano **14 componenti su 21** per raggiungere l’80% indica che l’informazione è **distribuita** su molte feature e che la PCA **non comporta una reale riduzione drastica della dimensionalità**.

### Heatmap dei loadings PCA e Matrice di Covarianza

È stata disegnata una heatmap dei **loadings** (componenti della matrice di trasformazione PCA) e una **matrice di covarianza** delle feature standardizzate.

- **Heatmap PCA**: mostra i pesi con cui ogni feature contribuisce a ciascuna componente principale.
  - Non si osserva una **feature dominante** su tutte le componenti.
  - Ogni componente raccoglie contributi da più feature.
- **Matrice di covarianza**: le covarianze tra pair di feature risultano in gran parte **piccole**, indicando **debole correlazione** tra le variabili.

**Conclusione sull’uso della PCA:**
- La bassa correlazione tra le feature suggerisce che **quasi tutte apportano informazione** e non esistono gruppi di variabili fortemente ridondanti.
- La necessità di 14/21 componenti per spiegare l’80% della varianza indica che la PCA **non riduce in modo sostanziale la dimensionalità**, pur introducendo un certo grado di perdita di interpretabilità (si passa da feature cliniche leggibili a combinazioni lineari).
- Per questo motivo, nel resto del lavoro la PCA viene utilizzata **solo come termine di confronto** (pipeline con PCA vs senza PCA), ma non è adottata come trasformazione principale.

---

# Reti Neurali

L’obiettivo delle reti neurali è modellare la relazione non lineare tra le feature di salute e la variabile target binaria (`Diabetes_binary`), con particolare attenzione alla gestione dello **sbilanciamento** e al **compromesso tra accuratezza e recall sui positivi**.

## Architettura di base (No PCA, dataset originale)

### Struttura del modello

Il primo modello di rete neurale è definito come:
- **Input:** vettori standardizzati di dimensione pari a `FEATURES_NUMBER = 21`.
- **Layer nascosti:**
  - Densa(128, attivazione `ReLU`) + Dropout(0.3)
  - Densa(64, attivazione `ReLU`) + Dropout(0.3)
  - Densa(32, attivazione `ReLU`) + Dropout(0.3)
- **Output:**
  - Densa(1, attivazione `sigmoid`) per produrre una **probabilità** di appartenenza alla classe positiva.
- **Loss:** `binary_crossentropy`
- **Ottimizzatore:** `adam`
- **Epoche:** 150
- **Batch size:** 256
- **Validation split:** 0.1 (10% del training set usato come validation interna)

### Motivazioni delle scelte architetturali

- **Profondità e numero di neuroni (128 → 64 → 32):**
  - La struttura decrescente è un pattern comune che permette alla rete di **comprimere progressivamente l’informazione**.
  - 128 neuroni al primo livello consentono di modellare interazioni complesse fra 21 feature senza esagerare in complessità.
  - La riduzione a 64 e poi 32 neuroni limita il rischio di overfitting e riduce il numero di parametri.
- **ReLU come attivazione nei layer nascosti:**
  - Standard de facto per reti profonde grazie a:
    - semplicità computazionale,
    - mitigazione (parziale) del problema del vanishing gradient,
    - capacità di modellare relazioni non lineari.
- **Dropout (0.3) dopo ogni layer nascosto:**
  - Lo scopo è ridurre l’overfitting, particolarmente rilevante su un dataset con forte sbilanciamento: senza regolarizzazione la rete rischia di imparare pattern troppo specifici della classe dominante.
  - Un tasso di dropout del 30% è un compromesso ampiamente utilizzato: abbastanza alto da regolarizzare, non così alto da impedire l’apprendimento.
- **Output sigmoide e Binary Cross-Entropy:**
  - Con un problema di classificazione binaria, la scelta `sigmoid` + `binary_crossentropy` è naturale: la loss misura correttamente la distanza tra probabilità predetta e label binaria.
  - La presenza di feature binarie/ordinali (dopo standardizzazione) non pone problemi a questa combinazione.
- **Adam come ottimizzatore:**
  - Adam combina i vantaggi di AdaGrad e RMSProp, adattando il learning rate parametro per parametro.
  - È adatto a dataset mediamente grandi e a funzioni di loss non stazionarie; riduce la necessità di un tuning manuale fine del learning rate.
- **Numero di epoche (150) e batch size (256):**
  - 150 epoche permettono al modello di convergere (come si osserva dalle curve di loss e accuracy senza divergenza marcata della validation loss).
  - Un batch size di 256 è un buon compromesso tra:
    - efficienza computazionale (batch grandi sfruttano meglio il parallelismo),
    - rumore del gradiente (batch troppo piccoli introducono rumore eccessivo),
    - memoria disponibile.

### Risultati e interpretazione

Dalla matrice di confusione, dal classification report e dalla curva ROC emergono i seguenti punti (in forma qualitativa, in linea con i commenti nel notebook):

- Il modello è **fortemente influenzato dalla classe negativa**:
  - Alto numero di veri negativi, ma ancora troppe istanze positive etichettate come negative (falsi negativi).
- La **ROC-AUC è > 0.8**, segno che il modello ha una **buona capacità di separare le classi** in termini di probabilità predetta.
- Le curve di training/validation loss non mostrano divergenze marcate; esiste un inizio di overfitting ma **contenuto** grazie al dropout.

**Conclusione:**
- Come baseline, il modello è **stabile** e discriminante, ma **non soddisfa i requisiti clinici**: il numero di falsi negativi è ancora troppo alto. Da qui nasce la necessità di tecniche specifiche per gestire lo sbilanciamento.

## Gestione dello sbilanciamento nelle reti neurali

Sono stati sperimentati tre approcci principali:
1. **Undersampling + Class Balancing**
2. **Class Weights** sul dataset originale
3. **Focal Loss** sul dataset originale (con e senza PCA)

### 1. Undersampling del dataset

#### Costruzione del dataset bilanciato

- Si individuano gli indici di classe minoritaria (`y == 1`) e maggioritaria (`y == 0`).
- Si effettua un **undersampling casuale** della classe maggioritaria selezionando un numero di campioni uguale a quello della classe minoritaria.
- Si concatenano i due insiemi e si mescolano, ottenendo un dataset bilanciato (`X_bal`, `y_bal`).
- Lo split train/test (70/30) viene rifatto sul dataset bilanciato con `stratify=y_bal` per mantenere il bilanciamento in entrambe le partizioni.

**Motivazioni:**
- L’undersampling elimina lo sbilanciamento a livello di training, permettendo al modello di **trattare le classi alla pari**.
- Riduce il numero di campioni, velocizzando l’addestramento.
- Lo svantaggio principale è la **perdita di informazione**: molti esempi della classe maggioritaria non vengono mai visti.

#### Architettura della rete con undersampling

Su questo dataset più piccolo e bilanciato si adotta una rete con **capacità ridotta**:
- Densa(64) → Densa(32) → Densa(16) con ReLU e dropout (0.5 sul primo layer, 0.3 sugli altri).
- Loss: `binary_crossentropy`, ottimizzatore `adam`.
- 100 epoche, batch size 128.

**Giustificazione dei parametri:**
- Riduzione del numero di neuroni rispetto al modello base: con meno dati di training è più facile overfittare; un modello meno complesso è quindi più appropriato.
- Dropout più alto (0.5) nel primo layer per contrastare ulteriormente l’overfitting dovuto al minor numero di esempi.
- Meno epoche (100) perché il dataset è più piccolo e la convergenza viene raggiunta più rapidamente.
- Batch size 128 (più piccolo rispetto a 256) per avere gradienti leggermente più rumorosi e una regolarizzazione implicita, senza impattare troppo sui tempi di training.

#### Risultati qualitativi

- Le **precision e recall medie aumentano** in modo significativo rispetto al modello base.
- Il modello diventa più propenso a classificare come **positive** le istanze dubbie:
  - La percentuale di **falsi negativi** diminuisce,
  - Aumentano i **falsi positivi**.
- La curva ROC resta pressoché invariata, segno che la capacità discriminante globale non cambia ma il **punto operativo** sulla curva varia.
- Le curve di train/validation loss e accuracy risultano più parallele e stabili, evidenziando **minor overfitting**.

### 2. Class Weights sul dataset originale

In questo approccio si **mantiene l’intero dataset** ma si introducono pesi sulle classi nella funzione di loss.

- Si calcolano i pesi con `class_weight.compute_class_weight(class_weight="balanced", ...)` sui soli dati di training.
- Si costruisce un modello con **stessa architettura della baseline** (128-64-32 neuroni, dropout 0.3), ma passando `class_weight=class_weights` a `model.fit`.

**Motivazioni della scelta:**
- A differenza dell’undersampling, i **dati non vengono buttati**: tutte le osservazioni contribuiscono all’addestramento.
- Il modello viene esplicitamente penalizzato quando sbaglia la classe minoritaria, senza alterare la distribuzione dei dati.
- Il calcolo dei pesi bilanciati è automatico e dipende solo dalla distribuzione osservata nel training set.

#### Risultati qualitativi

- Il **recall della classe positiva** è intorno a ~0.78, comparabile a quello ottenuto con undersampling.
- La **precision** sulla classe positiva è più bassa (~0.31), indicando un numero maggiore di falsi positivi.
- Le curve di validation sono più rumorose rispetto al caso undersampling e mostrano una **leggera divergenza** dal train, segno che il modello fatica un po’ di più a stabilizzarsi ma beneficia comunque dei pesi.
- La matrice di confusione risulta molto simile a quella del caso `undersampling + class balancing` in termini di riduzione dei falsi negativi.

**Conclusione per i Class Weights:**
- È un compromesso efficace: si **riduce la perdita di informazione** (a differenza dell’undersampling) mantenendo un buon recall sui positivi, al costo di più falsi positivi.
- È particolarmente interessante in contesti dove è preferibile **non perdere pazienti malati**, anche a costo di aumentare gli allarmi falsi.

### 3. Focal Loss

La **Focal Loss** è stata sperimentata sia senza PCA sia con PCA.

#### Definizione e parametrizzazione

- La loss è definita come:
  $$
  \text{FL}(p_t) = -\alpha (1-p_t)^\gamma \log(p_t)
  $$
  dove $p_t$ è la probabilità della classe vera, $\gamma$ è il parametro di "focalizzazione" e $\alpha$ bilancia le classi.
- Nel codice:
  - `gamma = 2.0`: aumenta il peso degli esempi difficili (dove $p_t$ è lontano da 1).
  - `alpha` viene calcolato automaticamente dalla distribuzione delle classi (approx. frequenza della classe negativa) per dare **maggior peso alla classe positiva**.

**Motivazioni della scelta dei parametri:**
- `gamma = 2.0` è il valore più comunemente usato in letteratura (RetinaNet) e rappresenta un buon compromesso tra focalizzazione e stabilità numerica.
- Calcolare `alpha` in base alla distribuzione reale delle classi rende l’algoritmo adattivo allo sbilanciamento osservato.

#### Risultati e considerazioni

- Senza PCA, la rete con Focal Loss usa la stessa architettura della baseline (128-64-32 + dropout 0.3) e produce un leggero miglioramento di bilanciamento tra precision e recall, ma **non un salto qualitativo** rispetto a Class Weights.
- Con PCA, la Focal Loss non porta miglioramenti significativi rispetto alla combinazione `PCA + Class Weights`.
- Il costo computazionale è **maggiore** rispetto alla simple binary cross-entropy con class weights.

**Conclusione sulla Focal Loss:**
- Dal punto di vista pratico, i benefici non giustificano l’aumento di complessità e di costo computazionale.
- Nel report finale, questa variante può essere menzionata come esperimento, ma **non come soluzione preferita**.

## Reti neurali con PCA

Per completezza, sono stati costruiti modelli MLP anche sul dataset trasformato dalla PCA (con `n_components` scelto per spiegare ~80% della varianza) e bilanciato tramite class weights.

- Architettura più compatta (32-16-8 neuroni) per riflettere il minor numero di feature.
- Stessa impostazione di dropout, epoche (150), batch size (256) e Adam come ottimizzatore.
- Vengono calcolati class weights sulla distribuzione di `y_train_pca`.

**Osservazioni:**
- `PCA + Class Weights` tende a **massimizzare la sensibilità sulla classe minoritaria** a scapito dell’accuratezza globale e di un aumento dei falsi positivi.
- Questo modello è particolarmente adatto in scenari in cui l’obiettivo principale è **minimizzare i falsi negativi**, accettando un costo maggiore in termini di falsi allarmi.

---

# Alberi di Decisione

In questa sezione vengono analizzati modelli di **Alberi di Decisione** con diverse configurazioni:
- Bilanciamento delle classi tramite `class_weight` vs undersampling
- Diversi livelli di pruning (`ccp_alpha`)
- Con e senza PCA

## Alberi di Decisione senza PCA

### Modello con class_weight e pruning moderato

È stato addestrato un `DecisionTreeClassifier` con i seguenti parametri:
- `random_state = 42` per la riproducibilità
- `class_weight = 'balanced'` per gestire lo sbilanciamento delle classi
- `ccp_alpha = 0.001` per applicare un pruning moderato

**Motivazioni:**
- Il parametro `class_weight='balanced'` è fondamentale per affrontare lo sbilanciamento del dataset: assegna automaticamente pesi inversamente proporzionali alle frequenze delle classi, penalizzando maggiormente gli errori sulla classe minoritaria (diabete presente).
- `ccp_alpha = 0.001` introduce un **cost-complexity pruning** moderato che riduce l'overfitting, eliminando i rami con guadagni di impurità molto piccoli.
- Questo valore rappresenta un compromesso: rimuove split poco informativi senza semplificare eccessivamente l'albero.

### Modello con dataset bilanciato (undersampling)

Per confronto, è stato addestrato un `DecisionTreeClassifier` sul dataset bilanciato tramite undersampling:
- `random_state = 42`
- Nessun `class_weight` (non necessario su dataset già bilanciato)
- Nessun `ccp_alpha` esplicito (parametri di default)

**Motivazioni:**
- L'undersampling elimina lo sbilanciamento a monte, permettendo all'albero di apprendere con uguale importanza da entrambe le classi.
- L'assenza di parametri aggiuntivi permette di valutare se il dataset bilanciato è sufficiente a ottenere buone prestazioni.
- Questo approccio serve come termine di confronto con il modello che usa `class_weight` sul dataset completo.

I risultati vengono sintetizzati da:
- Accuracy sul test set
- Profondità dell'albero (`get_depth()`)
- Numero di foglie (`get_n_leaves()`)
- Matrice di confusione e classification report

### Albero con pruning conservativo

Per ulteriormente esplorare l'effetto del pruning, è stato addestrato un `DecisionTreeClassifier` con:
- `random_state = 42`
- `class_weight = 'balanced'`
- `ccp_alpha = 0.0001` (pruning più conservativo, 10 volte più piccolo di 0.001)

**Motivazioni di `ccp_alpha = 0.0001`:**
- Questo valore più piccolo permette un albero potenzialmente più profondo e complesso rispetto al caso con `ccp_alpha = 0.001`.
- Elimina solo i rami con i guadagni di impurità davvero minimi dovuti al rumore.
- Il confronto tra i due valori di `ccp_alpha` (0.001 vs 0.0001) permette di valutare il trade-off tra complessità del modello e capacità di generalizzazione.
- `class_weight='balanced'` rimane attivo per gestire lo sbilanciamento delle classi.

**Osservazioni generali:**
- L'uso di `class_weight='balanced'` è cruciale per gestire lo sbilanciamento senza perdere informazione (come invece accade con l'undersampling).
- Il pruning controllato tramite `ccp_alpha` riduce l'overfitting mantenendo una buona interpretabilità.
- Gli alberi mantengono il vantaggio di essere facilmente visualizzabili e interpretabili, mostrando le regole decisionali esplicite.

## Alberi di Decisione con PCA

Per valutare l’effetto della riduzione dimensionale sulla famiglia degli alberi, sono stati addestrati modelli sugli **score PCA** (`X_train_pca`, `X_test_pca`).

### Albero con pruning moderato su PCA

- `DecisionTreeClassifier(random_state=42, class_weight='balanced', ccp_alpha=0.001)` addestrato su `X_train_pca`.

**Motivazione:**
- Applicare lo stesso approccio del modello senza PCA (class_weight + pruning moderato) alle componenti principali.
- Valutare se la riduzione dimensionale combinata con il bilanciamento delle classi migliora le prestazioni.

### Albero con pruning conservativo su PCA

- `DecisionTreeClassifier(random_state=42, class_weight='balanced', ccp_alpha=0.0001)` addestrato su `X_train_pca`.

**Motivazione:**
- Stessa logica di pruning conservativo del caso senza PCA, ma sulle componenti PCA.
- Confrontare l'effetto della PCA su alberi con diversi livelli di complessità.

**Osservazioni qualitative generali:**
- L’uso della PCA non porta a **miglioramenti sostanziali** nelle prestazioni degli alberi rispetto ai corrispettivi senza PCA, coerentemente con i risultati globali sull’analisi PCA.
- La perdita di interpretabilità (componenti PCA al posto di feature cliniche) rende gli alberi meno leggibili, riducendo uno dei principali vantaggi dei modelli ad albero.

## Support Vector Machines (SVM)

Le SVM sono state utilizzate come ulteriore riferimento, con diversi kernel.

### Kernel lineare

- Modello: `SVC(kernel='linear', C=1.0, random_state=42)`
- Addestrato su `X_train`, testato su `X_test`.

**Motivazioni:**
- Il kernel lineare è interpretabile (pesi delle feature) e spesso efficace su dati ad alta dimensione.
- `C = 1.0` è il valore standard che bilancia margine ampio e penalità degli errori; rappresenta un buon punto di partenza.

### Kernel RBF

- Modello: `SVC(kernel='rbf', C=1.0, random_state=42)`
- Per motivi computazionali, addestrato su un sottoinsieme: `X_train_small = X_train[:10000]`, `y_train_small` corrispondente.

**Motivazioni dei parametri:**
- Il kernel RBF permette di modellare **frontiere altamente non lineari**, adatte a dati complessi.
- `C = 1.0` ancora una volta come default ragionevole.
- L’uso di un sottoinsieme di 10.000 campioni è necessario perché:
  - La complessità di SVC con kernel non lineare cresce più che linearmente con il numero di campioni.
  - Addestrare su tutto il dataset sarebbe eccessivamente lento.

### Kernel polinomiale

- Modello: `SVC(kernel='poly', degree=3, C=1.0, random_state=42, class_weight='balanced')`
- Addestrato su `X_train_small`.

**Motivazioni dei parametri:**
- `degree = 3` è una scelta classica per il kernel polinomiale, in grado di modellare interazioni di grado moderato senza esplodere in complessità.
- `class_weight = 'balanced'` affronta lo sbilanciamento in modo analogo a Random Forest e reti neurali con class weights.

**Sintesi sul ruolo delle SVM:**
- Offrono un confronto utile con modelli ad albero e reti neurali.
- Confermano la necessità di gestire lo sbilanciamento (`class_weight='balanced'` o sottocampionamento) e di lavorare con feature standardizzate.

---

# Conclusioni

## Sintesi dei risultati principali

1. **Analisi Esplorativa e PCA**
   - Il dataset presenta un **forte sbilanciamento** tra classe negativa e positiva.
   - Le feature risultano **debolmente correlate** fra loro; non esistono gruppi chiari di variabili ridondanti.
   - La PCA richiede **14 componenti su 21** per spiegare ~80% della varianza, indicando che la riduzione dimensionale è limitata e comporta una perdita di interpretabilità.

2. **Reti Neurali**
   - Il modello MLP di base (21 feature standardizzate, 3 hidden layer 128-64-32, dropout 0.3, Adam, 150 epoche, batch 256) fornisce una buona **ROC-AUC (>0.8)** ma è ancora sbilanciato verso la classe negativa (troppi falsi negativi).
   - L’**undersampling** produce un modello più equilibrato, con recall più alto sulla classe positiva ma perdita di informazione sulla classe negativa.
   - L’uso di **Class Weights** sul dataset completo riesce a mantenere tutti i dati e a ottenere un **recall dei positivi elevato (~0.78)**, accettando una precision più bassa (~0.31) e qualche oscillazione nelle curve di validazione.
   - La **Focal Loss** offre un miglioramento marginale, ma a fronte di un **maggiore costo computazionale** e complessità.
   - L’uso della **PCA** nelle reti neurali non porta vantaggi netti: la combinazione `PCA + Class Weights` può essere utile solo se si accetta una forte priorità al recall dei positivi rispetto a precision e accuratezza globale.

3. **Alberi di Decisione e Random Forest**
   - Gli **alberi senza pruning** mostrano overfitting, con profondità elevate e struttura complessa.
   - Il **pruning (ccp_alpha=0.0001)** semplifica l’albero, migliorando la generalizzazione a costo di una leggera riduzione di accuratezza sul training.
   - La **Random Forest (n_estimators=100, class_weight='balanced')** fornisce un buon compromesso tra prestazioni, robustezza e capacità di gestione dello sbilanciamento.
   - L’uso della PCA con gli alberi non offre guadagni sostanziali e riduce l’interpretabilità.

4. **SVM**
   - Le SVM con kernel lineare, RBF e polinomiale confermano l’importanza della **standardizzazione** e della **gestione dello sbilanciamento** (sottoinsieme di training, class weights).
   - Per motivi computazionali, è stato necessario usare un sottoinsieme del training set per i kernel non lineari.

## Modello/i raccomandati in base allo scenario

In un contesto clinico, la scelta del modello dipende dal trade-off desiderato tra:
- **Minimizzazione dei falsi negativi** (non perdere pazienti malati),
- **Controllo dei falsi positivi** (evitare troppi allarmi inutili),
- **Interpretabilità** del modello.

### Scenario 1: massima attenzione ai positivi (minimizzare i falsi negativi)

- Modello consigliato: **MLP con Class Weights**, senza PCA.
  - Mantiene tutte le informazioni dalle 21 feature.
  - Class Weights spostano il focus sugli esempi positivi.
  - Recall elevato sui positivi (~0.78) rende il modello adatto a uno scenario "screening" in cui è meglio avere falsi allarmi che perdere casi reali.
- In alternativa, se si accetta un’ulteriore perdita di interpretabilità, può essere preso in considerazione **PCA + Class Weights**, che tende ancora di più a privilegiare la sensibilità.

### Scenario 2: limitare i falsi allarmi (maggiore precision)

- Modello consigliato: **MLP con Class Weights senza PCA**, calibrando la soglia decisionale.
  - Permette di modulare la soglia sulla probabilità predetta per aumentare la precision a scapito del recall.
  - In alternativa, un **Decision Tree potato** offre maggiore interpretabilità pur mantenendo buone performance.

### Scenario 3: massima interpretabilità del modello

- Modello consigliato: **Decision Tree potato** (con pruning).
  - L'albero potato è più leggibile: è possibile tracciare esplicitamente regole del tipo "se BMI > soglia e fumo = sì allora...".
  - Il bilanciamento delle classi tramite `class_weight` garantisce che il modello non ignori la classe minoritaria.

## Ruolo della PCA nel progetto

L’analisi condotta porta alle seguenti conclusioni sulla PCA:
- Nonostante sia uno strumento standard per la riduzione dimensionale, nel nostro problema **non porta vantaggi significativi** in termini di performance rispetto ai modelli che lavorano sulle 21 feature originali.
- La PCA riduce la dimensionalità solo moderatamente (da 21 a ~14 componenti per l’80% di varianza) e introduce una **perdita di interpretabilità** delle feature, importante in ambito clinico.
- Di conseguenza, la PCA è stata **mantenuta solo come confronto sperimentale**, ma non viene raccomandata come trasformazione principale per un sistema da mettere in produzione.

## Conclusione finale

Nel complesso, il lavoro mostra come, partendo da un dataset fortemente sbilanciato e con molte feature debolmente correlate, sia necessario:
- **Standardizzare** le feature,
- **Gestire esplicitamente lo sbilanciamento** (tramite undersampling, class weights, o Focal Loss),
- **Valutare attentamente il ruolo della PCA**, tenendo conto sia della varianza spiegata sia della perdita di interpretabilità,
- **Scegliere il modello** (rete neurale o albero di decisione) in base al trade-off desiderato tra recall, precisione e interpretabilità.

Tra i modelli testati, una **rete neurale MLP con Class Weights sul dataset standardizzato e completo (senza PCA)** emerge come soluzione particolarmente efficace quando la priorità è identificare il maggior numero possibile di pazienti a rischio di diabete, mentre un **Decision Tree bilanciato e potato** rappresenta un'ottima alternativa quando si desidera massima interpretabilità del modello.