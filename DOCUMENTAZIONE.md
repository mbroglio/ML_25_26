# Documentazione Progetto: Predizione del Diabete con Machine Learning

## Indice
1. [Introduzione](#introduzione)
2. [Dataset e Preprocessing](#dataset-e-preprocessing)
3. [Analisi Esplorativa](#analisi-esplorativa)
4. [PCA (Principal Component Analysis)](#pca-principal-component-analysis)
5. [Reti Neurali](#reti-neurali)
6. [Decision Tree](#decision-tree)
7. [Confronto e Conclusioni](#confronto-e-conclusioni)

---

## Introduzione

### Obiettivo del Progetto
Sviluppare modelli di Machine Learning per la predizione binaria del diabete utilizzando il dataset "Diabetes Binary Health Indicators BRFSS2015" dall'UCI Machine Learning Repository.

### Problema e Contesto
Il dataset presenta uno **sbilanciamento significativo** tra le classi (circa 86% negativi, 14% positivi), caratteristica comune nei problemi di screening medico. In ambito clinico, è fondamentale **minimizzare i falsi negativi** (persone diabetiche classificate come sane), anche a costo di aumentare i falsi positivi.

---

## Dataset e Preprocessing

### Caratteristiche del Dataset
- **Numero di feature**: 21 (tutte numeriche/binarie)
- **Target**: `Diabetes_binary` (0 = no diabete, 1 = diabete)
- **Dimensione**: ~70,000 osservazioni
- **Sbilanciamento**: ~86% classe negativa, ~14% classe positiva

### Standardizzazione
**Scelta**: Utilizzo di `StandardScaler` per normalizzare tutte le feature.

**Giustificazione**:
- Le feature hanno scale diverse (es. età vs. binarie)
- Le reti neurali e PCA beneficiano significativamente dalla standardizzazione
- Migliora la convergenza degli algoritmi di ottimizzazione (Adam)
- Essenziale per la corretta applicazione della PCA (che è sensibile alla scala)

```python
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features[numeric_features])
```

### Split Train-Test
**Parametri**:
- `test_size=0.3` (70% training, 30% test)
- `random_state=42` (riproducibilità)
- `stratify=y_bal` (solo per dataset bilanciato)

**Giustificazione**:
- **70/30**: Compromesso standard che mantiene sufficiente training set preservando una validazione robusta
- **Stratify**: Nel dataset bilanciato è essenziale mantenere la proporzione 50/50 anche nel test set

---

## Analisi Esplorativa

### Distribuzione delle Classi
Il dataset mostra un forte sbilanciamento verso la classe negativa (~86%), tipico dei problemi di screening medico dove la prevalenza della patologia è minore rispetto alla popolazione sana.

### Matrice di Covarianza
**Osservazione**: Le feature mostrano **debole correlazione** reciproca.

**Implicazione**: Ogni feature contiene informazioni potenzialmente uniche e rilevanti, giustificando l'inclusione di tutte nel modello iniziale.

---

## PCA (Principal Component Analysis)

### Obiettivo
Ridurre la dimensionalità del problema mantenendo l'80% della varianza totale.

### Parametri e Risultati
**Soglia di varianza**: 80%
**Componenti necessarie**: 14 su 21

**Giustificazione della soglia**:
- **80%**: Standard nella letteratura come compromesso tra riduzione dimensionale e preservazione informativa
- Valori più bassi (es. 60-70%) perderebbero troppa informazione
- Valori più alti (es. 90-95%) non ridurrebbero significativamente la complessità

### Analisi Heatmap PCA
**Osservazioni**:
1. Nessuna singola feature domina tutti i componenti principali
2. I contributi sono distribuiti tra multiple feature
3. Coerente con la bassa correlazione osservata nella matrice di covarianza

**Conclusione sulla PCA**: 
La necessità di utilizzare **14 componenti su 21** (67% delle feature originali) per spiegare l'80% della varianza indica che la PCA porta a una **riduzione dimensionale limitata**. Questo giustifica l'approccio comparativo PCA vs. No-PCA adottato nel progetto.

---

## Reti Neurali

### Architettura Base (No PCA - Dataset Originale)

#### Struttura
```
Input (21 features) → Dense(128, ReLU) → Dropout(0.3) 
                   → Dense(64, ReLU) → Dropout(0.3)
                   → Dense(32, ReLU) → Dropout(0.3)
                   → Dense(1, Sigmoid)
```

#### Giustificazione dei Parametri

**1. Numero di neuroni per layer (128 → 64 → 32)**
- **Architettura decrescente**: Progressiva estrazione di feature sempre più astratte
- **128 neuroni iniziali**: ~6x il numero di input features, consente di catturare relazioni complesse
- **Riduzione progressiva**: Evita colli di bottiglia improvvisi che potrebbero perdere informazione
- **32 neuroni finali**: Sufficiente rappresentazione prima della decisione binaria

**2. Funzione di attivazione ReLU**
- **Vantaggi**: Computazionalmente efficiente, evita vanishing gradient
- **Alternativa considerata**: LeakyReLU/ELU non necessarie per questo problema
- **Motivazione**: Performance eccellenti in classificazione binaria con feature normalizzate

**3. Dropout (0.3)**
- **Valore moderato**: Bilancia prevenzione overfitting e capacità di apprendimento
- **Posizionamento**: Dopo ogni layer nascosto per regolarizzazione distribuita
- **Giustificazione**: Con 70k esempi e dataset sbilanciato, il rischio di overfitting sulla classe maggioritaria è elevato
- **Alternativa scartata**: Dropout 0.5 risultava troppo aggressivo, riducendo troppo la capacità del modello

**4. Output Layer (Sigmoid)**
- **Motivazione**: Standard per classificazione binaria, output in [0,1] interpretabile come probabilità
- **Alternativa**: Softmax con 2 neuroni output sarebbe ridondante

**5. Loss Function: Binary Cross-Entropy**
- **Motivazione**: Loss standard ottimale per classificazione binaria
- **Proprietà matematica**: Penalizza fortemente predizioni confidenti ma errate
- **Compatibilità**: Perfetta con output sigmoid

**6. Optimizer: Adam**
- **Learning rate adattivo**: Si adatta automaticamente senza tuning manuale
- **Momentum**: Accelera convergenza in direzioni consistenti
- **Parametri default** (lr=0.001, β₁=0.9, β₂=0.999): Funzionano eccellentemente per la maggior parte dei problemi
- **Alternativa scartata**: SGD richiederebbe tuning manuale del learning rate

**7. Epoche (150) e Batch Size (256)**
- **150 epoche**: Osservando le curve di training, la convergenza avviene ~100-120 epoche; 150 garantisce ottimalità
- **Batch size 256**: 
  - Compromesso tra stabilità gradiente (batch grandi) e frequenza aggiornamenti (batch piccoli)
  - Con 70k esempi: ~273 batch per epoca, buon bilancio
  - Memoria GPU: 256 è gestibile per questa architettura
  - **Alternativa 128**: Più rumore ma convergenza più lenta
  - **Alternativa 512**: Più stabile ma potrebbe perdere capacità di generalizzazione

**8. Validation Split (0.1)**
- **10% per validazione**: 7k esempi sufficienti per monitoraggio affidabile
- **Preserva training data**: Mantiene 63k esempi per training con dataset già limitato (70k)

### Approccio Undersampling + Class Balancing

#### Architettura Modificata
```
Input (21 features) → Dense(64, ReLU) → Dropout(0.5) 
                   → Dense(32, ReLU) → Dropout(0.3)
                   → Dense(16, ReLU) → Dropout(0.3)
                   → Dense(1, Sigmoid)
```

#### Giustificazione delle Modifiche

**1. Numero neuroni dimezzato (64 → 32 → 16)**
- **Motivazione**: Dataset bilanciato contiene ~35k esempi (metà dell'originale)
- **Prevenzione overfitting**: Rete più piccola proporzionale al dataset ridotto
- **Capacità adeguata**: Sufficiente per 21 feature con relazioni più equilibrate

**2. Dropout aumentato (0.5 nel primo layer)**
- **Rischio overfitting elevato**: Dataset più piccolo richiede regolarizzazione più aggressiva
- **Primo layer**: Dropout 0.5 previene dipendenza eccessiva da singole feature
- **Layer successivi**: 0.3 mantiene bilanciamento come modello base

**3. Batch size ridotto (128)**
- **Dataset più piccolo**: 35k esempi → ~273 batch con size 128
- **Maggior rumore**: Benefico per generalizzazione su dataset ridotto
- **Aggiornamenti più frequenti**: Favorisce esplorazione dello spazio dei parametri

**4. Epoche ridotte (100)**
- **Convergenza più rapida**: Dataset bilanciato facilita l'apprendimento
- **Rischio overfitting**: Meno epoche prevengono adattamento eccessivo al training set ridotto

#### Tecnica di Undersampling

```python
def balance_dataset(X, y):
    idx_minority = np.where(y == 1)[0]
    idx_majority = np.where(y == 0)[0]
    n_minority = len(idx_minority)
    
    idx_majority_downsampled = np.random.choice(
        idx_majority, size=n_minority, replace=False
    )
```

**Giustificazione**:
- **Random undersampling**: Semplice ed efficace per questo problema
- **Shuffle**: Previene bias nell'ordine dei dati
- **Stratify nello split**: Mantiene 50/50 anche nel test set

### Approccio Class Weights

#### Parametri
```python
class_weights = {0: 0.58, 1: 3.94}
```

**Calcolo automatico**:
```python
class_weights_vals = class_weight.compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
```

#### Giustificazione

**1. Formula dei pesi**:
$$w_i = \frac{n_{samples}}{n_{classes} \times n_{samples\_class\_i}}$$

- **Classe 0 (maggioritaria)**: Peso ~0.58 → errori meno costosi
- **Classe 1 (minoritaria)**: Peso ~3.94 → errori ~7x più costosi

**2. Vantaggi rispetto a undersampling**:
- **Nessuna perdita di dati**: Utilizza tutti i 70k esempi
- **Più informazione**: Il modello vede tutti i pattern della classe maggioritaria
- **Computazionalmente efficiente**: No preprocessing, pesi applicati nella loss

**3. Architettura identica al modello base**:
- **Mantiene complessità**: Dataset completo richiede stessa capacità
- **Stessi iperparametri**: Dropout 0.3, epoche 150, batch 256

**4. Differenze nel training**:
- **Loss pesata**: $Loss = w_i \times BCE(y_i, \hat{y}_i)$
- **Gradiente modificato**: Errori su classe 1 generano gradiente ~4x maggiore
- **Convergenza**: Leggermente più rumorosa ma efficace

### Approccio Focal Loss

#### Definizione
```python
def focal_loss(gamma=2.0, alpha=0.25):
    FL(p_t) = -α_t(1 - p_t)^γ log(p_t)
```

Dove:
- $p_t$ = probabilità predetta della classe corretta
- $\gamma$ = focusing parameter (default 2.0)
- $\alpha$ = peso classe positiva (calcolato automaticamente)

#### Giustificazione dei Parametri

**1. Gamma (γ = 2.0)**
- **Focusing effect**: $(1 - p_t)^2$ riduce drasticamente il peso degli esempi facili
- **Esempio**:
  - Se $p_t = 0.9$ (esempio facile): peso = $(1-0.9)^2 = 0.01$ (quasi ignorato)
  - Se $p_t = 0.5$ (esempio difficile): peso = $(1-0.5)^2 = 0.25$ (peso normale)
  - Se $p_t = 0.1$ (esempio molto difficile): peso = $(1-0.1)^2 = 0.81$ (forte focus)

**2. Alpha (α ≈ 0.86, calcolato)**
```python
alpha_optimal = class_freq[0] / (class_freq[0] + class_freq[1])
```
- **Proporzionale alla frequenza**: Bilancia automaticamente per lo sbilanciamento
- **α = 0.86**: Peso maggiore alla classe minoritaria (1 - 0.86 = 0.14 per classe 1)

**3. Vantaggi Focal Loss**:
- **Focus sugli hard negatives**: Particolarmente utile con sbilanciamento estremo
- **Riduzione automatica**: Esempi facili (ben classificati) contribuiscono poco alla loss
- **Migliore separazione**: Forza il modello a distinguere casi ambigui

**4. Costi computazionali**:
- **Overhead**: ~15-20% più lento di BCE per calcolo potenza e logaritmi
- **Memory**: Identica a BCE
- **Trade-off**: Giustificato se migliora recall classe positiva

### Architettura con PCA

#### Struttura Ridotta
```
Input (14 PCA components) → Dense(32, ReLU) → Dropout(0.3) 
                         → Dense(16, ReLU) → Dropout(0.3)
                         → Dense(8, ReLU) → Dropout(0.3)
                         → Dense(1, Sigmoid)
```

#### Giustificazione

**1. Riduzione proporzionale dei neuroni**:
- **Input ridotto**: 14 componenti vs 21 feature → ~67% dell'input originale
- **Neuroni ridotti**: 32 vs 128 → ~25% (riduzione più aggressiva)
- **Motivazione**: PCA concentra l'informazione, serve meno capacità

**2. Pattern 32 → 16 → 8**:
- **Dimezzamento progressivo**: Stessa filosofia del modello originale
- **8 neuroni finali**: Sufficiente per decisione su spazio ridotto

**3. Dropout invariato (0.3)**:
- **Stesso rischio overfitting**: Anche con meno dimensioni, serve regolarizzazione
- **Dataset invariato**: 70k esempi richiedono stessa prevenzione

**4. Altri iperparametri identici**:
- Epoche 150, batch 256, Adam, BCE/Class Weights
- **Motivazione**: PCA modifica solo input, non dinamica di apprendimento

---

## Decision Tree

### Metodologia: Grid Search con Cross-Validation

A differenza degli approcci manuali precedenti, gli alberi di decisione sono stati ottimizzati tramite **Grid Search con 5-Fold Cross-Validation**, permettendo la selezione automatica degli iperparametri ottimali.

#### Griglia di Parametri
```python
dt_param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'ccp_alpha': [0, 0.0001, 0.0005, 0.001, 0.005, 0.01]
}
```

#### Giustificazione dei Parametri Testati

**1. max_depth (Profondità massima)**
- **Range [5, 10, 15, 20, None]**: Da albero semplice a crescita illimitata
- **5-10**: Modelli semplici, alta interpretabilità, possibile underfitting
- **15-20**: Compromesso complessità/generalizzazione
- **None**: Crescita completa, rischio overfitting ma utile come riferimento

**2. min_samples_split (Campioni minimi per split)**
- **Range [2, 5, 10, 20]**: Controllo granularità degli split
- **2 (default)**: Split anche con 2 campioni
- **10-20**: Previene split su sottoinsiemi troppo piccoli, riduce overfitting

**3. min_samples_leaf (Campioni minimi per foglia)**
- **Range [1, 2, 5, 10]**: Evita foglie con pochi esempi
- **1 (default)**: Foglie pure possibili
- **5-10**: Foglie più robuste statisticamente

**4. criterion (Criterio di split)**
- **gini**: Misura impurità, computazionalmente efficiente
- **entropy**: Information gain, teoricamente più "puro"
- **Differenza pratica**: Minima, ma testati entrambi

**5. ccp_alpha (Cost-Complexity Pruning)**
- **Range [0, 0.0001, 0.0005, 0.001, 0.005, 0.01]**
- **0**: Nessun pruning post-training
- **0.0001-0.001**: Pruning moderato
- **0.005-0.01**: Pruning aggressivo

#### Metrica di Ottimizzazione: F1-Score

**Giustificazione**:
- **Formula**: $F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$
- **Vantaggi per dataset sbilanciato**:
  - Non dominata dalla classe maggioritaria (come accuracy)
  - Bilancia precision e recall
  - Penalizza modelli che predicono sempre classe 0
- **Alternative considerate**:
  - Recall puro: Favorisce troppi falsi positivi
  - Precision pura: Favorisce troppi falsi negativi
  - Accuracy: Inadatta per 86/14 split

### Funzione di Grid Search

```python
def run_grid_search(X_train, y_train, param_grid, class_weight="balanced", cv=5, scoring="f1"):
    base_model = DecisionTreeClassifier(random_state=42, class_weight=class_weight)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search
```

**Parametri chiave**:
- **cv=5**: 5-Fold Cross-Validation per stima robusta
- **scoring="f1"**: Ottimizza F1-score
- **n_jobs=-1**: Parallelizzazione su tutti i core
- **return_train_score=True**: Monitora overfitting

### Configurazioni Testate

#### 1. Dataset Standard con Class Weights
```python
dt_best, grid_results = run_grid_search(
    X_train, y_train,
    dt_param_grid,
    class_weight="balanced",
    scoring="f1"
)
```

**Caratteristiche**:
- Dataset completo (~70k campioni)
- Class weight bilanciato per gestire sbilanciamento
- Selezione automatica iperparametri ottimali

#### 2. Dataset Bilanciato (Undersampling)
```python
dt_best_bal, grid_results_bal = run_grid_search(
    X_train_bal, y_train_bal,
    dt_param_grid,
    class_weight=None,  # Non necessario
    scoring="f1"
)
```

**Caratteristiche**:
- Dataset bilanciato 50/50 (~35k campioni)
- Nessun class weight necessario
- Grid Search più veloce su dataset ridotto

#### 3. Dataset con PCA
```python
dt_pca_best, grid_results_pca = run_grid_search(
    X_train_pca, y_train_pca,
    dt_param_grid,
    class_weight="balanced",
    scoring="f1"
)
```

**Caratteristiche**:
- 14 componenti PCA
- Class weight bilanciato
- Valuta effetto riduzione dimensionale

### Riepilogo Risultati Grid Search

```python
results_summary = pd.DataFrame({
    'Configurazione': [
        'No PCA - Standard Dataset',
        'No PCA - Balanced Dataset', 
        'PCA - Standard Dataset'
    ],
    'Best F1 (CV)': [
        grid_results.best_score_,
        grid_results_bal.best_score_,
        grid_results_pca.best_score_
    ],
    'Test Accuracy': [...],
    'Tree Depth': [...]
})
```

### Visualizzazione degli Alberi

#### Parametro max_depth=4
```python
plot_tree(tree, max_depth=4, ...)
```

**Giustificazione**:
- **Profondità reale**: Può variare in base agli iperparametri ottimali
- **Depth 4**: Mostra primi 4 livelli decisionali (più importanti)
- **16-31 foglie visibili**: Cattura pattern principali
- **Interpretabilità**: Clinici possono seguire logica decisionale

#### Feature Names
```python
plot_tree(tree, feature_names=numeric_features, ...)
```
- **Tracciabilità**: Collega split a feature medicali originali
- **Validazione clinica**: Esperti possono verificare sensatezza delle decisioni

---

## Confronto e Conclusioni

### Criteri di Valutazione

**In un contesto clinico, l'ordine di priorità è**:
1. **Recall classe positiva** (sensibilità): Minimizzare falsi negativi
2. **ROC-AUC**: Capacità discriminativa generale
3. **Precision classe positiva**: Ridurre falsi allarmi (secondario)
4. **Accuracy**: Meno rilevante con classi sbilanciate

### Risultati delle Reti Neurali

| Modello | Recall(1) | Precision(1) | AUC | FN | FP | Contesto d'uso |
|---------|-----------|--------------|-----|----|----|----------------|
| Base No PCA | ~0.50 | ~0.45 | 0.82 | Alto | Medio | Non raccomandato |
| Undersampling | ~0.78 | ~0.35 | 0.82 | Basso | Alto | Screening ampio |
| Class Weights No PCA | ~0.78 | ~0.31 | 0.84 | Basso | Molto Alto | Screening preventivo |
| Class Weights PCA | ~0.85 | ~0.28 | 0.84 | Molto Basso | Altissimo | Max sensibilità |
| Focal Loss | ~0.75 | ~0.32 | 0.83 | Medio-Basso | Alto | Non giustifica costo |

### Risultati Decision Tree (Grid Search)

| Configurazione | Best F1 (CV) | Test Accuracy | Tree Depth | Interpretabilità |
|----------------|--------------|---------------|------------|------------------|
| No PCA - Standard Dataset | Ottimizzato via CV | Variabile | Variabile | Media-Alta |
| No PCA - Balanced Dataset | Ottimizzato via CV | Variabile | Variabile | Alta |
| PCA - Standard Dataset | Ottimizzato via CV | Variabile | Variabile | Bassa |

**Note**: I valori esatti dipendono dai risultati del Grid Search. La configurazione ottimale viene selezionata automaticamente massimizzando l'F1-score.

### Raccomandazioni Finali

#### Per Screening su Larga Scala
**Modello raccomandato**: Neural Network con Class Weights (No PCA)
- **Recall 0.78**: Cattura 78% dei diabetici
- **AUC 0.84**: Eccellente capacità discriminativa
- **Utilizza tutti i dati**: No undersampling, max informazione
- **Costo computazionale**: Accettabile per deployment

#### Per Massima Sensibilità (Popolazioni a Rischio)
**Modello raccomandato**: Neural Network Class Weights con PCA
- **Recall 0.85**: Massima cattura di positivi
- **Trade-off**: 28% precision accettabile per alto rischio
- **Più veloce**: 14 features vs 21 in inference

#### Per Interpretabilità Clinica
**Modello raccomandato**: Decision Tree ottimizzato con Grid Search su Dataset Balanced
- **F1-score ottimizzato**: Iperparametri selezionati automaticamente
- **Interpretabile**: Medici possono seguire logica decisionale
- **Validabile**: Regole esplicitabili
- **Riproducibile**: Grid Search garantisce selezione sistematica

#### PCA: Quando Utilizzarla?
**Conclusione**: In questo progetto, **PCA offre benefici limitati**
- **Riduzione modesta**: 14/21 features (67%)
- **NN senza PCA**: Performance comparabili con più informazione
- **Decision Tree senza PCA**: Migliore con feature originali interpretabili
- **Unico vantaggio**: Inferenza leggermente più veloce se critico

**Quando considerare PCA**:
- Problemi con >100 features altamente correlate
- Vincoli hardware stringenti
- Visualizzazione 2D/3D richiesta

### Considerazioni sul Bilanciamento

#### Undersampling
**Pro**:
- Modello perfettamente bilanciato
- Addestramento veloce (50% dati)
- Ottimo per NN e DT

**Contro**:
- Perdita 60k esempi classe negativa
- Meno informazione su pattern complessi

#### Class Weights
**Pro**:
- Nessuna perdita di dati
- Flessibile (modificabile α)
- Efficiente

**Contro**:
- Training leggermente più instabile
- Richiede tuning α per ottimalità

#### Focal Loss
**Pro**:
- Focus automatico su hard examples
- Teoricamente superiore

**Contro**:
- 15-20% overhead computazionale
- Performance non significativamente migliori
- **Non giustificato** per questo problema

---

## Considerazioni Tecniche Aggiuntive

### Scelte di Implementazione

#### Validation Split vs K-Fold
**Scelta**: Validation split 10%
**Motivazione**: 
- Dataset grande (70k) → single split sufficiente
- K-fold richiederebbe 5-10x tempo di training
- Validation set 7k esempi statisticamente robusto

#### Batch Size e Memoria
**256 per dataset completo, 128 per balanced**
- GPU/CPU moderne gestiscono facilmente
- Tradeoff ottimale stabilità/velocità
- Valori tipici: 32-512 per problemi simili

#### Epoche e Early Stopping
**Scelta**: Fixed epochs senza early stopping
**Motivazione**:
- Monitoraggio manuale delle curve loss/accuracy
- Permette osservazione completa convergenza
- Early stopping rischierebbe stop prematuro con focal loss

### Riproducibilità

Tutti i modelli utilizzano:
- `random_state=42` (sklearn)
- `np.random.seed(42)` (numpy)
- Tensorflow: seed implicito in Sequential API

---

## Metriche di Valutazione

### Confusion Matrix
```
                Predicted
              0         1
Actual 0     TN        FP
       1     FN        TP
```

### Metriche Chiave

**Recall (Sensibilità)**:
$$Recall = \frac{TP}{TP + FN}$$
- **Interpretazione clinica**: % diabetici correttamente identificati
- **Priorità massima** in screening

**Precision (Valore Predittivo Positivo)**:
$$Precision = \frac{TP}{TP + FP}$$
- **Interpretazione clinica**: % test positivi realmente diabetici
- **Trade-off** con recall in contesto sbilanciato

**ROC-AUC**:
- Area sotto curva ROC
- Misura capacità discriminativa complessiva
- **Valore >0.8**: Buono, **>0.9**: Eccellente

**F1-Score**:
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$
- Media armonica precision/recall
- **Meno rilevante** quando recall è prioritario

---

## Possibili Estensioni Future

### Ensemble Methods
- **Random Forest**: Combinazione di decision trees
- **XGBoost/LightGBM**: Gradient boosting ottimizzato
- **Stacking**: Combinazione NN + DT

### Hyperparameter Tuning Sistematico
- **Grid Search**: Su α, dropout, learning rate
- **Random Search**: Esplorazione spazio parametri
- **Bayesian Optimization**: Tuning avanzato

### Tecniche di Bilanciamento Avanzate
- **SMOTE**: Synthetic oversampling
- **ADASYN**: Adaptive synthetic sampling
- **Combination**: SMOTE + undersampling

### Feature Engineering
- **Interaction features**: BMI × Age, etc.
- **Polynomial features**: Relazioni non-lineari
- **Domain knowledge**: Feature cliniche derivate

### Threshold Optimization
- **Ottimizzazione soglia**: Trade-off recall/precision customizzato
- **Youden Index**: Massimizzazione sensibilità + specificità
- **Costo-beneficio**: Pesatura economica FN vs FP

---

## Riferimenti e Best Practices

### Letteratura
- Lin et al. (2017): "Focal Loss for Dense Object Detection" - origine Focal Loss
- Chawla et al. (2002): "SMOTE" - oversampling tecnica
- sklearn documentation: Class weight implementation

### Best Practices Adottate
✅ Standardizzazione prima di PCA e NN
✅ Split stratificato per dataset bilanciato
✅ Validation set per monitoraggio overfitting
✅ Multiple approaches per confronto robusto
✅ Metriche appropriate per sbilanciamento
✅ Visualizzazioni complete (ROC, confusion matrix, learning curves)
✅ Documentazione parametri e motivazioni

### Codice Riproducibile
- Random seeds consistenti
- Versioni librerie specificate in `requirements.txt`
- Funzioni riutilizzabili (`evaluate_and_plot`, `balance_dataset`)
- Commenti inline per chiarezza

---

## Conclusioni Generali

Questo progetto dimostra come:

1. **Lo sbilanciamento delle classi** richieda approcci specifici (class weights, undersampling)
2. **La PCA non sia sempre vantaggiosa**, specialmente con feature debolmente correlate
3. **Le reti neurali offrano flessibilità** con diversi approcci di bilanciamento
4. **I decision trees bilancino** performance e interpretabilità
5. **Il contesto applicativo** (screening vs diagnosi) guidi la scelta del modello

Il modello **Neural Network con Class Weights (No PCA)** emerge come soluzione ottimale per il trade-off performance/utilizzo dati/deployability in un contesto di screening del diabete.
