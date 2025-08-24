# Kanserli Hücre Tespiti & Şeker Hastalığı Tespiti 

> Bu README, **iki Jupyter defteri** için **tek ve aynı** şablon olarak tasarlanmıştır:
>
> * `KanserliHucreTeshisi.ipynb`
> * `SekerHastaligiTespit.ipynb`
>
> İkisi de bir **sınıflandırma** problemidir ve veri hazırlama → modelleme → değerlendirme akışı aynıdır.

## 1) Proje Özeti

Bu çalışma, gözetimli öğrenme ile **sağlık alanında sınıflandırma** modelleri geliştirmeyi hedefler:

* **Kanserli Hücre Tespiti:** Hücresel ölçümlerden *malign/benign* ayrımı.
* **Şeker Hastalığı Tespiti:** Klinik ölçümlerden *diyabet riski* tahmini (0/1).

Her iki not defterinde de tipik iş akışı:

1. **Veri Yükleme & Keşif (EDA)**
2. **Ön İşleme** (eksik değer, kodlama, ölçekleme)
3. **Modelleme** (LR, RF, SVM, KNN, XGBoost vb.)
4. **Değerlendirme** (Accuracy, Precision, Recall, F1, ROC-AUC; Confusion Matrix/ROC grafikleri)
5. **(Opsiyonel) Model Kaydetme** (`joblib`/`pickle`)

> **Not:** Bu README, not defterleri için standart bir çerçeve sunar. Veri sütun adları ve bazı hiperparametreler projeye göre özelleştirilmelidir.

---

## 2) Önerilen Klasör Yapısı

```
project_root/
├─ data/
│  ├─ raw/                  # orijinal veri dosyaları (CSV/Excel)
│  └─ processed/            # temizlenmiş/özellik mühendisliği yapılmış veri
├─ notebooks/
│  ├─ KanserliHucreTeshisi.ipynb
│  └─ SekerHastaligiTespit.ipynb
├─ models/                  # kayıtlı modeller (.joblib/.pkl)
├─ reports/                 # çıktı görselleri (cm.png, roc.png, vb.)
├─ requirements.txt
└─ README.md                # bu dosya
```

---

## 3) Kurulum

### a) Ortam

Python **3.10+** önerilir. Sanal ortam oluşturup etkinleştirin:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### b) Gereksinimler (örnek)

> Not: Not defterleri şu an boş göründüğü için liste **genel** tutulmuştur. Kendi import’larınıza göre daraltıp genişletebilirsiniz.

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.13.0
jupyter>=1.0.0
xgboost>=2.0.0           # kullanmayacaksanız kaldırın
joblib>=1.3.0
```

Kurulum:

```bash
pip install -r requirements.txt
```

---

## 4) Veri Beklentisi

* Girdi dosyası **CSV/Excel** formatında olmalıdır (örn. `data/raw/dataset.csv`).
* **Hedef değişken** (etiket) belirgin olmalıdır (örn. `target`, `diagnosis`, `Outcome`).
* Kategorik/sayısal sütun ayrımı net olmalıdır.

**Genel şema (örnek):**

| Sütun       | Tip          | Açıklama                                   |
| ----------- | ------------ | ------------------------------------------ |
| `feature_1` | float/int    | Sayısal özellik                            |
| `feature_2` | float/int    | Sayısal özellik                            |
| `feature_3` | category/str | Kategorik özellik                          |
| `...`       | ...          | ...                                        |
| `target`    | int/str      | **Sınıf etiketi** (0/1, Benign/Malign vb.) |

**Veri yolu parametreleri (defter başında öneri):**

```python
DATA_PATH = "data/raw/dataset.csv"
TARGET_COL = "target"          # kendi hedef kolonunuza göre güncelleyin
RANDOM_STATE = 42
TEST_SIZE = 0.2
```

> **Dengesiz sınıflar** için: stratified split, `class_weight="balanced"` veya **SMOTE** gibi yöntemleri ek olarak değerlendirin.

---

## 5) Not Defterlerini Çalıştırma

1. Jupyter’ı başlatın:

```bash
jupyter notebook
# veya
jupyter lab
```

2. İlgili `.ipynb` dosyasını açın:
   `notebooks/KanserliHucreTeshisi.ipynb` veya `notebooks/SekerHastaligiTespit.ipynb`
3. Hücreleri **sırayla** çalıştırın.
4. Modelleme kısmında birden fazla algoritmayı deneyin; sonuçları tablo halinde özetleyin.

---

## 6) Tipik Modelleme Bloğu (örnek)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

candidates = {
    "LogReg": LogisticRegression(max_iter=1000, n_jobs=None, class_weight="balanced"),
    "RF":     RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"),
    "SVM":    SVC(probability=True, kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
}

summary = []
for name, model in candidates.items():
    model.fit(X_train_sc, y_train)
    proba = model.predict_proba(X_test_sc)[:, 1] if hasattr(model, "predict_proba") else None
    pred  = model.predict(X_test_sc)
    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba) if proba is not None else None
    }
    summary.append(row)

import pandas as pd
pd.DataFrame(summary).sort_values(by="f1", ascending=False)
```

---

## 7) Değerlendirme & Görselleştirme (örnek)

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

best = candidates["RF"]   # örnek: en iyi model seçimi

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(best, X_test_sc, y_test)
plt.title("Confusion Matrix - RF")
plt.tight_layout()
plt.savefig("reports/confusion_matrix_rf.png", dpi=200)

# ROC Curve
if hasattr(best, "predict_proba"):
    RocCurveDisplay.from_estimator(best, X_test_sc, y_test)
    plt.title("ROC Curve - RF")
    plt.tight_layout()
    plt.savefig("reports/roc_curve_rf.png", dpi=200)
```

---

## 8) Modeli Kaydetme & Yükleme (opsiyonel)

```python
import joblib

joblib.dump(best, "models/best_model.joblib")

# ...
loaded = joblib.load("models/best_model.joblib")
loaded.predict(X_test_sc)
```

---

## 9) Yeniden Üretilebilirlik

* **Rastgelelik:** `random_state` sabitleyin.
* **Versiyonlama:** `requirements.txt` bulundurun, paket sürümlerini not edin.
* **Tohumlama:** NumPy/Scikit-learn için tohum belirleyin.

```python
import numpy as np
np.random.seed(42)
```

---

## 10) Proje Özgü Notlar

* **Kanserli Hücre Tespiti:**

  * Hedef değişken: *malign/benign* (ör. `target ∈ {0,1}` veya `diagnosis ∈ {M,B}`)
  * Görüntü tabanlı ise önce öznitelik çıkarımı; tablo verisi ise doğrudan modelleme.
* **Şeker Hastalığı Tespiti:**

  * Hedef değişken: *diyabet durumu* (0/1)
  * Dengesizlik görülebilir → stratified split, `class_weight`, SMOTE gibi yöntemler önerilir.

> **Aynı README**: Bu dosya iki defter için aynıdır. Sadece `DATA_PATH` ve `TARGET_COL` değerlerini, ayrıca gerektiğinde model/parametreleri ilgili verinize göre değiştirin.

---

## 11) Sorun Giderme

* **Boş / açılmayan defter:** Not defterleri boş görünüyorsa (içerik yoksa), önce kod hücrelerinizi ekleyin.
* **ImportError:** `pip list` ile paketleri doğrulayın; doğru **sanal ortamda** olduğunuzdan emin olun.
* **CSV okuma hataları:** `encoding="utf-8"` veya `sep=";"` gibi parametreleri deneyin.
* **Dengesiz sınıf:** `class_weight`, yeniden örnekleme, veya farklı eşik (threshold) denemeleri uygulayın.
* **Overfitting:** çapraz doğrulama, düzenlileştirme, erken durdurma, özellik seçimi kullanın.

---

## 12) Lisans

Uygun bir lisans seçin (örn. **MIT**) ve `LICENSE` dosyası ekleyin.

## 13) Atıf / Kaynakça (opsiyonel)

* Scikit-learn User Guide
* XGBoost Documentation (kullanıldıysa)
* Veri kümesi kaynağı (UCI, Kaggle, kurum içi veri, vb.)

---

**İletişim & Katkı**
Öneri/sorularınız için issue/PR açabilirsiniz. Bu README, iki not defterini **tek şablon** ile standardize etmek için hazırlanmıştır.

