# Fine-Tuning the Sentinel NLI Model

This folder contains **all training scripts** for fine-tuning the hallucination detector.
These scripts are **completely isolated** from the main backend — they never run at server startup.

## Workflow

```
Step 1: prepare_data.py   →  downloads + formats datasets
Step 2: train.py          →  fine-tunes the model (run in Colab!)
Step 3: evaluate.py       →  benchmarks fine-tuned vs baseline
Step 4: copy the model    →  drop it into backend/models/
```

---

## Option A — Run in Google Colab (Recommended, Free GPU)

1. Upload this entire `finetune/` folder to your Google Drive
2. Open a new Colab notebook
3. Run these cells:

```python
# Cell 1 — Mount Drive + Install deps
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/finetune
!pip install -q transformers datasets accelerate evaluate scikit-learn
```

```python
# Cell 2 — Prepare data
!python prepare_data.py
```

```python
# Cell 3 — Train
!python train.py
```

```python
# Cell 4 — Evaluate
!python evaluate.py
```

```python
# Cell 5 — Zip model for download
import shutil
shutil.make_archive('sentinel-finetuned', 'zip', './models/sentinel-finetuned')
from google.colab import files
files.download('sentinel-finetuned.zip')
```

---

## Option B — Run Locally

```powershell
cd backend
.\.venv\Scripts\activate.ps1
pip install -r finetune/requirements_finetune.txt
cd finetune
python prepare_data.py
python train.py
python evaluate.py
```

---

## After Training — Integrate Into Backend

1. Unzip the downloaded model (if from Colab)
2. Place it at: `backend/models/sentinel-finetuned/`
3. The backend will **auto-detect** it on next startup — no code changes needed!

```
backend/
  models/
    sentinel-finetuned/    ← put your downloaded folder here
      config.json
      model.safetensors
      tokenizer_config.json
      ...
```

4. Restart the backend and you'll see in logs:
   ```
   INFO: Sentinel ready | model=LOCAL fine-tuned | load time: 1200ms
   ```

---

## Datasets Used

| Dataset | Source | Size | Purpose |
|---|---|---|---|
| HaluEval | Li et al. 2023 | ~35K | Real ChatGPT hallucinations |
| FEVER | Thorne et al. 2018 | ~185K | Fact verification NLI pairs |
| TruthfulQA | Lin et al. 2022 | ~817 | Factual accuracy benchmark |

---

## Expected Results

| Metric | Baseline | Fine-tuned (expected) |
|---|---|---|
| Accuracy | ~74% | ~84–88% |
| Macro F1 | ~0.71 | ~0.82–0.86 |
| Contradiction F1 | ~0.68 | ~0.84 |
