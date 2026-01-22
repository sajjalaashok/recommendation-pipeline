# Run Recommendation Pipeline
# ---------------------------

Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "`n[1/5] Running Data Validation..." -ForegroundColor Cyan
python data_validation.py
if ($LASTEXITCODE -ne 0) { Write-Error "Validation failed"; exit 1 }

Write-Host "`n[2/5] Running Data Preprocessing..." -ForegroundColor Cyan
python data_preprocessing.py
if ($LASTEXITCODE -ne 0) { Write-Error "Preprocessing failed"; exit 1 }

Write-Host "`n[3/5] Running Transform (Enrichment)..." -ForegroundColor Cyan
python transform.py
if ($LASTEXITCODE -ne 0) { Write-Error "Transformation failed"; exit 1 }

Write-Host "`n[4/5] Building Feature Store..." -ForegroundColor Cyan
python feature_store.py
if ($LASTEXITCODE -ne 0) { Write-Error "Feature Store build failed"; exit 1 }

Write-Host "`n[5/5] Training Model..." -ForegroundColor Cyan
python train.py
if ($LASTEXITCODE -ne 0) { Write-Error "Training failed"; exit 1 }

Write-Host "`nPipeline Completed Successfully!" -ForegroundColor Green
