# notebooks/

Analysis notebooks (added as experiments run), e.g.:
- `01_data_exploration.ipynb`: attribute distributions, window lengths, label noise.
- `02_results.ipynb`: tables and figures from `outputs/eval/*/summary.json`.
- `03_error_analysis.ipynb`: failure cases by attribute, genre and description style.

Training itself runs as scripts (`pixi run -e cuda train-composer`, `train-parser`), so
it works the same on the laptop GPU and on Duke CS OnDemand.
