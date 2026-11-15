# ML_practice_encoder

Практикум по машинному обучению: внедрение ML-модели предсказания ошибки угла
(энкодер − эталон) и сравнение её с разложением сигнала в ряд Фурье.

## Что внутри

- `head_period_analysis.py` — оборотная статистика углов (исходный скрипт;
  импортируется как модуль для получения целевой кривой).
- `ml_error_prediction.py` — минимальный ML-пайплайн: SARIMAX, деревья/GBDT
  (DecisionTree, RandomForest, GradientBoosting, HistGradientBoosting),
  BiGRU (seq2seq, PyTorch) против ряда Фурье (МНК; K по BIC и по CV).
  Протокол: блочный тест 80/20 по оборотам + 5-fold GroupKFold по оборотам.
- `REPORT.md` — отчёт: постановка, методика, результаты, выводы.
- `test_230400_no_ARU_s_20000/`, `test_230400_no_ARU_s_20000_after_calib/` — наборы данных (CSV).
- `requirements-ml.txt` — зависимости (venv).
- `outputs/` (в .gitignore) — метрики и графики.

## Быстрый старт

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-ml.txt
.venv/bin/python head_period_analysis.py test_230400_no_ARU_s_20000 --no-plots --no-show
.venv/bin/python ml_error_prediction.py                 # оба набора
.venv/bin/python ml_error_prediction.py --skip-gru      # без GRU
```

Целевая кривая — остаток сигнала `circular_mean` из `head_period_analysis.py`.
Целевые метрики и выводы — в [REPORT.md](REPORT.md); сырые значения — в
`outputs/<набор>/metrics_*.csv`.
