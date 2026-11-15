# ML_practice_encoder

Внедрение ML алгоритма предсказания ошибки угла в рамках практикума по машинному обучению.

## Что внутри

- `head_period_analysis.py` — оборотная статистика углов (исходный скрипт анализа;
  используется как модуль для получения целевых кривых).
- `ml_error_prediction.py` — ML-пайплайн сравнения моделей предсказания
  периодической ошибки угла (энкодер − эталон): ARIMA-семейство (SARIMAX),
  деревья/GBDT (DecisionTree, RandomForest, GradientBoosting,
  HistGradientBoosting), DNN (двунаправленная GRU seq2seq, PyTorch) —
  против классического разложения ошибки в ряд Фурье (МНК).
- `REPORT.md` — полный отчёт: постановка, методика (блочный тест + GroupKFold CV),
  метрики (RMSE/MAE/MaxAbs/R², p2p до/после коррекции), выводы.
- `test_230400_no_ARU_s_20000/` — запись до калибровки (57 оборотов, ~1576 отсчётов/оборот).
- `test_230400_no_ARU_s_20000_after_calib/` — запись после калибровки (58 оборотов).
- `requirements-ml.txt` — зависимости (venv).
- `outputs/` (в .gitignore) — метрики, графики сравнения и лог прогона.

## Быстрый старт

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-ml.txt
.venv/bin/python head_period_analysis.py test_230400_no_ARU_s_20000 --no-plots --no-show
.venv/bin/python ml_error_prediction.py                 # оба набора, все 4 таргета
.venv/bin/python ml_error_prediction.py --skip-gru      # без DNN (быстрее)
```

Итоговые таблицы и выводы — в [REPORT.md](REPORT.md); сырые метрики — в
`outputs/<набор>/metrics_*.csv`.
