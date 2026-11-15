#!/usr/bin/env python3
"""ML-предсказание ошибки угла (circular_mean): сравнение моделей с рядом Фурье.

Минимально рабочая версия. Постановка: по отсчётам абсолютного энкодера
(две записи, папки test_*) предсказать остаток сигнала circular_mean,
полученный из head_period_analysis.py (импортируется как модуль, timing не
используется). Сравнение трёх семейств моделей с разложением в ряд Фурье.

Данные и цель:
  * Нарезка по оборотам первой головки, единая угловая сетка [0, 360),
    вычитание общего ската и гладкого базового уровня (окно ≈ оборот) —
    как в head_period_analysis.py. Остаток — целевой ряд.
  * Сетка: 1576 бинов/оборот; 57/58 оборотов на набор.
  * Полуразность головок в анализе, обучении и оценке не используется.

Признаки (одинаковые для всех ML-моделей):
  * sin/cos k*theta, k=1..8, и нормированный угол (theta/pi − 1); модель
    предсказывает кривую y(theta), применяемую к любому обороту.

Протокол оценки (единый для всех моделей):
  * Все гиперпараметры выбираются только на обучающей выборке.
  * Основной протокол — блочное разбиение по оборотам 80/20 (обучение на
    первых оборотах, тест на последних).
  * Контрольная оценка — 5-fold GroupKFold по оборотам.
  * Модели оцениваются на тестовых отсчётах всех отложенных оборотов и на
    средней тестовой кривой. Рядом приводится межоборотное СКО (выборочная
    дисперсия остатка между оборотами) — масштаб несводимого разброса.

Метрики: RMSE / MAE / MaxAbs / R² (на отсчётах и на средней кривой), размах
остатка (p2p) до и после предсказания, RMSE на обучающей выборке.
1° = 3600 угловых секунд.

Примеры:
  .venv/bin/python ml_error_prediction.py                 # оба набора
  .venv/bin/python ml_error_prediction.py --skip-gru      # без GRU
  .venv/bin/python ml_error_prediction.py --exp test_230400_no_ARU_s_20000
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import head_period_analysis as hpa

ARCSEC = 3600.0
TARGET_SIGNAL = 'circular_mean'
EXPERIMENTS = [
    'test_230400_no_ARU_s_20000',
    'test_230400_no_ARU_s_20000_after_calib',
]
OUTDIR = Path(__file__).resolve().parent / 'outputs'


# ---------------------------------------------------------------------------
# 1. Данные и целевая кривая
# ---------------------------------------------------------------------------

def build_profile(exp_name: str, signal: str = TARGET_SIGNAL):
    path = hpa.resolve_csv(exp_name)
    df, columns = hpa.load_log(path)
    x_ref, ang_ref, signals = hpa.build_signals(df, columns, 'seq')
    crossings, direction, _, angle_grid, xc, ref_slope = hpa.build_reference_grid(
        x_ref, ang_ref, None)
    profile = hpa.analyze_aligned(signals[signal], crossings, direction,
                                  angle_grid, xc, ref_slope,
                                  x_label=f'Угол по {columns[0]}, град')
    return profile, path


def data_quality_report(exp_name: str) -> dict:
    path = hpa.resolve_csv(exp_name)
    df, columns = hpa.load_log(path)
    report = {'experiment': exp_name, 'rows': int(len(df))}
    for c in ('ang_1', 'ang_2'):
        if c in df:
            v = pd.to_numeric(df[c], errors='coerce')
            report[f'{c}_nan_pct'] = float(v.isna().mean() * 100)
            report[f'{c}_out_of_range'] = int(((v < 0) | (v >= 360)).sum())
    x_ref, ang_ref, signals = hpa.build_signals(df, columns, 'seq')
    crossings, direction, lengths, angle_grid, xc, slope = hpa.build_reference_grid(
        x_ref, ang_ref, None)
    report['n_revolutions'] = int(len(lengths))
    report['period_samples_median'] = float(np.median(lengths))
    report['period_samples_std'] = float(np.std(lengths))
    report['grid_bins'] = int(len(angle_grid))
    return report


# ---------------------------------------------------------------------------
# 2. Признаки и вспомогательные функции
# ---------------------------------------------------------------------------

def harmonic_design(angle_deg: np.ndarray, harmonics: int) -> np.ndarray:
    """Матрица дизайна [1, cos kθ, sin kθ, k=1..H] для разложения Фурье (МНК)."""
    th = np.deg2rad(np.asarray(angle_deg, dtype=float))
    cols = [np.ones_like(th)]
    for k in range(1, harmonics + 1):
        cols.append(np.cos(k * th))
        cols.append(np.sin(k * th))
    return np.column_stack(cols)


def harmonic_features(angle_deg: np.ndarray, harmonics: int = 8) -> np.ndarray:
    """Признаки для ML-моделей: sin/cos kθ, k=1..H, и нормированный угол."""
    th = np.deg2rad(np.asarray(angle_deg, dtype=float))
    cols = []
    for k in range(1, harmonics + 1):
        cols.append(np.sin(k * th))
        cols.append(np.cos(k * th))
    cols.append(th / math.pi - 1.0)
    return np.column_stack(cols)


def periodic_interp(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Линейная интерполяция с периодическим продолжением по 360°."""
    xp = np.concatenate([x - 360.0, x, x + 360.0])
    yp = np.concatenate([y, y, y])
    order = np.argsort(xp)
    return np.interp(xq, xp[order], yp[order])


# ---------------------------------------------------------------------------
# 3. Модели
# ---------------------------------------------------------------------------

class FourierModel:
    """Ряд Фурье (МНК по средней кривой оборота). Эквивалентен ДПФ на равной
    сетке полного периода. K выбирается по BIC или по групповой CV на трейне."""

    def __init__(self, harmonics: int):
        self.harmonics = int(harmonics)
        self.coef_: np.ndarray | None = None

    def fit(self, angle_deg, y_mean):
        mask = np.isfinite(y_mean)
        A = harmonic_design(angle_deg[mask], self.harmonics)
        y = y_mean[mask]
        try:  # гармоники на полном периоде ортогональны — решаем через Грама
            G = A.T @ A
            B = A.T @ y
            L = np.linalg.cholesky(G)
            self.coef_ = np.linalg.solve(L.T, np.linalg.solve(L, B))
        except np.linalg.LinAlgError:
            self.coef_, *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict(self, angle_deg):
        return harmonic_design(angle_deg, self.harmonics) @ self.coef_

    @staticmethod
    def select_harmonics_bic(angle_deg, y_mean, k_max: int = 400) -> int:
        """K по BIC (гауссовы остатки); быстро — через Грама и Холецкого."""
        mask = np.isfinite(y_mean)
        n = int(mask.sum())
        y = y_mean[mask]
        k_max = int(min(k_max, (n - 1) // 2))
        A = harmonic_design(angle_deg[mask], k_max)
        G = A.T @ A
        B = A.T @ y
        yty = float(y @ y)
        log_n = math.log(n)
        best_k, best_bic = 1, np.inf
        for k in range(1, k_max + 1):
            p = 2 * k + 1
            Gp = G[:p, :p]
            try:
                L = np.linalg.cholesky(Gp)
                coef = np.linalg.solve(L.T, np.linalg.solve(L, B[:p]))
            except np.linalg.LinAlgError:
                coef, *_ = np.linalg.lstsq(Gp, B[:p], rcond=None)
            rss = max(yty - 2.0 * float(B[:p] @ coef) + float(coef @ Gp @ coef), 1e-18)
            bic = n * math.log(rss / n) + p * log_n
            if bic < best_bic - 1e-9:
                best_bic, best_k = bic, k
        return best_k

    @staticmethod
    def select_harmonics_cv(angle_deg, rows,
                            k_grid=(4, 8, 12, 16, 24, 32, 48, 64, 80, 96,
                                    128, 157, 192, 256, 384),
                            n_splits=5):
        """K по 5-fold GroupKFold по оборотам (критерий — RMSE средней кривой
        на отложенных фолдах). Тестовые данные не используются."""
        from sklearn.model_selection import GroupKFold
        rows = np.asarray(rows, dtype=float)
        gkf = GroupKFold(n_splits=n_splits)
        cv_rmse = {k: [] for k in k_grid}
        for train_idx, val_idx in gkf.split(rows, None, np.arange(rows.shape[0])):
            mean_tr = np.nanmean(rows[train_idx], axis=0)
            mean_va = np.nanmean(rows[val_idx], axis=0)
            for k in k_grid:
                pred = FourierModel(k).fit(angle_deg, mean_tr).predict(angle_deg)
                err = (pred - mean_va)[np.isfinite(mean_va)]
                cv_rmse[k].append(float(np.sqrt(np.mean(err ** 2))))
        means = pd.Series({k: float(np.mean(v)) for k, v in cv_rmse.items()})
        return int(means.idxmin()), means


class ArimaPredictor:
    """SARIMAX на прореженной сетке с сезонным периодом = оборот.

    Ряд = остаток, усреднённый по оборотам и прореженный до bins_per_rev;
    модель подбирается по AIC на трейне (малая сетка конфигураций), прогноз
    на n_revolutions вперёд; итоговая кривая — средний прогноз на оборот,
    перенесённый на исходную сетку периодической интерполяцией.
    """

    CONFIGS = [
        ((1, 0, 0), (1, 0, 0)),
        ((1, 0, 0), (0, 0, 1)),
        ((2, 0, 0), (1, 0, 1)),
        ((0, 0, 1), (1, 0, 1)),
        ((1, 0, 1), (1, 0, 1)),
        ((2, 0, 1), (0, 0, 1)),
    ]

    def __init__(self, bins_per_rev: int = 24):
        self.bins_per_rev = int(bins_per_rev)
        self.centers_ = (np.arange(self.bins_per_rev) + 0.5) * 360.0 / self.bins_per_rev
        self.best_config_: tuple | None = None
        self.best_aic_: float = np.inf
        self.model_ = None

    @staticmethod
    def _resample(angle_native, rows, centers):
        out = np.full((rows.shape[0], len(centers)), np.nan)
        for k in range(rows.shape[0]):
            row = rows[k]
            mask = np.isfinite(row)
            if mask.sum() >= 2:
                order = np.argsort(angle_native[mask])
                out[k] = periodic_interp(angle_native[mask][order], row[mask][order], centers)
        return out

    def fit(self, angle_native, train_rows):
        import statsmodels.api as sm
        series = self._resample(angle_native, train_rows, self.centers_).ravel()
        series = series[np.isfinite(series)]
        m = self.bins_per_rev
        for order, seas in self.CONFIGS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fit = sm.tsa.SARIMAX(series, order=order,
                                         seasonal_order=(*seas, m),
                                         trend='n').fit(disp=False, maxiter=150)
                if fit.aic < self.best_aic_:
                    self.best_aic_ = float(fit.aic)
                    self.best_config_ = (order, (*seas, m))
            except Exception:
                continue
        if self.best_config_ is None:
            raise RuntimeError('SARIMAX: ни одна конфигурация не сошлась')
        order, sorder = self.best_config_
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.model_ = sm.tsa.SARIMAX(series, order=order, seasonal_order=sorder,
                                         trend='n').fit(disp=False, maxiter=150)
        return self

    def predict_curve(self, angle_native, n_revolutions: int) -> np.ndarray:
        m = self.bins_per_rev
        start = len(self.model_.fittedvalues)
        forecast = np.asarray(self.model_.predict(start=start,
                                                  end=start + n_revolutions * m - 1))
        per_rev = forecast[: n_revolutions * m].reshape(n_revolutions, m)
        return periodic_interp(self.centers_, per_rev.mean(axis=0), angle_native)


def make_tree_models(seed: int):
    from sklearn.ensemble import (GradientBoostingRegressor,
                                  HistGradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.tree import DecisionTreeRegressor
    return {
        'DecisionTree (depth=8)': DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=100, random_state=seed),
        'RandomForest (n=300)': RandomForestRegressor(
            n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=seed),
        'GradientBoosting (n=300)': GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=seed),
        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.06, max_leaf_nodes=63,
            early_stopping=True, validation_fraction=0.15, random_state=seed),
    }


def train_gru(angle_native, train_rows, seed=0, hidden=32, layers=1,
              harmonics=8, max_epochs=80, batch_size=8, patience=12,
              val_frac=0.15, lr=3e-3):
    """BiGRU seq2seq: признаки угла -> ряд предсказаний по углу оборота."""
    import os
    import torch
    import torch.nn as nn

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.manual_seed(seed)

    n = train_rows.shape[0]
    n_val = max(2, int(round(n * val_frac))) if n > 10 else 0
    val_rows = train_rows[n - n_val:] if n_val else None
    fit_rows = train_rows[: n - n_val] if n_val else train_rows

    X = harmonic_features(angle_native, harmonics=harmonics)
    y_mean = float(np.nanmean(fit_rows))
    y_std = float(np.nanstd(fit_rows)) or 1.0

    Xt = torch.tensor(np.tile(X, (fit_rows.shape[0], 1, 1)), dtype=torch.float32)
    yt = torch.tensor(((np.nan_to_num(fit_rows) - y_mean) / y_std)[..., None],
                      dtype=torch.float32)

    class BiGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(X.shape[1], hidden, num_layers=layers,
                              batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.LayerNorm(2 * hidden),
                                      nn.Linear(2 * hidden, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 1))

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out)

    model = BiGRU()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    best_val, best_state, bad, epoch = np.inf, None, 0, -1
    idx = np.arange(fit_rows.shape[0])
    for epoch in range(max_epochs):
        model.train()
        np.random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            b = idx[start:start + batch_size]
            opt.zero_grad()
            loss = torch.mean((model(Xt[b]) - yt[b]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if val_rows is not None:
            model.eval()
            with torch.no_grad():
                pred_v = model(torch.tensor(X[None], dtype=torch.float32))[0, ..., 0].numpy() * y_std + y_mean
            err = (pred_v[None, :].repeat(len(val_rows), 0) - np.nan_to_num(val_rows)).ravel()
            val_rmse = float(np.sqrt(np.mean(err ** 2)))
            if val_rmse < best_val - 1e-8:
                best_val = val_rmse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        curve = model(torch.tensor(X[None], dtype=torch.float32))[0, ..., 0].numpy() * y_std + y_mean
    return curve, {'best_val_rmse_deg': float(best_val), 'epochs_ran': epoch + 1}


# ---------------------------------------------------------------------------
# 4. Метрики
# ---------------------------------------------------------------------------

def sample_metrics(y_true_rows: np.ndarray, pred_curve: np.ndarray) -> dict:
    pred = np.broadcast_to(pred_curve, y_true_rows.shape)
    mask = np.isfinite(y_true_rows)
    err = (pred - y_true_rows)[mask]
    yt = y_true_rows[mask]
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {'rmse_deg': float(np.sqrt(np.mean(err ** 2))),
            'mae_deg': float(np.mean(np.abs(err))),
            'maxabs_deg': float(np.max(np.abs(err))),
            'r2': float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')}


def profile_metrics(y_true_rows: np.ndarray, pred_curve: np.ndarray) -> dict:
    mean_true = np.nanmean(y_true_rows, axis=0)
    mask = np.isfinite(mean_true)
    err = (pred_curve - mean_true)[mask]
    resid = (mean_true - pred_curve)[mask]
    return {'profile_rmse_deg': float(np.sqrt(np.mean(err ** 2))),
            'profile_mae_deg': float(np.mean(np.abs(err))),
            'profile_maxabs_deg': float(np.max(np.abs(err))),
            'p2p_before_deg': float(np.nanmax(mean_true) - np.nanmin(mean_true)),
            'p2p_after_deg': float(resid.max() - resid.min()),
            'std_after_deg': float(np.std(resid))}


def between_revolution_std(rows: np.ndarray) -> float:
    """Межоборотное СКО остатка (дисперсия между оборотами, усреднённая по углу)."""
    return float(np.sqrt(np.nanmean(np.nanvar(rows, axis=0, ddof=1))))


def summarize_metrics(te_rows, tr_rows, pred_curve) -> dict:
    m = sample_metrics(te_rows, pred_curve)
    m.update(profile_metrics(te_rows, pred_curve))
    m['rmse_train_deg'] = sample_metrics(tr_rows, pred_curve)['rmse_deg']
    m['revvar_deg'] = between_revolution_std(tr_rows)
    return m


# ---------------------------------------------------------------------------
# 5. Протоколы оценки
# ---------------------------------------------------------------------------

def evaluate_blocked(profile, seed=0, arima_bins=24, harmonics=8, skip_gru=False):
    """Блочный протокол 80/20 по оборотам."""
    angle, rows = profile.angle_deg, profile.rows
    n_rev = rows.shape[0]
    n_train = max(4, int(math.floor(0.8 * n_rev)))
    train_rows, test_rows = rows[:n_train], rows[n_train:]

    results, predictions = [], {}

    def record(model_name, pred_curve, train_time, extra=None):
        m = summarize_metrics(test_rows, train_rows, pred_curve)
        rec = {'model': model_name, 'protocol': 'blocked_80_20',
               'n_train_rev': int(n_train), 'n_test_rev': int(n_rev - n_train),
               'train_time_s': round(train_time, 3)}
        rec.update(m)
        if extra:
            rec.update(extra)
        results.append(rec)
        predictions[model_name] = pred_curve
        print(f'    {model_name}: RMSE(тест)={m["rmse_deg"] * ARCSEC:.2f}", '
              f'RMSE(train)={m["rmse_train_deg"] * ARCSEC:.2f}", '
              f'время={train_time:.1f} с', flush=True)
        return pred_curve

    t0 = time.time()
    record('Без предсказания (0)', np.zeros_like(angle), time.time() - t0)

    t0 = time.time()
    mean_train = np.nanmean(train_rows, axis=0)
    k_bic = FourierModel.select_harmonics_bic(angle, mean_train)
    record(f'Ряд Фурье (K={k_bic} по BIC)',
           FourierModel(k_bic).fit(angle, mean_train).predict(angle),
           time.time() - t0, {'harmonics': k_bic})

    t0 = time.time()
    k_cv, cv_curve = FourierModel.select_harmonics_cv(angle, train_rows)
    record(f'Ряд Фурье (K={k_cv} по CV)',
           FourierModel(k_cv).fit(angle, mean_train).predict(angle),
           time.time() - t0, {'harmonics': k_cv})

    try:
        t0 = time.time()
        arima = ArimaPredictor(bins_per_rev=arima_bins).fit(angle, train_rows)
        cfg = arima.best_config_
        record(f'SARIMAX{cfg[0]}×({cfg[1][0]},{cfg[1][1]},{cfg[1][2]},s={cfg[1][3]})',
               arima.predict_curve(angle, n_revolutions=test_rows.shape[0]),
               time.time() - t0, {'arima_aic': arima.best_aic_})
    except Exception as exc:
        results.append({'model': 'SARIMAX', 'protocol': 'blocked_80_20',
                        'error': f'{type(exc).__name__}: {exc}'})

    X_train = np.tile(harmonic_features(angle, harmonics), (n_train, 1))
    y_train = np.nan_to_num(train_rows).ravel()
    X_angle = harmonic_features(angle, harmonics)
    for name, est in make_tree_models(seed).items():
        t0 = time.time()
        est.fit(X_train, y_train)
        record(name, est.predict(X_angle), time.time() - t0, {'harmonics': harmonics})

    if not skip_gru:
        t0 = time.time()
        curve, info = train_gru(angle, train_rows, seed=seed, harmonics=harmonics)
        record('BiGRU (seq2seq, H=8)', curve, time.time() - t0, info)

    return results, angle, rows, (train_rows, test_rows), \
        {'k_cv': k_cv, 'cv_curve_harmonics': cv_curve.to_dict(), 'predictions': predictions}


def evaluate_cv(profile, seed=0, harmonics=8, n_splits=5, arima_bins=24):
    """Контрольная оценка: 5-fold GroupKFold по оборотам (быстрые модели)."""
    from sklearn.model_selection import GroupKFold
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    angle, rows = profile.angle_deg, profile.rows
    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics: dict[str, list] = {}
    for train_idx, test_idx in gkf.split(rows, None, np.arange(rows.shape[0])):
        tr, te = rows[train_idx], rows[test_idx]
        mean_tr = np.nanmean(tr, axis=0)

        def add(name, curve):
            m = summarize_metrics(te, tr, curve)
            fold_metrics.setdefault(name, []).append(m)

        add('Без предсказания (0)', np.zeros_like(angle))
        k_bic = FourierModel.select_harmonics_bic(angle, mean_tr)
        add('Ряд Фурье (BIC)', FourierModel(k_bic).fit(angle, mean_tr).predict(angle))

        try:
            arima = ArimaPredictor(bins_per_rev=arima_bins).fit(angle, tr)
            add('SARIMAX', arima.predict_curve(angle, n_revolutions=te.shape[0]))
        except Exception:
            pass

        X_tr = np.tile(harmonic_features(angle, harmonics), (tr.shape[0], 1))
        y_tr = np.nan_to_num(tr).ravel()
        X_all = harmonic_features(angle, harmonics)
        for name, est in {
            'DecisionTree (depth=8)': DecisionTreeRegressor(
                max_depth=8, min_samples_leaf=100, random_state=seed),
            'HistGradientBoosting': HistGradientBoostingRegressor(
                max_iter=500, learning_rate=0.06, max_leaf_nodes=63,
                early_stopping=True, validation_fraction=0.15, random_state=seed),
        }.items():
            est.fit(X_tr, y_tr)
            add(name, est.predict(X_all))

    summary = []
    keys = ['rmse_deg', 'mae_deg', 'maxabs_deg', 'r2', 'profile_rmse_deg',
            'p2p_after_deg', 'std_after_deg', 'rmse_train_deg', 'revvar_deg']
    for name, lst in fold_metrics.items():
        rec = {'model': name, 'protocol': f'cv_groupkfold_{n_splits}',
               'n_folds': len(lst)}
        for key in keys:
            vals = np.array([m[key] for m in lst], dtype=float)
            rec[f'{key}_mean'] = float(np.nanmean(vals))
            rec[f'{key}_std'] = float(np.nanstd(vals))
        summary.append(rec)
    return summary


# ---------------------------------------------------------------------------
# 6. Графики
# ---------------------------------------------------------------------------

def plot_comparison(exp_name, outdir, angle, test_rows, predictions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mean_test = np.nanmean(test_rows, axis=0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(angle, mean_test, 'k', lw=1.6, label='Тест (среднее по оборотам)')
    for label, curve in predictions.items():
        if label.startswith('Без предсказания'):
            continue
        axes[0].plot(angle, curve, lw=1.0, alpha=0.85, label=label)
    axes[0].set_ylabel('Ошибка угла, град')
    axes[0].set_title(f'{exp_name} / {TARGET_SIGNAL}: предсказание на тесте')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.3)

    axes[1].plot(angle, mean_test, 'k', lw=1.6)
    for label, curve in predictions.items():
        if label.startswith('Без предсказания'):
            continue
        axes[1].plot(angle, mean_test - curve, lw=1.0, alpha=0.85, label=f'Остаток: {label}')
    axes[1].axhline(0, color='black', lw=.5, alpha=.5)
    axes[1].set_xlabel('Угол оборота, град')
    axes[1].set_ylabel('Остаток, град')
    axes[1].grid(alpha=.3)
    axes[1].legend(fontsize=8)
    png = outdir / f'{exp_name}_{TARGET_SIGNAL}_comparison.png'
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def plot_rmse_bars(exp_name, outdir, metrics_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = [r for r in metrics_rows if 'rmse_deg' in r]
    names = [r['model'] for r in rows]
    vals = [r['rmse_deg'] * ARCSEC for r in rows]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.6 * len(names) + 1.5)))
    ax.barh([names[i] for i in order], [vals[i] for i in order], color='steelblue')
    ax.set_xlabel('RMSE на тесте, угл. сек')
    ax.set_title(f'{exp_name} / {TARGET_SIGNAL}: RMSE (меньше — лучше)')
    ax.grid(axis='x', alpha=.3)
    fig.tight_layout()
    png = outdir / f'{exp_name}_{TARGET_SIGNAL}_rmse.png'
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# 7. Оркестрация
# ---------------------------------------------------------------------------

def run_experiment(exp_name, outdir, seed=0, skip_gru=False, skip_cv=False,
                   arima_bins=24):
    outdir.mkdir(parents=True, exist_ok=True)
    exp_dir = outdir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 100)
    print(f'Набор: {exp_name}')

    dq = data_quality_report(exp_name)
    print('Качество данных:', json.dumps(dq, ensure_ascii=False))
    pd.DataFrame([dq]).to_csv(exp_dir / 'data_quality.csv', index=False)

    t0 = time.time()
    profile, _ = build_profile(exp_name, TARGET_SIGNAL)
    angle, rows = profile.angle_deg, profile.rows
    print(f'Оборотов={rows.shape[0]}, сетка={len(angle)}, '
          f'p2p средней кривой={np.nanmax(profile.mean) - np.nanmin(profile.mean):.6f} град, '
          f'межоборотное СКО={between_revolution_std(rows) * ARCSEC:.2f}"')

    blocked, angle, rows, (train_rows, test_rows), info = evaluate_blocked(
        profile, seed=seed, arima_bins=arima_bins, skip_gru=skip_gru)
    for rec in blocked:
        rec['experiment'] = exp_name

    dfm = pd.DataFrame(blocked)
    dfm.to_csv(exp_dir / 'metrics_blocked.csv', index=False)
    show_cols = [c for c in ['model', 'rmse_deg', 'rmse_train_deg', 'maxabs_deg',
                             'r2', 'profile_rmse_deg', 'p2p_after_deg'] if c in dfm.columns]
    print(dfm[show_cols].to_string(index=False))

    keep = ('по CV', 'RandomForest', 'HistGradientBoosting', 'BiGRU', 'SARIMAX')
    preds = {k: v for k, v in info['predictions'].items()
             if any(tok in k for tok in keep)}
    plot_comparison(exp_name, exp_dir, angle, test_rows, preds)
    plot_rmse_bars(exp_name, exp_dir, blocked)
    pd.Series(info['cv_curve_harmonics']).to_csv(
        exp_dir / 'fourier_cv_curve.csv', header=['rmse_deg'])

    if not skip_cv:
        print('5-fold GroupKFold по оборотам...')
        cv = evaluate_cv(profile, seed=seed, arima_bins=arima_bins)
        for rec in cv:
            rec['experiment'] = exp_name
        pd.DataFrame(cv).to_csv(exp_dir / 'metrics_cv.csv', index=False)
        for rec in cv:
            print(f'    {rec["model"]}: RMSE={rec["rmse_deg_mean"] * ARCSEC:.2f} ± '
                  f'{rec["rmse_deg_std"] * ARCSEC:.2f}"', flush=True)

    print(f'Время набора: {time.time() - t0:.1f} с')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exp', action='append', default=None,
                        help='Папка набора данных (можно повторять)')
    parser.add_argument('--output-dir', type=Path, default=OUTDIR)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--arima-bins', type=int, default=24)
    parser.add_argument('--skip-gru', action='store_true')
    parser.add_argument('--skip-cv', action='store_true')
    args = parser.parse_args()

    experiments = args.exp or EXPERIMENTS
    t0 = time.time()
    for exp_name in experiments:
        run_experiment(exp_name, args.output_dir, seed=args.seed,
                       skip_gru=args.skip_gru, skip_cv=args.skip_cv,
                       arima_bins=args.arima_bins)
    print('=' * 100)
    print(f'Готово за {time.time() - t0:.1f} с. Результаты в {args.output_dir}')


if __name__ == '__main__':
    main()
