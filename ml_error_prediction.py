#!/usr/bin/env python3
"""ML-модели для предсказания периодической ошибки угла энкодера (энкодер − эталон).

Практикум по машинному обучению: сравнение трёх семейств моделей с классическим
численным методом (разложение в ряд Фурье) на сырых данных из папок test_*.

Целевая кривая — остаточная ошибка угла за оборот, полученная через
head_period_analysis.py (импортируется как модуль):

  * half_diff     — полуразность показаний двух головок, (ang_2 − ang_1)/2.
                    Это чистая эксцентриситетная ошибка каждой головки
                    (общий ход вычитается); межоборотный разброс ~1.3".
  * circular_mean — среднее EDA, общая систематическая составляющая.
  * ang_1 / ang_2 — остатки по каждой головке (содержат и общий ход).

Протокол сравнения (одинаковый для всех моделей):

  * Блочное разбиение по оборотам: первые train_frac оборотов — обучение,
    остальные — отложенный тест (калибровка на прошлых оборотах, проверка
    на будущих). Перемешивания нет — это важно для временного ряда.
  * Дополнительно — 5-fold GroupKFold по оборотам (все обороты равноправны
    в оценке; метрики усредняются по фолдам).
  * Модели видят только угол внутри оборота (гармоники sin/cos kθ), поэтому
    предсказание — детерминированная кривая поправки f(θ), применимая онлайн.
  * Отчётные метрики: RMSE / MAE / MaxAbs / R² по отсчётам тестовых оборотов
    и по средней тестовой кривой; p2p и СКО остатка до/после коррекции
    (в градусах и угловых секундах); межоборотный «шумовой пол» (СКО остатка
    между оборотами) — теоретический минимум достижимой ошибки.

Семейства моделей:

  1. ARIMA-семейство — SARIMAX на прореженной угловой сетке (grid_step=15°,
     сезонный период = оборот), выбор конфигурации по AIC на обучающей выборке.
     Адаптировано для периодического сигнала с периодом = оборот.
  2. Деревья / GBDT (sklearn): DecisionTree, RandomForest,
     GradientBoosting, HistGradientBoosting. Признаки — гармоники k=1..H.
  3. GRU (torch) — двунаправленная GRU seq2seq: на вход угол, закодированный
     гармониками, на выход — ошибка в каждом угле оборота.
  0. Классический метод — разложение в ряд Фурье (МНК; для полного периода
     эквивалентно ДПФ) с выбором числа гармоник по BIC.

Использование:

  .venv/bin/python ml_error_prediction.py                     # оба набора, все сигналы
  .venv/bin/python ml_error_prediction.py --exp test_...      # один набор
  .venv/bin/python ml_error_prediction.py --signals half_diff # только полуразность
  .venv/bin/python ml_error_prediction.py --skip-gru          # без GRU (быстрее)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import (GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
import os
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import head_period_analysis as hpa

ARCSEC = 3600.0  # 1 градус = 3600 угловых секунд

EXPERIMENTS = [
    'test_230400_no_ARU_s_20000',
    'test_230400_no_ARU_s_20000_after_calib',
]
TARGET_SIGNALS = ['half_diff', 'circular_mean', 'ang_1', 'ang_2']

OUTDIR = Path(__file__).resolve().parent / 'outputs'


# --------------------------------------------------------------------------
# 1. Загрузка целевых кривых через head_period_analysis
# --------------------------------------------------------------------------

def build_profile(exp_name: str, signal: str):
    """Строит периодический профиль сигнала через head_period_analysis."""
    path = hpa.resolve_csv(exp_name)
    df, columns = hpa.load_log(path)
    x_ref, ang_ref, signals = hpa.build_signals(df, columns, 'seq')
    crossings, direction, _, angle_grid, xc, ref_slope = hpa.build_reference_grid(
        x_ref, ang_ref, None)
    profile = hpa.analyze_aligned(
        signals[signal], crossings, direction, angle_grid, xc, ref_slope,
        x_label=f'Угол по {columns[0]}, град')
    return profile, path


def data_quality_report(exp_name: str) -> dict:
    """Проверка качества сырых CSV: пропуски, недопустимые углы, обороты."""
    path = hpa.resolve_csv(exp_name)
    df, columns = hpa.load_log(path)
    report = {'experiment': exp_name, 'csv': str(path), 'rows': int(len(df))}
    for c in ['ang_1', 'ang_2']:
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
    err = pd.to_numeric(df.get('error', pd.Series([0])), errors='coerce')
    report['error_column_nonzero'] = int((err.fillna(0) != 0).sum())
    return report


# --------------------------------------------------------------------------
# 2. Признаки и классические утилиты
# --------------------------------------------------------------------------

def harmonic_design(angle_deg: np.ndarray, harmonics: int) -> np.ndarray:
    """Матрица дизайна [1, cos kθ, sin kθ, k=1..H] для Фурье (МНК)."""
    th = np.deg2rad(np.asarray(angle_deg, dtype=float))
    cols = [np.ones_like(th)]
    for k in range(1, harmonics + 1):
        cols.append(np.cos(k * th))
        cols.append(np.sin(k * th))
    return np.column_stack(cols)


def harmonic_features(angle_deg: np.ndarray, harmonics: int = 8,
                      include_linear: bool = True) -> np.ndarray:
    """Признаки для ML: [sin kθ, cos kθ, k=1..H, (опц.) нормированный угол]."""
    th = np.deg2rad(np.asarray(angle_deg, dtype=float))
    cols = []
    for k in range(1, harmonics + 1):
        cols.append(np.sin(k * th))
        cols.append(np.cos(k * th))
    if include_linear:
        cols.append(th / math.pi - 1.0)  # нормированный угол в [-1, 1]
    return np.column_stack(cols)


def periodic_interp(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Периодическая (по 360°) линейная интерполяция."""
    xp = np.concatenate([x - 360.0, x, x + 360.0])
    yp = np.concatenate([y, y, y])
    order = np.argsort(xp)
    return np.interp(xq, xp[order], yp[order])


# --------------------------------------------------------------------------
# 3. Модели
# --------------------------------------------------------------------------

class FourierModel:
    """Разложение в ряд Фурье (МНК по средней кривой оборота)."""

    def __init__(self, harmonics: int):
        self.harmonics = int(harmonics)
        self.coef_: np.ndarray | None = None

    def fit(self, angle_deg, y_mean):
        mask = np.isfinite(y_mean)
        A = harmonic_design(angle_deg[mask], self.harmonics)
        y = y_mean[mask]
        # Гармоники на полном периоде почти ортогональны; решаем через Грама
        # (на порядки быстрее SVD-lstsq при K~сотнях). При вырождении — lstsq.
        try:
            G = A.T @ A
            B = A.T @ y
            L = np.linalg.cholesky(G)
            z = np.linalg.solve(L, B)
            self.coef_ = np.linalg.solve(L.T, z)
        except np.linalg.LinAlgError:
            self.coef_, *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict(self, angle_deg):
        return harmonic_design(angle_deg, self.harmonics) @ self.coef_

    @staticmethod
    def select_harmonics_bic(angle_deg, y_mean, k_max: int = 400) -> int:
        """Выбор числа гармоник по BIC (гауссовы остатки).

        RSS(k) считаются через матрицу Грама AᵀA / Aᵀy и разложение Холецкого
        для префиксов — O(K²n + K³) вместо O(K²n·K) у серии lstsq.
        """
        mask = np.isfinite(y_mean)
        n = int(mask.sum())
        y = y_mean[mask]
        k_max = int(min(k_max, (n - 1) // 2))
        A = harmonic_design(angle_deg[mask], k_max)  # n × (2K+1)
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
                z = np.linalg.solve(L, B[:p])
                coef = np.linalg.solve(L.T, z)
            except np.linalg.LinAlgError:
                coef, *_ = np.linalg.lstsq(Gp, B[:p], rcond=None)
            rss = yty - 2.0 * float(B[:p] @ coef) + float(coef @ Gp @ coef)
            rss = max(rss, 1e-18)
            bic = n * math.log(rss / n) + p * log_n
            if bic < best_bic - 1e-9:
                best_bic, best_k = bic, k
        return best_k

    @staticmethod
    def select_harmonics_cv(angle_deg, rows, k_grid=(4, 8, 12, 16, 24, 32, 48, 64, 80,
                                                     96, 128, 157, 192, 256, 384),
                            n_splits=5, seed=0) -> tuple[int, pd.Series]:
        """Выбор K по 5-fold GroupKFold по оборотам (критерий — RMSE средней
        кривой на отложенных оборотах). Не использует тестовые данные."""
        rows = np.asarray(rows, dtype=float)
        groups = np.arange(rows.shape[0])
        gkf = GroupKFold(n_splits=n_splits)
        cv_rmse = {k: [] for k in k_grid}
        for train_idx, val_idx in gkf.split(rows, None, groups):
            mean_tr = np.nanmean(rows[train_idx], axis=0)
            mean_va = np.nanmean(rows[val_idx], axis=0)
            for k in k_grid:
                model = FourierModel(k).fit(angle_deg, mean_tr)
                pred = model.predict(angle_deg)
                err = (pred - mean_va)[np.isfinite(mean_va)]
                cv_rmse[k].append(float(np.sqrt(np.mean(err ** 2))))
        means = pd.Series({k: float(np.mean(v)) for k, v in cv_rmse.items()})
        best_k = int(means.idxmin())
        return best_k, means


class ArimaPredictor:
    """SARIMAX на прореженной угловой сетке с сезонным периодом = оборот.

    Между двумя соседними оборотами ошибка почти повторяется, поэтому
    сезонная компонента (s = bins_per_rev) моделирует эту повторяемость.
    Конфигурация выбирается по AIC на обучающей выборке.
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
            raise RuntimeError('Все конфигурации SARIMAX не сошлись')
        order, sorder = self.best_config_
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.model_ = sm.tsa.SARIMAX(series, order=order, seasonal_order=sorder,
                                         trend='n').fit(disp=False, maxiter=150)
        return self

    def predict_curve(self, angle_native, n_revolutions: int) -> np.ndarray:
        """Прогноз на n_revolutions вперёд; возвращает кривую по углу (усреднение прогнозов)."""
        m = self.bins_per_rev
        forecast = np.asarray(self.model_.predict(start=len(self.model_.fittedvalues),
                                                  end=len(self.model_.fittedvalues) + n_revolutions * m - 1))
        per_rev = forecast[: n_revolutions * m].reshape(n_revolutions, m)
        curve_bins = per_rev.mean(axis=0)  # средний прогноз на оборот вперёд
        return periodic_interp(self.centers_, curve_bins, angle_native)


def make_tree_models(seed: int):
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
            early_stopping=True, validation_fraction=0.15,
            random_state=seed),
    }


def train_gru(angle_native, train_rows, seed=0, hidden=32, layers=1,
              harmonics=8, max_epochs=80, batch_size=8, patience=12,
              val_frac=0.15, lr=3e-3):
    """Двунаправленная GRU seq2seq: угол (гармоники) -> ошибка в каждом угле.

    Однослойная BiGRU — оптимальный компромисс для CPU (градиентный спуск
    по 1576 шагам BPTT дорог); на 2-ядерной машине эпоха ≈ 4–5 с.
    """
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = train_rows.shape[0]
    n_val = max(2, int(round(n * val_frac))) if n > 10 else 0
    val_rows = train_rows[n - n_val:] if n_val else None
    fit_rows = train_rows[: n - n_val] if n_val else train_rows

    X = harmonic_features(angle_native, harmonics=harmonics, include_linear=True)
    y_mean = float(np.nanmean(fit_rows))
    y_std = float(np.nanstd(fit_rows)) or 1.0

    Xt = torch.tensor(np.tile(X, (fit_rows.shape[0], 1, 1)), dtype=torch.float32)
    yt = torch.tensor(((np.nan_to_num(fit_rows) - y_mean) / y_std)[..., None],
                      dtype=torch.float32)

    class BiGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(X.shape[1], hidden, num_layers=layers,
                              batch_first=True, bidirectional=True,
                              dropout=0.1 if layers > 1 else 0.0)
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
    best_val, best_state, bad = np.inf, None, 0
    idx = np.arange(fit_rows.shape[0])
    for epoch in range(max_epochs):
        model.train()
        np.random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            b = idx[start:start + batch_size]
            opt.zero_grad()
            pred = model(Xt[b])
            loss = torch.mean((pred - yt[b]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if val_rows is not None:
            model.eval()
            with torch.no_grad():
                Xv = torch.tensor(X[None, ...], dtype=torch.float32)
                pred_v = model(Xv)[0, ..., 0].numpy() * y_std + y_mean
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
        Xv = torch.tensor(X[None, ...], dtype=torch.float32)
        curve = model(Xv)[0, ..., 0].numpy() * y_std + y_mean
    return curve, {'best_val_rmse_deg': float(best_val), 'epochs_ran': epoch + 1}


# --------------------------------------------------------------------------
# 4. Метрики
# --------------------------------------------------------------------------

def sample_metrics(y_true_rows: np.ndarray, pred_curve: np.ndarray) -> dict:
    """Метрики по отсчётам всех тестовых оборотов."""
    pred = np.broadcast_to(pred_curve, y_true_rows.shape)
    mask = np.isfinite(y_true_rows)
    err = (pred - y_true_rows)[mask]
    yt = y_true_rows[mask]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    maxabs = float(np.max(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return {'rmse_deg': rmse, 'mae_deg': mae, 'maxabs_deg': maxabs, 'r2': r2}


def profile_metrics(y_true_rows: np.ndarray, pred_curve: np.ndarray) -> dict:
    """Метрики по средней (по тестовым оборотам) кривой — практический интерес."""
    mean_true = np.nanmean(y_true_rows, axis=0)
    mask = np.isfinite(mean_true)
    err = (pred_curve - mean_true)[mask]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    maxabs = float(np.max(np.abs(err)))
    resid = (mean_true - pred_curve)[mask]
    return {
        'profile_rmse_deg': rmse,
        'profile_mae_deg': mae,
        'profile_maxabs_deg': maxabs,
        'p2p_before_deg': float(np.nanmax(mean_true) - np.nanmin(mean_true)),
        'p2p_after_deg': float(resid.max() - resid.min()),
        'std_after_deg': float(np.std(resid)),
    }


def noise_floor(train_rows: np.ndarray) -> float:
    """Среднее СКО между оборотами (шум эталона) — нижняя граница ошибки модели."""
    return float(np.sqrt(np.nanmean(np.nanvar(train_rows, axis=0, ddof=1))))


def summarize_metrics(y_true_rows, pred_curve, train_rows) -> dict:
    m = sample_metrics(y_true_rows, pred_curve)
    m.update(profile_metrics(y_true_rows, pred_curve))
    m['floor_deg'] = noise_floor(train_rows)
    return m


# --------------------------------------------------------------------------
# 5. Один эксперимент × один сигнал
# --------------------------------------------------------------------------

def evaluate_blocked(profile, seed=0, arima_bins=24, harmonics=8, skip_gru=False):
    """Основной протокол: блочное разбиение по оборотам 80/20."""
    angle = profile.angle_deg
    rows = profile.rows
    n_rev = rows.shape[0]
    n_train = max(4, int(math.floor(0.8 * n_rev)))
    train_rows, test_rows = rows[:n_train], rows[n_train:]

    results = []
    predictions = {}

    def record(model_name, pred_curve, train_time, extra=None):
        m = summarize_metrics(test_rows, pred_curve, train_rows)
        rec = {'model': model_name, 'protocol': 'blocked_80_20',
               'n_train_rev': int(n_train), 'n_test_rev': int(n_rev - n_train),
               'train_time_s': round(train_time, 3)}
        rec.update(m)
        if extra:
            rec.update(extra)
        results.append(rec)
        predictions[model_name] = pred_curve
        print(f'    {model_name}: RMSE(тест)={m["rmse_deg"] * ARCSEC:.2f}", '
              f'время={train_time:.1f} с', flush=True)
        return pred_curve

    # 0. Нулевой предиктор (без коррекции)
    t0 = time.time()
    record('Без коррекции (0)', np.zeros_like(angle), time.time() - t0)

    # 1. Фурье — два способа выбора числа гармоник (только по обучающим данным):
    #    (а) BIC — статистически обоснованное усечение;
    #    (б) групповая 5-fold CV по оборотам — предсказательный критерий.
    t0 = time.time()
    mean_train = np.nanmean(train_rows, axis=0)
    k_bic = FourierModel.select_harmonics_bic(angle, mean_train)
    fourier_bic = FourierModel(k_bic).fit(angle, mean_train)
    record(f'Фурье (МНК, K={k_bic} по BIC)', fourier_bic.predict(angle), time.time() - t0,
           {'harmonics': k_bic})

    t0 = time.time()
    k_cv, cv_curve = FourierModel.select_harmonics_cv(angle, train_rows, seed=seed)
    fourier_cv = FourierModel(k_cv).fit(angle, mean_train)
    record(f'Фурье (МНК, K={k_cv} по CV)', fourier_cv.predict(angle), time.time() - t0,
           {'harmonics': k_cv})

    # 2. ARIMA-семейство — SARIMAX, выбор конфигурации по AIC на обучении
    try:
        t0 = time.time()
        arima = ArimaPredictor(bins_per_rev=arima_bins).fit(angle, train_rows)
        pred = arima.predict_curve(angle, n_revolutions=test_rows.shape[0])
        cfg = arima.best_config_
        record(f'SARIMAX{cfg[0]}×({cfg[1][0]},{cfg[1][1]},{cfg[1][2]},s={cfg[1][3]})', pred,
               time.time() - t0, {'arima_aic': arima.best_aic_})
    except Exception as exc:
        results.append({'model': 'SARIMAX', 'protocol': 'blocked_80_20',
                        'error': f'{type(exc).__name__}: {exc}'})

    # 3. Деревья и GBDT
    X_train = np.tile(harmonic_features(angle, harmonics=harmonics), (n_train, 1))
    y_train = np.nan_to_num(train_rows).ravel()
    X_test_angle = harmonic_features(angle, harmonics=harmonics)
    for name, est in make_tree_models(seed).items():
        t0 = time.time()
        est.fit(X_train, y_train)
        pred = est.predict(X_test_angle)
        record(name, pred, time.time() - t0, {'harmonics': harmonics})

    # 4. GRU
    if not skip_gru:
        t0 = time.time()
        gru_curve, gru_info = train_gru(angle, train_rows, seed=seed, harmonics=harmonics)
        record('BiGRU (seq2seq, H=8)', gru_curve, time.time() - t0, gru_info)

    return results, angle, rows, (train_rows, test_rows), {'k_bic': k_bic,
                                                           'k_cv': k_cv,
                                                           'cv_curve_harmonics': cv_curve.to_dict(),
                                                           'predictions': predictions}


def evaluate_cv(profile, seed=0, harmonics=8, n_splits=5, skip_gru=True,
                skip_arima=False, arima_bins=24):
    """Контрольная оценка: 5-fold GroupKFold по оборотам."""
    angle = profile.angle_deg
    rows = profile.rows
    groups = np.arange(rows.shape[0])
    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics: dict[str, list] = {}

    for train_idx, test_idx in gkf.split(rows, None, groups):
        tr, te = rows[train_idx], rows[test_idx]
        mean_tr = np.nanmean(tr, axis=0)

        def add(name, curve):
            m = summarize_metrics(te, curve, tr)
            fold_metrics.setdefault(name, []).append(m)

        add('Без коррекции (0)', np.zeros_like(angle))
        k_bic = FourierModel.select_harmonics_bic(angle, mean_tr)
        add('Фурье (BIC)', FourierModel(k_bic).fit(angle, mean_tr).predict(angle))

        if not skip_arima:
            try:
                arima = ArimaPredictor(bins_per_rev=arima_bins).fit(angle, tr)
                add('SARIMAX', arima.predict_curve(angle, n_revolutions=te.shape[0]))
            except Exception:
                pass

        X_tr = np.tile(harmonic_features(angle, harmonics=harmonics), (tr.shape[0], 1))
        y_tr = np.nan_to_num(tr).ravel()
        X_all = harmonic_features(angle, harmonics=harmonics)
        # В CV — только быстрые модели: деревья/бустинг/RF уже оценены в блочном тесте

        cv_models = {
            'DecisionTree (depth=8)': DecisionTreeRegressor(
                max_depth=8, min_samples_leaf=100, random_state=seed),
            'HistGradientBoosting': HistGradientBoostingRegressor(
                max_iter=500, learning_rate=0.06, max_leaf_nodes=63,
                early_stopping=True, validation_fraction=0.15,
                random_state=seed),
        }
        for name, est in cv_models.items():
            est.fit(X_tr, y_tr)
            add(name, est.predict(X_all))

        if not skip_gru:
            curve, _ = train_gru(angle, tr, seed=seed, harmonics=harmonics)
            add('BiGRU (seq2seq, H=8)', curve)

    summary = []
    for name, lst in fold_metrics.items():
        rec = {'model': name, 'protocol': f'cv_groupkfold_{n_splits}', 'n_folds': len(lst)}
        keys = ['rmse_deg', 'mae_deg', 'maxabs_deg', 'r2', 'profile_rmse_deg',
                'p2p_after_deg', 'std_after_deg', 'floor_deg']
        for key in keys:
            vals = np.array([m[key] for m in lst], dtype=float)
            rec[f'{key}_mean'] = float(np.nanmean(vals))
            rec[f'{key}_std'] = float(np.nanstd(vals))
        summary.append(rec)
    return summary


# --------------------------------------------------------------------------
# 6. Графики
# --------------------------------------------------------------------------

def plot_comparison(exp_name, signal, outdir, angle, test_rows, predictions):

    mean_test = np.nanmean(test_rows, axis=0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    axes[0].plot(angle, mean_test, 'k', lw=1.6, label='Фактическая (тест, среднее)')
    for label, curve in predictions.items():
        if label == 'Без коррекции (0)':
            continue
        axes[0].plot(angle, curve, lw=1.1, alpha=0.85, label=label)
    axes[0].set_ylabel('Ошибка угла, град')
    axes[0].set_title(f'{exp_name} / {signal}: предсказание ошибки угла (тест)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.3)

    axes[1].plot(angle, mean_test, 'k', lw=1.6, label='Без коррекции')
    for label, curve in predictions.items():
        if label == 'Без коррекции (0)':
            continue
        axes[1].plot(angle, mean_test - curve, lw=1.1, alpha=0.85,
                     label=f'Остаток после: {label}')
    axes[1].axhline(0, color='black', lw=.5, alpha=.5)
    axes[1].set_xlabel('Угол оборота, град')
    axes[1].set_ylabel('Остаток, град')
    axes[1].grid(alpha=.3)
    axes[1].legend(fontsize=8)
    png = outdir / f'{exp_name}_{signal}_comparison.png'
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def plot_metric_bars(exp_name, signal, outdir, metrics_rows):

    names = [r['model'] for r in metrics_rows if 'rmse_deg' in r]
    vals = [r['profile_rmse_deg'] * ARCSEC for r in metrics_rows if 'rmse_deg' in r]
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.7 * len(names) + 1.5)))
    order = np.argsort(vals)
    ax.barh([names[i] for i in order], [vals[i] for i in order], color='steelblue')
    ax.set_xlabel('RMSE средней тестовой кривой, угл. сек')
    ax.set_title(f'{exp_name} / {signal}: RMSE на тесте (меньше — лучше)')
    ax.grid(axis='x', alpha=.3)
    fig.tight_layout()
    png = outdir / f'{exp_name}_{signal}_rmse.png'
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


# --------------------------------------------------------------------------
# 7. Оркестрация
# --------------------------------------------------------------------------

def run_experiment(exp_name, signals, outdir, seed=0, skip_gru=False,
                   skip_cv=False, arima_bins=24):
    outdir.mkdir(parents=True, exist_ok=True)
    exp_dir = outdir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 100)
    print(f'Эксперимент: {exp_name}')

    # Проверка качества сырых данных
    dq = data_quality_report(exp_name)
    print('Проверка качества данных:', json.dumps(dq, ensure_ascii=False))
    pd.DataFrame([dq]).to_csv(exp_dir / 'data_quality.csv', index=False)

    all_blocked, all_cv = [], []
    for signal in signals:
        print('-' * 100)
        print(f'Сигнал (таргет): {signal}')
        t0 = time.time()
        profile, csv_path = build_profile(exp_name, signal)
        angle, rows = profile.angle_deg, profile.rows
        print(f'  Оборотов={rows.shape[0]}, сетка={len(angle)}, '
              f'p2p средней кривой={np.nanmax(profile.mean) - np.nanmin(profile.mean):.6f} град '
              f'({(np.nanmax(profile.mean) - np.nanmin(profile.mean)) * ARCSEC:.1f}"), '
              f'межоборотный разброс={np.nanmean(profile.std) * ARCSEC:.2f}"')

        blocked, angle, rows, (train_rows, test_rows), info = evaluate_blocked(
            profile, seed=seed, arima_bins=arima_bins, skip_gru=skip_gru)
        for rec in blocked:
            rec['experiment'] = exp_name
            rec['signal'] = signal
        all_blocked.extend(blocked)

        n_train = train_rows.shape[0]
        test_rows = rows[n_train:]
        dfm = pd.DataFrame(blocked)
        dfm.to_csv(exp_dir / f'metrics_blocked_{signal}.csv', index=False)
        show_cols = [c for c in ['model', 'rmse_deg', 'maxabs_deg', 'r2',
                                 'profile_rmse_deg', 'p2p_after_deg'] if c in dfm.columns]
        print(dfm[show_cols].to_string(index=False))

        # Графики по предсказаниям, полученным в evaluate_blocked (без переобучения)
        preds = {}
        for rec_name, curve in info['predictions'].items():
            if ('по CV' in rec_name or 'RandomForest' in rec_name
                    or 'HistGradientBoosting' in rec_name or 'BiGRU' in rec_name
                    or 'SARIMAX' in rec_name):
                preds[rec_name] = curve
        plot_comparison(exp_name, signal, exp_dir, angle, test_rows, preds)
        plot_metric_bars(exp_name, signal, exp_dir, blocked)
        pd.Series(info['cv_curve_harmonics']).to_csv(
            exp_dir / f'fourier_cv_curve_{signal}.csv', header=['rmse_deg'])

        if not skip_cv:
            print('  5-fold GroupKFold по оборотам...')
            cv = evaluate_cv(profile, seed=seed, arima_bins=arima_bins, skip_gru=True)
            for rec in cv:
                rec['experiment'] = exp_name
                rec['signal'] = signal
            all_cv.extend(cv)
            pd.DataFrame(cv).to_csv(exp_dir / f'metrics_cv_{signal}.csv', index=False)
        print(f'  Время сигнала: {time.time() - t0:.1f} с')

    blocked_df = pd.DataFrame(all_blocked)
    blocked_df.to_csv(exp_dir / 'metrics_blocked_all.csv', index=False)
    if all_cv:
        pd.DataFrame(all_cv).to_csv(exp_dir / 'metrics_cv_all.csv', index=False)
    return blocked_df, all_cv


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exp', action='append', default=None,
                        help='Имя папки эксперимента (можно повторять); по умолчанию оба')
    parser.add_argument('--signals', nargs='+', default=TARGET_SIGNALS,
                        choices=['half_diff', 'circular_mean', 'ang_1', 'ang_2'])
    parser.add_argument('--output-dir', type=Path, default=OUTDIR)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--arima-bins', type=int, default=24,
                        help='Угловых бинов на оборот для SARIMAX (сезонный период)')
    parser.add_argument('--skip-gru', action='store_true')
    parser.add_argument('--skip-cv', action='store_true')
    args = parser.parse_args()

    experiments = args.exp or EXPERIMENTS
    t0 = time.time()
    for exp_name in experiments:
        run_experiment(exp_name, args.signals, args.output_dir, seed=args.seed,
                       skip_gru=args.skip_gru, skip_cv=args.skip_cv,
                       arima_bins=args.arima_bins)
    print('=' * 100)
    print(f'Готово за {time.time() - t0:.1f} с. Результаты в {args.output_dir}')


if __name__ == '__main__':
    main()
