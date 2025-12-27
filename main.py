"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

import os
import random
import numpy as np
import pandas as pd

import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import OrdinalEncoder


# SEED
SEED = 322
random.seed(SEED)
np.random.seed(SEED)



def calibrate_width(y05, y95, alpha):
    center = (y05 + y95) / 2
    width = y95 - y05
    width = np.maximum(width, 1e-6)
    width = width * alpha
    return center - width / 2, center + width / 2


def create_submission(predictions):
    """
    Создание submission.csv на основе sample_submission.csv
    """

    os.makedirs("results", exist_ok=True)

    sample_submission = pd.read_csv("data/sample_submission.csv")

    sample_submission["price_p05"] = predictions["price_p05"]
    sample_submission["price_p95"] = predictions["price_p95"]

    submission_path = "results/submission.csv"
    sample_submission.to_csv(submission_path, index=False)

    print(f"Submission файл сохранен: {submission_path}")
    return submission_path




def main():
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)


    # Загрузка данных

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    train["dt"] = pd.to_datetime(train["dt"])
    train = train.sort_values("dt")

    test["dt"] = pd.to_datetime(test["dt"])
    test = test.drop(columns=["row_id"])


    # Признаки
    NUM_FEATURES = [
        "dow", "day_of_month", "week_of_year", "month",
        "n_stores", "holiday_flag", "activity_flag",
        "precpt", "avg_temperature", "avg_humidity", "avg_wind_level"
    ]

    CAT_FEATURES = [
        "product_id", "management_group_id",
        "first_category_id", "second_category_id", "third_category_id"
    ]

    FEATURES = NUM_FEATURES + CAT_FEATURES

    for col in CAT_FEATURES:
        train[col] = train[col].astype("category")
        test[col] = test[col].astype("category")

    X = train[FEATURES]
    y_p05 = train["price_p05"]
    y_p95 = train["price_p95"]

    # Time split 
    split_date = train["dt"].quantile(0.8)

    train_idx = train["dt"] <= split_date
    val_idx = train["dt"] > split_date

    X_train = X[train_idx]
    X_val = X[val_idx]

    y05_train = y_p05[train_idx]
    y05_val = y_p05[val_idx]

    y95_train = y_p95[train_idx]
    y95_val = y_p95[val_idx]

    # OrdinalEncoder 
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    X_train_enc = X_train.copy()
    X_train_enc[CAT_FEATURES] = encoder.fit_transform(X_train[CAT_FEATURES])

    X_val_enc = X_val.copy()
    X_val_enc[CAT_FEATURES] = encoder.fit_transform(X_val[CAT_FEATURES])

    test_enc = test.copy()
    test_enc[CAT_FEATURES] = encoder.transform(test[CAT_FEATURES])

    # LightGBM
    y05_train_lgb = np.log1p(y05_train)
    y95_train_lgb = np.log1p(y95_train)

    lgb_params = {
        "objective": "quantile",
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
    }

    params_p05 = lgb_params.copy()
    params_p05.update({
        "alpha": 0.05,
        "learning_rate": 0.15,
        "num_leaves": 190,
        "min_data_in_leaf": 5,
    })

    model_lgb_p05 = lgb.LGBMRegressor(
        **params_p05,
        n_estimators=200,
        random_state=SEED
    )

    model_lgb_p05.fit(
        X_train,
        y05_train_lgb,
        eval_set=[(X_val, y05_val)],
        eval_metric="quantile",
        categorical_feature=CAT_FEATURES
    )

    params_p95 = lgb_params.copy()
    params_p95.update({
        "alpha": 0.95,
        "learning_rate": 0.001,
        "num_leaves": 10,
        "min_data_in_leaf": 10,
    })

    model_lgb_p95 = lgb.LGBMRegressor(
        **params_p95,
        n_estimators=200,
        random_state=SEED
    )

    model_lgb_p95.fit(
        X_train,
        y95_train_lgb,
        eval_set=[(X_val, y95_val)],
        eval_metric="quantile",
        categorical_feature=CAT_FEATURES
    )

    # XGBoost
    xgb_base_params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
        "random_state": SEED,
    }

    params_p05 = xgb_base_params.copy()
    params_p05.update({
        "learning_rate": 0.005,
        "max_depth": 6,
        "min_child_weight": 70,
        "reg_lambda": 10,
    })

    model_xgb_p05 = XGBRegressor(**params_p05, n_estimators=1500)

    model_xgb_p05.fit(
        X_train_enc,
        y05_train,
        eval_set=[(X_val_enc, y05_val)],
        verbose=False
    )

    params_p95 = xgb_base_params.copy()
    params_p95.update({
        "learning_rate": 0.001,
        "max_depth": 25,
        "min_child_weight": 10,
        "reg_lambda": 2,
    })

    model_xgb_p95 = XGBRegressor(**params_p95, n_estimators=1500)

    model_xgb_p95.fit(
        X_train_enc,
        y95_train,
        eval_set=[(X_val_enc, y95_val)],
        verbose=False
    )

    # CatBoost
    cat_base_params = {
        "loss_function": "Quantile",
        "learning_rate": 0.02,
        "depth": 8,
        "l2_leaf_reg": 10,
        "subsample": 0.8,
        "rsm": 0.8,
        "random_seed": SEED,
        "verbose": False,
        "allow_writing_files": False
    }

    params_p05 = cat_base_params.copy()
    params_p05["loss_function"] = "Quantile:alpha=0.05"

    model_cat_p05 = CatBoostRegressor(
        **params_p05,
        iterations=500
    )

    model_cat_p05.fit(
        X_train,
        y05_train,
        eval_set=(X_val, y05_val),
        cat_features=CAT_FEATURES
    )

    params_p95 = cat_base_params.copy()
    params_p95.update({
        "loss_function": "Quantile:alpha=0.95",
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 50,
        "subsample": 0.7
    })

    model_cat_p95 = CatBoostRegressor(
        **params_p95,
        iterations=500
    )

    model_cat_p95.fit(
        X_train,
        y95_train,
        eval_set=(X_val, y95_val),
        cat_features=CAT_FEATURES
    )

    # Предсказания + ансамбль
    p05_lgb = np.expm1(model_lgb_p05.predict(test[FEATURES]))
    p95_lgb = np.expm1(model_lgb_p95.predict(test[FEATURES]))

    p05_xgb = model_xgb_p05.predict(test_enc[FEATURES])
    p95_xgb = model_xgb_p95.predict(test_enc[FEATURES])

    p05_cat = model_cat_p05.predict(test[FEATURES])
    p95_cat = model_cat_p95.predict(test[FEATURES])

    p05_ens = (p05_lgb + p05_xgb + p05_cat) / 3
    p95_ens = (p95_lgb + p95_xgb + p95_cat) / 3

    p05_final, p95_final = calibrate_width(
        np.minimum(p05_ens, p95_ens),
        np.maximum(p05_ens, p95_ens),
        alpha=0.725
    )

    predictions = {
        "price_p05": p05_final,
        "price_p95": p95_final
    }

    # ОБЯЗАТЕЛЬНО
    create_submission(predictions)

    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()

