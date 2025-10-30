#!/usr/bin/env python3
"""
横断面二分类基线模型

预测任务：下一次随访肿瘤是否显著增大

模型：
- XGBoost
- RandomForest (备选)

评估指标：
- AUC-ROC
- Accuracy
- PR-AUC
- 5-Fold CV

用法:
    python scripts/run_classify.py \\
        --data work/feats/tabular_X.csv \\
        --labels work/feats/y.csv \\
        --model xgb \\
        --out work/models/xgb.pkl
"""

import sys
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_curve,
    auc, confusion_matrix, classification_report
)

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logging.warning("XGBoost 未安装，将使用 RandomForest")


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('run_classify.log')
        ]
    )


def load_data(X_path: Path, y_path: Path):
    """加载数据"""
    logger = logging.getLogger(__name__)

    logger.info(f"加载特征: {X_path}")
    X = pd.read_csv(X_path)

    logger.info(f"加载标签: {y_path}")
    y = pd.read_csv(y_path)

    # 如果 y 是 DataFrame，提取第一列
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    logger.info(f"数据形状: X {X.shape}, y {y.shape}")
    logger.info(f"标签分布: {y.value_counts().to_dict()}")

    # 检查缺失值
    if X.isnull().any().any():
        logger.warning(f"特征中有 {X.isnull().sum().sum()} 个缺失值")
        # 简单填充：数值型用中位数，分类型用众数
        for col in X.columns:
            if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0], inplace=True)

    return X, y


def create_pipeline(model_name: str, **kwargs):
    """创建模型管线"""
    logger = logging.getLogger(__name__)

    if model_name == 'xgb':
        if not HAS_XGBOOST:
            logger.error("XGBoost 未安装，请使用 'rf' 或安装 XGBoost")
            sys.exit(1)

        model = xgb.XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 5),
            learning_rate=kwargs.get('learning_rate', 0.1),
            subsample=kwargs.get('subsample', 0.8),
            colsample_bytree=kwargs.get('colsample_bytree', 0.8),
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        logger.info(f"创建 XGBoost 模型: n_estimators={model.n_estimators}, max_depth={model.max_depth}")

    elif model_name == 'rf':
        model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            random_state=42,
            n_jobs=-1
        )
        logger.info(f"创建 RandomForest 模型: n_estimators={model.n_estimators}, max_depth={model.max_depth}")

    else:
        raise ValueError(f"不支持的模型: {model_name}")

    # 创建 Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    return pipeline


def evaluate_model(pipeline, X, y, cv_folds: int = 5):
    """评估模型"""
    logger = logging.getLogger(__name__)

    logger.info(f"开始 {cv_folds}-Fold 交叉验证...")

    # 定义评分指标
    scoring = {
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }

    # 交叉验证
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    # 打印结果
    print("\n" + "=" * 70)
    print(f"{cv_folds}-Fold 交叉验证结果")
    print("=" * 70)

    for metric in scoring.keys():
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']

        print(f"\n{metric.upper()}:")
        print(f"  训练集: {train_scores.mean():.4f} ± {train_scores.std():.4f}")
        print(f"  测试集: {test_scores.mean():.4f} ± {test_scores.std():.4f}")

    # 计算 PR-AUC（手动）
    pr_aucs = []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)

    print(f"\nPR-AUC:")
    print(f"  测试集: {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")

    print("=" * 70 + "\n")

    # 在全部数据上训练最终模型
    logger.info("在全部数据上训练最终模型...")
    pipeline.fit(X, y)

    # 最终模型性能
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    print("最终模型性能（全部数据）:")
    print(f"  Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y, y_proba):.4f}")

    print("\n混淆矩阵:")
    print(confusion_matrix(y, y_pred))

    print("\n分类报告:")
    print(classification_report(y, y_pred))

    # 特征重要性
    if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
        importances = pipeline.named_steps['classifier'].feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\n特征重要性 (Top 10):")
        print(feature_importance.head(10).to_string(index=False))

    return pipeline, cv_results


def main():
    parser = argparse.ArgumentParser(
        description="横断面分类基线模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # XGBoost
  python scripts/run_classify.py --data work/feats/tabular_X.csv \\
      --labels work/feats/y.csv --model xgb --out work/models/xgb.pkl

  # RandomForest
  python scripts/run_classify.py --data work/feats/tabular_X.csv \\
      --labels work/feats/y.csv --model rf --out work/models/rf.pkl

  # 自定义参数
  python scripts/run_classify.py --data work/feats/tabular_X.csv \\
      --labels work/feats/y.csv --model xgb \\
      --n-estimators 200 --max-depth 7 --out work/models/xgb.pkl
        """
    )

    # 必需参数
    parser.add_argument("--data", type=Path, required=True, help="特征数据 CSV")
    parser.add_argument("--labels", type=Path, required=True, help="标签数据 CSV")
    parser.add_argument("--out", type=Path, required=True, help="输出模型文件 (PKL)")

    # 模型参数
    parser.add_argument("--model", choices=['xgb', 'rf'], default='xgb',
                       help="模型类型 (默认: xgb)")
    parser.add_argument("--n-estimators", type=int, default=100,
                       help="树的数量 (默认: 100)")
    parser.add_argument("--max-depth", type=int, default=5,
                       help="最大深度 (默认: 5)")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                       help="学习率 (仅 XGBoost, 默认: 0.1)")

    # 评估参数
    parser.add_argument("--cv-folds", type=int, default=5,
                       help="交叉验证折数 (默认: 5)")

    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")

    args = parser.parse_args()

    # 配置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    print("\n" + "=" * 70)
    print("横断面分类基线模型")
    print("=" * 70)
    print(f"模型:       {args.model.upper()}")
    print(f"数据:       {args.data}")
    print(f"标签:       {args.labels}")
    print("=" * 70 + "\n")

    # 加载数据
    X, y = load_data(args.data, args.labels)

    # 创建模型
    pipeline = create_pipeline(
        args.model,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )

    # 评估模型
    pipeline, cv_results = evaluate_model(pipeline, X, y, cv_folds=args.cv_folds)

    # 保存模型
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(pipeline, f)
    logger.info(f"模型已保存: {args.out}")

    # 保存交叉验证结果
    results_path = args.out.with_suffix('.results.json')
    import json
    results_dict = {k: [float(v) for v in vals] for k, vals in cv_results.items()}
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"交叉验证结果已保存: {results_path}")

    print("\n" + "=" * 70)
    print("分类模型训练完成！")
    print("=" * 70)
    print(f"模型文件:   {args.out}")
    print(f"结果文件:   {results_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
