#!/usr/bin/env python3
"""
生存分析基线模型

使用 Cox 比例风险模型预测时间到事件（进展/复发）

模型：
- CoxPHFitter (lifelines)

评估指标：
- C-index (Concordance Index)
- Log-rank test
- 显著性检验

用法:
    python scripts/run_survival.py \\
        --x work/feats/surv_X.csv \\
        --duration work/feats/duration.csv \\
        --event work/feats/event.csv \\
        --out work/models/coxph.pkl
"""

import sys
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# lifelines
try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    logging.error("lifelines 未安装，请运行: pip install lifelines")


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('run_survival.log')
        ]
    )


def load_data(X_path: Path, duration_path: Path, event_path: Path):
    """加载数据"""
    logger = logging.getLogger(__name__)

    logger.info(f"加载特征: {X_path}")
    X = pd.read_csv(X_path)

    logger.info(f"加载持续时间: {duration_path}")
    duration = pd.read_csv(duration_path)
    if isinstance(duration, pd.DataFrame):
        duration = duration.iloc[:, 0]

    logger.info(f"加载事件: {event_path}")
    event = pd.read_csv(event_path)
    if isinstance(event, pd.DataFrame):
        event = event.iloc[:, 0]

    logger.info(f"数据形状: X {X.shape}, duration {duration.shape}, event {event.shape}")
    logger.info(f"事件发生: {event.sum()} / {len(event)} ({event.mean() * 100:.1f}%)")
    logger.info(f"随访时长: {duration.mean():.1f} ± {duration.std():.1f} 天")

    # 检查缺失值
    if X.isnull().any().any():
        logger.warning(f"特征中有 {X.isnull().sum().sum()} 个缺失值")
        # 简单填充
        for col in X.columns:
            if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0], inplace=True)

    # 合并为生存数据框
    df = X.copy()
    df['duration'] = duration.values
    df['event'] = event.values

    return df


def train_cox_model(df: pd.DataFrame, duration_col: str = 'duration', event_col: str = 'event'):
    """训练 Cox 模型"""
    logger = logging.getLogger(__name__)

    if not HAS_LIFELINES:
        logger.error("lifelines 未安装")
        sys.exit(1)

    logger.info("训练 Cox 比例风险模型...")

    # 创建模型
    cph = CoxPHFitter(penalizer=0.01)  # 添加 L2 正则化

    # 训练
    try:
        cph.fit(df, duration_col=duration_col, event_col=event_col)
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        raise

    # 打印结果
    print("\n" + "=" * 70)
    print("Cox 比例风险模型摘要")
    print("=" * 70)
    cph.print_summary()
    print("=" * 70)

    # C-index
    c_index = cph.concordance_index_
    print(f"\nC-index (Concordance Index): {c_index:.4f}")

    # 显著特征
    print("\n显著特征 (p < 0.05):")
    summary = cph.summary
    significant = summary[summary['p'] < 0.05]
    if len(significant) > 0:
        print(significant[['coef', 'exp(coef)', 'p']].to_string())
    else:
        print("  (无显著特征)")

    # 风险比
    print("\n风险比 (Hazard Ratios, Top 10):")
    hr = cph.hazard_ratios_.sort_values(ascending=False)
    print(hr.head(10).to_string())

    return cph


def evaluate_cox_model(cph: CoxPHFitter, df: pd.DataFrame, duration_col: str = 'duration', event_col: str = 'event'):
    """评估 Cox 模型"""
    logger = logging.getLogger(__name__)

    # 预测风险分数
    X = df.drop([duration_col, event_col], axis=1)
    risk_scores = cph.predict_partial_hazard(X)

    # 重新计算 C-index
    c_index = concordance_index(df[duration_col], -risk_scores, df[event_col])

    print(f"\n验证 C-index: {c_index:.4f}")

    # 分层分析（高风险 vs 低风险）
    median_risk = risk_scores.median()
    high_risk = risk_scores >= median_risk

    print(f"\n风险分层:")
    print(f"  高风险组: {high_risk.sum()} 人, 事件 {df[high_risk][event_col].sum()} 例")
    print(f"  低风险组: {(~high_risk).sum()} 人, 事件 {df[~high_risk][event_col].sum()} 例")

    # Log-rank test（如果有 lifelines）
    try:
        from lifelines.statistics import logrank_test

        results = logrank_test(
            df[high_risk][duration_col],
            df[~high_risk][duration_col],
            df[high_risk][event_col],
            df[~high_risk][event_col]
        )

        print(f"\nLog-rank 检验 (高风险 vs 低风险):")
        print(f"  统计量: {results.test_statistic:.4f}")
        print(f"  p 值:   {results.p_value:.4f}")

        if results.p_value < 0.05:
            print("  结论:   两组生存曲线有显著差异 (p < 0.05)")
        else:
            print("  结论:   两组生存曲线无显著差异 (p >= 0.05)")

    except Exception as e:
        logger.warning(f"Log-rank 检验失败: {e}")

    return c_index


def main():
    parser = argparse.ArgumentParser(
        description="生存分析基线模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_survival.py \\
      --x work/feats/surv_X.csv \\
      --duration work/feats/duration.csv \\
      --event work/feats/event.csv \\
      --out work/models/coxph.pkl
        """
    )

    # 必需参数
    parser.add_argument("--x", type=Path, required=True, help="特征数据 CSV")
    parser.add_argument("--duration", type=Path, required=True, help="持续时间 CSV")
    parser.add_argument("--event", type=Path, required=True, help="事件 CSV")
    parser.add_argument("--out", type=Path, required=True, help="输出模型文件 (PKL)")

    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")

    args = parser.parse_args()

    # 配置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    print("\n" + "=" * 70)
    print("生存分析基线模型 (Cox PH)")
    print("=" * 70)
    print(f"特征:       {args.x}")
    print(f"持续时间:   {args.duration}")
    print(f"事件:       {args.event}")
    print("=" * 70 + "\n")

    # 加载数据
    df = load_data(args.x, args.duration, args.event)

    # 训练模型
    cph = train_cox_model(df)

    # 评估模型
    c_index = evaluate_cox_model(cph, df)

    # 保存模型
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(cph, f)
    logger.info(f"模型已保存: {args.out}")

    # 保存摘要
    summary_path = args.out.with_suffix('.summary.csv')
    cph.summary.to_csv(summary_path)
    logger.info(f"模型摘要已保存: {summary_path}")

    # 保存结果
    results_path = args.out.with_suffix('.results.json')
    import json
    results = {
        'c_index': float(c_index),
        'num_samples': len(df),
        'num_events': int(df['event'].sum()),
        'significant_features': cph.summary[cph.summary['p'] < 0.05].index.tolist()
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"结果已保存: {results_path}")

    print("\n" + "=" * 70)
    print("生存分析模型训练完成！")
    print("=" * 70)
    print(f"模型文件:   {args.out}")
    print(f"摘要文件:   {summary_path}")
    print(f"结果文件:   {results_path}")
    print(f"C-index:    {c_index:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
