"""
数据融合模块

将几何度量（从 timeline.json）、人口统计学数据和化验数据融合为统一的特征矩阵。

输出：
1. 横断面分析：tabular_X.csv, y.csv（预测下一次是否体积增大）
2. 生存分析：surv_X.csv, duration.csv, event.csv（时间到进展/复发）
"""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class DataJoiner:
    """数据融合器"""

    def __init__(self):
        self.timeline_df: Optional[pd.DataFrame] = None
        self.demographics_df: Optional[pd.DataFrame] = None
        self.labtests_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None

    def load_timeline(self, timeline_path: Path) -> pd.DataFrame:
        """
        加载 timeline.json 并转换为 DataFrame

        Args:
            timeline_path: timeline.json 路径

        Returns:
            DataFrame，每行代表一次就诊
        """
        logger.info(f"加载 timeline: {timeline_path}")

        if not timeline_path.exists():
            raise FileNotFoundError(f"文件不存在: {timeline_path}")

        with open(timeline_path, 'r') as f:
            timeline = json.load(f)

        # 展开为行
        rows = []
        for patient in timeline:
            patient_id = patient['patient_id']
            for visit in patient['visits']:
                row = {
                    'patient_id': patient_id,
                    'visit': visit['visit'],
                    'visit_date': visit['date'],
                    'vol_cc': visit.get('vol_cc', np.nan),
                    'surface_area_cm2': visit.get('surface_area_cm2', np.nan),
                    'num_points': visit.get('num_points', np.nan),
                }
                rows.append(row)

        df = pd.DataFrame(rows)

        # 解析日期
        df['visit_date'] = pd.to_datetime(df['visit_date'])

        # 排序
        df = df.sort_values(['patient_id', 'visit_date']).reset_index(drop=True)

        logger.info(f"Timeline 数据: {len(df)} 条记录, {df['patient_id'].nunique()} 个患者")

        self.timeline_df = df
        return df

    def load_demographics(self, demographics_path: Path) -> pd.DataFrame:
        """加载人口统计学数据"""
        logger.info(f"加载人口统计学数据: {demographics_path}")

        if not demographics_path.exists():
            logger.warning(f"文件不存在: {demographics_path}")
            return pd.DataFrame()

        df = pd.read_csv(demographics_path)

        # 确保 patient_id 存在
        if 'patient_id' not in df.columns:
            raise ValueError("人口统计学数据缺少 'patient_id' 列")

        # 解析日期（如果存在）
        if 'visit_date' in df.columns:
            df['visit_date'] = pd.to_datetime(df['visit_date'])

        logger.info(f"人口统计学数据: {len(df)} 条记录")

        self.demographics_df = df
        return df

    def load_labtests(self, labtests_path: Path) -> pd.DataFrame:
        """加载化验数据"""
        logger.info(f"加载化验数据: {labtests_path}")

        if not labtests_path.exists():
            logger.warning(f"文件不存在: {labtests_path}")
            return pd.DataFrame()

        df = pd.read_csv(labtests_path)

        # 确保 patient_id 存在
        if 'patient_id' not in df.columns:
            raise ValueError("化验数据缺少 'patient_id' 列")

        # 解析日期
        if 'visit_date' in df.columns:
            df['visit_date'] = pd.to_datetime(df['visit_date'])

        logger.info(f"化验数据: {len(df)} 条记录")

        self.labtests_df = df
        return df

    def merge_data(
        self,
        timeline_df: Optional[pd.DataFrame] = None,
        demographics_df: Optional[pd.DataFrame] = None,
        labtests_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        融合所有数据源

        Args:
            timeline_df: timeline DataFrame（可选，使用已加载的）
            demographics_df: 人口统计学 DataFrame（可选，使用已加载的）
            labtests_df: 化验 DataFrame（可选，使用已加载的）

        Returns:
            融合后的 DataFrame
        """
        logger.info("融合数据...")

        # 使用已加载的数据
        if timeline_df is None:
            timeline_df = self.timeline_df
        if demographics_df is None:
            demographics_df = self.demographics_df
        if labtests_df is None:
            labtests_df = self.labtests_df

        if timeline_df is None or timeline_df.empty:
            raise ValueError("Timeline 数据未加载或为空")

        # 从 timeline 开始
        merged = timeline_df.copy()

        # 合并人口统计学数据
        if demographics_df is not None and not demographics_df.empty:
            # 如果人口统计学数据有 visit_date，按 (patient_id, visit_date) 合并
            if 'visit_date' in demographics_df.columns:
                merged = pd.merge(
                    merged,
                    demographics_df,
                    on=['patient_id', 'visit_date'],
                    how='left'
                )
            else:
                # 否则只按 patient_id 合并（每个患者只有一条记录）
                merged = pd.merge(
                    merged,
                    demographics_df,
                    on='patient_id',
                    how='left'
                )
            logger.info("已合并人口统计学数据")

        # 合并化验数据
        if labtests_df is not None and not labtests_df.empty:
            if 'visit_date' in labtests_df.columns:
                # 按最近的化验日期合并
                merged = pd.merge_asof(
                    merged.sort_values('visit_date'),
                    labtests_df.sort_values('visit_date'),
                    on='visit_date',
                    by='patient_id',
                    direction='nearest',
                    tolerance=pd.Timedelta(days=30)  # 30天内的化验
                )
            else:
                merged = pd.merge(
                    merged,
                    labtests_df,
                    on='patient_id',
                    how='left'
                )
            logger.info("已合并化验数据")

        logger.info(f"融合完成: {len(merged)} 条记录, {merged.shape[1]} 列")

        self.merged_df = merged
        return merged

    def create_classification_dataset(
        self,
        merged_df: Optional[pd.DataFrame] = None,
        volume_increase_threshold: float = 0.2,  # 20% 增长
        output_dir: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        创建横断面分类数据集

        预测任务：下一次随访肿瘤是否显著增大（体积增长 >= 阈值）

        Args:
            merged_df: 融合后的 DataFrame（可选，使用已合并的）
            volume_increase_threshold: 体积增长阈值（相对比例）
            output_dir: 输出目录（可选）

        Returns:
            (X, y) 特征矩阵和标签
        """
        logger.info(f"创建分类数据集（体积增长阈值={volume_increase_threshold}）...")

        if merged_df is None:
            merged_df = self.merged_df

        if merged_df is None or merged_df.empty:
            raise ValueError("融合数据未准备好")

        # 计算下一次随访的体积增长
        merged_df = merged_df.sort_values(['patient_id', 'visit_date']).reset_index(drop=True)
        merged_df['vol_cc_next'] = merged_df.groupby('patient_id')['vol_cc'].shift(-1)

        # 计算相对增长
        merged_df['vol_growth_ratio'] = (
            (merged_df['vol_cc_next'] - merged_df['vol_cc']) / merged_df['vol_cc']
        )

        # 创建标签：体积增长 >= 阈值
        merged_df['y'] = (merged_df['vol_growth_ratio'] >= volume_increase_threshold).astype(int)

        # 移除没有下一次随访的记录（每个患者的最后一次）
        dataset = merged_df.dropna(subset=['vol_cc_next']).copy()

        logger.info(f"分类数据集: {len(dataset)} 个样本")

        # 选择特征列
        feature_cols = []

        # 几何特征
        geom_cols = ['vol_cc', 'surface_area_cm2', 'num_points']
        feature_cols.extend([c for c in geom_cols if c in dataset.columns])

        # 人口统计学特征
        demo_cols = ['age', 'sex', 'race', 'bmi', 'smoking', 'diabetes', 'hypertension']
        feature_cols.extend([c for c in demo_cols if c in dataset.columns])

        # 化验特征
        lab_cols = ['a1c', 'egfr', 'wbc', 'rbc', 'hemoglobin', 'platelets',
                   'creatinine', 'bun', 'glucose', 'ca125', 'ca199']
        feature_cols.extend([c for c in lab_cols if c in dataset.columns])

        # 提取特征和标签
        X = dataset[feature_cols].copy()
        y = dataset['y']

        # 处理分类变量
        if 'sex' in X.columns:
            X['sex'] = X['sex'].map({'M': 1, 'F': 0})

        logger.info(f"特征列 ({len(feature_cols)}): {feature_cols}")
        logger.info(f"标签分布: {y.value_counts().to_dict()}")

        # 保存（可选）
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            X.to_csv(output_dir / 'tabular_X.csv', index=False)
            y.to_csv(output_dir / 'y.csv', index=False, header=True)
            logger.info(f"保存分类数据集: {output_dir}")

        return X, y

    def create_survival_dataset(
        self,
        merged_df: Optional[pd.DataFrame] = None,
        event_column: Optional[str] = None,
        output_dir: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        创建生存分析数据集

        Args:
            merged_df: 融合后的 DataFrame（可选，使用已合并的）
            event_column: 事件列名（可选，自动检测或创建占位）
            output_dir: 输出目录（可选）

        Returns:
            (X, duration, event) 特征矩阵、持续时间、事件
        """
        logger.info("创建生存分析数据集...")

        if merged_df is None:
            merged_df = self.merged_df

        if merged_df is None or merged_df.empty:
            raise ValueError("融合数据未准备好")

        # 每个患者只保留第一次随访作为基线
        baseline = merged_df.sort_values(['patient_id', 'visit_date']).groupby('patient_id').first().reset_index()

        # 计算随访时长（从第一次到最后一次，单位：天）
        last_visit = merged_df.groupby('patient_id')['visit_date'].max().reset_index()
        last_visit.columns = ['patient_id', 'last_visit_date']
        baseline = baseline.merge(last_visit, on='patient_id')
        baseline['duration_days'] = (baseline['last_visit_date'] - baseline['visit_date']).dt.days

        # 事件：如果有指定的事件列，使用它；否则创建占位
        if event_column and event_column in baseline.columns:
            baseline['event'] = baseline[event_column]
        else:
            # 占位：假设体积增长超过 50% 为事件
            last_vol = merged_df.groupby('patient_id')['vol_cc'].last().reset_index()
            last_vol.columns = ['patient_id', 'last_vol_cc']
            baseline = baseline.merge(last_vol, on='patient_id')
            baseline['event'] = (
                (baseline['last_vol_cc'] - baseline['vol_cc']) / baseline['vol_cc'] >= 0.5
            ).astype(int)

        logger.info(f"生存数据集: {len(baseline)} 个患者")

        # 选择特征列（与分类任务类似）
        feature_cols = []

        # 几何特征
        geom_cols = ['vol_cc', 'surface_area_cm2', 'num_points']
        feature_cols.extend([c for c in geom_cols if c in baseline.columns])

        # 人口统计学特征
        demo_cols = ['age', 'sex', 'race', 'bmi', 'smoking', 'diabetes', 'hypertension']
        feature_cols.extend([c for c in demo_cols if c in baseline.columns])

        # 化验特征
        lab_cols = ['a1c', 'egfr', 'wbc', 'rbc', 'hemoglobin', 'platelets',
                   'creatinine', 'bun', 'glucose', 'ca125', 'ca199']
        feature_cols.extend([c for c in lab_cols if c in baseline.columns])

        # 提取特征、持续时间、事件
        X = baseline[feature_cols].copy()
        duration = baseline['duration_days']
        event = baseline['event']

        # 处理分类变量
        if 'sex' in X.columns:
            X['sex'] = X['sex'].map({'M': 1, 'F': 0})

        logger.info(f"特征列 ({len(feature_cols)}): {feature_cols}")
        logger.info(f"事件发生: {event.sum()} / {len(event)} ({event.mean() * 100:.1f}%)")
        logger.info(f"随访时长: {duration.mean():.1f} ± {duration.std():.1f} 天")

        # 保存（可选）
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            X.to_csv(output_dir / 'surv_X.csv', index=False)
            duration.to_csv(output_dir / 'duration.csv', index=False, header=True)
            event.to_csv(output_dir / 'event.csv', index=False, header=True)
            logger.info(f"保存生存数据集: {output_dir}")

        return X, duration, event


def main():
    """测试数据融合"""
    import argparse

    parser = argparse.ArgumentParser(description="数据融合测试")
    parser.add_argument("--timeline", type=Path, required=True, help="timeline.json 路径")
    parser.add_argument("--demographics", type=Path, help="人口统计学数据")
    parser.add_argument("--labtests", type=Path, help="化验数据")
    parser.add_argument("--output", type=Path, default=Path("work/feats"),
                       help="输出目录")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建 joiner
    joiner = DataJoiner()

    # 加载数据
    joiner.load_timeline(args.timeline)

    if args.demographics:
        joiner.load_demographics(args.demographics)

    if args.labtests:
        joiner.load_labtests(args.labtests)

    # 融合
    merged = joiner.merge_data()

    print("\n融合数据预览:")
    print(merged.head())
    print(f"\n形状: {merged.shape}")

    # 创建分类数据集
    print("\n创建分类数据集...")
    X_cls, y_cls = joiner.create_classification_dataset(output_dir=args.output)
    print(f"分类数据: X {X_cls.shape}, y {y_cls.shape}")

    # 创建生存数据集
    print("\n创建生存数据集...")
    X_surv, duration, event = joiner.create_survival_dataset(output_dir=args.output)
    print(f"生存数据: X {X_surv.shape}, duration {duration.shape}, event {event.shape}")


if __name__ == "__main__":
    main()
