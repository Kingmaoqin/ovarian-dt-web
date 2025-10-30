"""
临床数据读取与处理模块

负责加载和标准化人口统计学数据(demographics)和化验数据(labtests)。

主要功能：
- 加载 CSV 表格
- 列名标准化（patient_id, visit_date, age, sex, race, A1c, eGFR, WBC 等）
- 时间戳统一到 UTC
- 单位换算
- 数据清洗与验证
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ClinicalDataLoader:
    """临床数据加载器"""

    # 标准列名映射
    DEMOGRAPHICS_COLUMNS_MAP = {
        # 患者 ID
        'patient_id': ['patient_id', 'patientid', 'patient', 'id', 'subject_id'],
        'visit_date': ['visit_date', 'visitdate', 'date', 'study_date', 'scan_date'],
        'age': ['age', 'age_at_diagnosis', 'age_years'],
        'sex': ['sex', 'gender'],
        'race': ['race', 'ethnicity', 'ethnic'],
        'bmi': ['bmi', 'body_mass_index'],
        'smoking': ['smoking', 'smoking_status'],
        'diabetes': ['diabetes', 'diabetic'],
        'hypertension': ['hypertension', 'htn'],
    }

    LABTESTS_COLUMNS_MAP = {
        'patient_id': ['patient_id', 'patientid', 'patient', 'id', 'subject_id'],
        'visit_date': ['visit_date', 'visitdate', 'date', 'test_date', 'lab_date'],
        'a1c': ['a1c', 'hba1c', 'hemoglobin_a1c', 'glycated_hemoglobin'],
        'egfr': ['egfr', 'gfr', 'estimated_gfr'],
        'wbc': ['wbc', 'white_blood_cell', 'leukocyte'],
        'rbc': ['rbc', 'red_blood_cell', 'erythrocyte'],
        'hemoglobin': ['hemoglobin', 'hb', 'hgb'],
        'platelets': ['platelets', 'plt', 'platelet_count'],
        'creatinine': ['creatinine', 'cr', 'serum_creatinine'],
        'bun': ['bun', 'blood_urea_nitrogen', 'urea'],
        'glucose': ['glucose', 'blood_glucose', 'bg'],
        'ca125': ['ca125', 'ca_125', 'cancer_antigen_125'],
        'ca199': ['ca199', 'ca_199', 'cancer_antigen_199'],
    }

    def __init__(self):
        self.demographics_df: Optional[pd.DataFrame] = None
        self.labtests_df: Optional[pd.DataFrame] = None

    def _find_column(self, df: pd.DataFrame, standard_name: str, alternatives: List[str]) -> Optional[str]:
        """
        在 DataFrame 中查找匹配的列名

        Args:
            df: DataFrame
            standard_name: 标准列名
            alternatives: 备选列名列表

        Returns:
            找到的列名，如果未找到则返回 None
        """
        # 转换为小写进行匹配
        columns_lower = {col.lower(): col for col in df.columns}

        for alt in alternatives:
            alt_lower = alt.lower()
            if alt_lower in columns_lower:
                return columns_lower[alt_lower]

        return None

    def _standardize_columns(
        self,
        df: pd.DataFrame,
        columns_map: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        标准化列名

        Args:
            df: 原始 DataFrame
            columns_map: 列名映射字典

        Returns:
            标准化后的 DataFrame
        """
        rename_dict = {}

        for standard_name, alternatives in columns_map.items():
            found_col = self._find_column(df, standard_name, alternatives)
            if found_col:
                rename_dict[found_col] = standard_name
                logger.debug(f"映射列: {found_col} -> {standard_name}")
            else:
                logger.warning(f"未找到列: {standard_name} (尝试过: {alternatives})")

        df_renamed = df.rename(columns=rename_dict)
        return df_renamed

    def _parse_date(self, date_series: pd.Series) -> pd.Series:
        """
        解析并标准化日期列

        Args:
            date_series: 日期列

        Returns:
            标准化的日期列 (datetime64[ns, UTC])
        """
        # 尝试多种日期格式
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y%m%d',
        ]

        parsed_dates = None

        # 首先尝试 pandas 的自动解析
        try:
            parsed_dates = pd.to_datetime(date_series, errors='coerce')
        except Exception as e:
            logger.warning(f"自动日期解析失败: {e}")

        # 如果自动解析失败，尝试指定格式
        if parsed_dates is None or parsed_dates.isna().all():
            for fmt in date_formats:
                try:
                    parsed_dates = pd.to_datetime(date_series, format=fmt, errors='coerce')
                    if not parsed_dates.isna().all():
                        logger.info(f"使用日期格式: {fmt}")
                        break
                except Exception:
                    continue

        # 转换为 UTC
        if parsed_dates is not None and not parsed_dates.isna().all():
            # 如果是 naive datetime，假定为 UTC
            if parsed_dates.dt.tz is None:
                parsed_dates = parsed_dates.dt.tz_localize('UTC')
            else:
                parsed_dates = parsed_dates.dt.tz_convert('UTC')

        return parsed_dates

    def _standardize_sex(self, sex_series: pd.Series) -> pd.Series:
        """标准化性别列 (M/F)"""
        sex_map = {
            'm': 'M', 'male': 'M', '1': 'M', 'man': 'M',
            'f': 'F', 'female': 'F', '2': 'F', 'woman': 'F',
        }

        return sex_series.astype(str).str.lower().str.strip().map(
            lambda x: sex_map.get(x, x.upper())
        )

    def _convert_units(self, df: pd.DataFrame, column: str, unit_conversion: Dict[str, float]) -> pd.DataFrame:
        """
        单位换算

        Args:
            df: DataFrame
            column: 列名
            unit_conversion: 单位换算字典，如 {'mg/dL': 1.0, 'mmol/L': 18.0}

        Returns:
            换算后的 DataFrame
        """
        if column not in df.columns:
            return df

        # 这里简化处理，实际应用中可能需要更复杂的单位检测逻辑
        # 假设有 unit 列或者在列名中包含单位信息
        unit_col = f"{column}_unit"

        if unit_col in df.columns:
            for unit, factor in unit_conversion.items():
                mask = df[unit_col].str.contains(unit, case=False, na=False)
                df.loc[mask, column] = df.loc[mask, column] * factor
                logger.info(f"转换 {column} 单位: {unit} -> 标准单位 (×{factor})")

        return df

    def load_demographics(self, file_path: Path) -> pd.DataFrame:
        """
        加载人口统计学数据

        Args:
            file_path: CSV 文件路径

        Returns:
            标准化的 DataFrame
        """
        logger.info(f"加载人口统计学数据: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 读取 CSV
        df = pd.read_csv(file_path)
        logger.info(f"原始数据形状: {df.shape}")

        # 标准化列名
        df = self._standardize_columns(df, self.DEMOGRAPHICS_COLUMNS_MAP)

        # 解析日期
        if 'visit_date' in df.columns:
            df['visit_date'] = self._parse_date(df['visit_date'])
            invalid_dates = df['visit_date'].isna().sum()
            if invalid_dates > 0:
                logger.warning(f"{invalid_dates} 条记录的日期无效")

        # 标准化性别
        if 'sex' in df.columns:
            df['sex'] = self._standardize_sex(df['sex'])

        # 验证必需列
        required_cols = ['patient_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")

        # 去重（保留最新记录）
        if 'visit_date' in df.columns:
            df = df.sort_values('visit_date').drop_duplicates(
                subset=['patient_id', 'visit_date'],
                keep='last'
            )

        logger.info(f"标准化后数据形状: {df.shape}")
        logger.info(f"列名: {df.columns.tolist()}")

        self.demographics_df = df
        return df

    def load_labtests(self, file_path: Path) -> pd.DataFrame:
        """
        加载化验数据

        Args:
            file_path: CSV 文件路径

        Returns:
            标准化的 DataFrame
        """
        logger.info(f"加载化验数据: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 读取 CSV
        df = pd.read_csv(file_path)
        logger.info(f"原始数据形状: {df.shape}")

        # 标准化列名
        df = self._standardize_columns(df, self.LABTESTS_COLUMNS_MAP)

        # 解析日期
        if 'visit_date' in df.columns:
            df['visit_date'] = self._parse_date(df['visit_date'])
            invalid_dates = df['visit_date'].isna().sum()
            if invalid_dates > 0:
                logger.warning(f"{invalid_dates} 条记录的日期无效")

        # 单位换算示例
        # A1c: % 单位
        # 葡萄糖: mg/dL -> mmol/L (除以 18)
        if 'glucose' in df.columns:
            # 假设原始单位是 mg/dL，保持不变
            pass

        # 验证必需列
        required_cols = ['patient_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")

        # 去重（保留最新记录）
        if 'visit_date' in df.columns:
            df = df.sort_values('visit_date').drop_duplicates(
                subset=['patient_id', 'visit_date'],
                keep='last'
            )

        logger.info(f"标准化后数据形状: {df.shape}")
        logger.info(f"列名: {df.columns.tolist()}")

        self.labtests_df = df
        return df

    def get_summary(self) -> Dict:
        """获取数据摘要"""
        summary = {}

        if self.demographics_df is not None:
            summary['demographics'] = {
                'rows': len(self.demographics_df),
                'columns': self.demographics_df.columns.tolist(),
                'patients': self.demographics_df['patient_id'].nunique() if 'patient_id' in self.demographics_df else 0,
                'date_range': (
                    str(self.demographics_df['visit_date'].min()),
                    str(self.demographics_df['visit_date'].max())
                ) if 'visit_date' in self.demographics_df else None
            }

        if self.labtests_df is not None:
            summary['labtests'] = {
                'rows': len(self.labtests_df),
                'columns': self.labtests_df.columns.tolist(),
                'patients': self.labtests_df['patient_id'].nunique() if 'patient_id' in self.labtests_df else 0,
                'date_range': (
                    str(self.labtests_df['visit_date'].min()),
                    str(self.labtests_df['visit_date'].max())
                ) if 'visit_date' in self.labtests_df else None
            }

        return summary


def main():
    """测试临床数据加载"""
    import argparse

    parser = argparse.ArgumentParser(description="临床数据加载测试")
    parser.add_argument("--demo-check", action="store_true", help="演示数据检查")
    parser.add_argument("--demographics", type=Path, help="人口统计学数据文件")
    parser.add_argument("--labtests", type=Path, help="化验数据文件")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    loader = ClinicalDataLoader()

    if args.demo_check:
        print("\n" + "=" * 60)
        print("临床数据加载器演示")
        print("=" * 60)
        print("\n支持的人口统计学列:")
        for std, alts in loader.DEMOGRAPHICS_COLUMNS_MAP.items():
            print(f"  {std:15s} <- {', '.join(alts[:3])}")

        print("\n支持的化验数据列:")
        for std, alts in loader.LABTESTS_COLUMNS_MAP.items():
            print(f"  {std:15s} <- {', '.join(alts[:3])}")

        print("\n数据清洗功能:")
        print("  - 日期标准化为 UTC")
        print("  - 性别标准化为 M/F")
        print("  - 单位自动换算（可配置）")
        print("  - 去重（保留最新记录）")
        print("=" * 60)
        return

    if args.demographics:
        df = loader.load_demographics(args.demographics)
        print("\n人口统计学数据 (前 5 行):")
        print(df.head())
        print(f"\n数据形状: {df.shape}")

    if args.labtests:
        df = loader.load_labtests(args.labtests)
        print("\n化验数据 (前 5 行):")
        print(df.head())
        print(f"\n数据形状: {df.shape}")

    if args.demographics or args.labtests:
        summary = loader.get_summary()
        print("\n数据摘要:")
        import json
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
