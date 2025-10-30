#!/usr/bin/env python3
"""
TCIA 数据下载脚本

用法:
    python scripts/fetch_tcia.py --collection TCGA-OV --patient TCGA-04-1331 --out raw/tcia
    python scripts/fetch_tcia.py --list-collections
    python scripts/fetch_tcia.py --list-patients --collection TCGA-OV
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dt_pipeline.tcia_client import TCIAClient


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('tcia_download.log')
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="从 TCIA 下载医学影像数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出所有可用数据集
  python scripts/fetch_tcia.py --list-collections

  # 列出数据集中的患者
  python scripts/fetch_tcia.py --list-patients --collection TCGA-OV

  # 下载单个患者数据
  python scripts/fetch_tcia.py --collection TCGA-OV --patient TCGA-04-1331 --out raw/tcia

  # 下载整个数据集（谨慎使用！）
  python scripts/fetch_tcia.py --collection TCGA-OV --out raw/tcia --all-patients

  # 指定日期范围
  python scripts/fetch_tcia.py --collection TCGA-OV --patient TCGA-04-1331 \\
      --start-date 2020-01-01 --end-date 2020-12-31 --out raw/tcia
        """
    )

    # 基本参数
    parser.add_argument(
        "--api-key",
        help="TCIA API Key（可选，部分数据集需要）"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("raw/tcia"),
        help="输出目录 (默认: raw/tcia)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细日志输出"
    )

    # 查询操作
    query_group = parser.add_argument_group("查询操作")
    query_group.add_argument(
        "--list-collections",
        action="store_true",
        help="列出所有可用数据集"
    )
    query_group.add_argument(
        "--list-patients",
        action="store_true",
        help="列出数据集中的患者"
    )

    # 下载参数
    download_group = parser.add_argument_group("下载参数")
    download_group.add_argument(
        "--collection",
        help="数据集名称（例如: TCGA-OV）"
    )
    download_group.add_argument(
        "--patient",
        help="患者 ID"
    )
    download_group.add_argument(
        "--all-patients",
        action="store_true",
        help="下载数据集中的所有患者（谨慎使用！）"
    )
    download_group.add_argument(
        "--start-date",
        help="起始日期 (YYYY-MM-DD)"
    )
    download_group.add_argument(
        "--end-date",
        help="结束日期 (YYYY-MM-DD)"
    )
    download_group.add_argument(
        "--max-patients",
        type=int,
        help="最大下载患者数（与 --all-patients 配合使用）"
    )

    args = parser.parse_args()

    # 配置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # 创建客户端
    api_key = args.api_key or os.environ.get("TCIA_API_KEY")
    client = TCIAClient(api_key=api_key)

    # 列出数据集
    if args.list_collections:
        logger.info("获取 TCIA 数据集列表...")
        collections = client.get_collections()
        if collections:
            print(f"\n可用数据集 ({len(collections)} 个):")
            for i, col in enumerate(collections, 1):
                print(f"  {i:3d}. {col}")
        else:
            print("未获取到数据集列表")
        return

    # 列出患者
    if args.list_patients:
        if not args.collection:
            parser.error("--list-patients 需要指定 --collection")

        logger.info(f"获取数据集 {args.collection} 的患者列表...")
        patients = client.get_patients(args.collection)
        if patients:
            print(f"\n数据集 {args.collection} 包含 {len(patients)} 个患者:")
            for i, pat in enumerate(patients[:50], 1):
                print(f"  {i:3d}. {pat}")
            if len(patients) > 50:
                print(f"  ... 还有 {len(patients) - 50} 个患者")
        else:
            print(f"\n数据集 {args.collection} 中未找到患者")
            print("\n" + "=" * 60)
            print("可能的原因:")
            print("  1. TCIA API 需要认证（某些数据集需要 API Key）")
            print("  2. 数据集名称不正确")
            print("  3. API 访问限制或临时不可用")
            print("\n建议的解决方案:")
            print("  ① 手动下载:")
            print("     访问 https://nbia.cancerimagingarchive.net/")
            print("     搜索并下载 DICOM 数据")
            print("\n  ② 申请 API Key:")
            print("     https://wiki.cancerimagingarchive.net/x/X4ATBg")
            print(f"     然后使用: python scripts/fetch_tcia.py --api-key YOUR_KEY ...")
            print("\n  ③ 使用本地 DICOM 文件:")
            print("     查看文档: docs/USE_LOCAL_DICOM.md")
            print("     将 DICOM 文件放在: raw/tcia/{patient_id}/study_X/series_Y/")
            print("=" * 60)
        return

    # 下载数据
    if not args.collection:
        parser.error("下载数据需要指定 --collection")

    if not args.patient and not args.all_patients:
        parser.error("下载数据需要指定 --patient 或 --all-patients")

    # 创建输出目录
    args.out.mkdir(parents=True, exist_ok=True)

    # 下载单个患者
    if args.patient:
        logger.info(f"开始下载患者 {args.patient} 的数据...")
        stats = client.download_patient(
            collection=args.collection,
            patient_id=args.patient,
            output_dir=args.out,
            start_date=args.start_date,
            end_date=args.end_date
        )

        print("\n" + "=" * 60)
        print(f"下载完成 - 患者 {args.patient}")
        print("=" * 60)
        print(f"研究数量:   {stats['total_studies']}")
        print(f"序列总数:   {stats['total_series']}")
        print(f"成功下载:   {stats['downloaded_series']}")
        print(f"失败:       {stats['failed_series']}")
        print("=" * 60)

        if stats['studies']:
            print("\n研究详情:")
            for study in stats['studies']:
                print(f"\n  研究 UID: {study['study_uid'][:20]}...")
                print(f"  日期: {study['study_date']}")
                print(f"  描述: {study['study_description']}")
                print(f"  序列: {study['downloaded']}/{study['series_count']} 成功")

    # 下载所有患者
    elif args.all_patients:
        logger.info(f"获取数据集 {args.collection} 的患者列表...")
        patients = client.get_patients(args.collection)

        if not patients:
            logger.error(f"数据集 {args.collection} 中未找到患者")
            return

        if args.max_patients:
            patients = patients[:args.max_patients]
            logger.info(f"限制下载前 {args.max_patients} 个患者")

        print(f"\n将下载 {len(patients)} 个患者的数据")
        confirm = input("确认继续? (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            print("取消下载")
            return

        all_stats = []
        for i, patient_id in enumerate(patients, 1):
            logger.info(f"\n处理患者 {i}/{len(patients)}: {patient_id}")
            try:
                stats = client.download_patient(
                    collection=args.collection,
                    patient_id=patient_id,
                    output_dir=args.out,
                    start_date=args.start_date,
                    end_date=args.end_date
                )
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"患者 {patient_id} 下载失败: {e}")

        # 汇总统计
        total_studies = sum(s['total_studies'] for s in all_stats)
        total_series = sum(s['total_series'] for s in all_stats)
        downloaded = sum(s['downloaded_series'] for s in all_stats)
        failed = sum(s['failed_series'] for s in all_stats)

        print("\n" + "=" * 60)
        print(f"批量下载完成 - 数据集 {args.collection}")
        print("=" * 60)
        print(f"患者数量:   {len(all_stats)}")
        print(f"研究总数:   {total_studies}")
        print(f"序列总数:   {total_series}")
        print(f"成功下载:   {downloaded}")
        print(f"失败:       {failed}")
        print("=" * 60)


if __name__ == "__main__":
    main()
