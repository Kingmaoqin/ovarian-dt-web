#!/usr/bin/env python3
"""
同步数据到前端

将 timeline.json 和各 visit 的 .ply/.vtp 文件同步到 ovarian-dt-web/data/。

用法:
    python scripts/sync_to_web.py
    python scripts/sync_to_web.py --timeline work/timeline.json --web-dir ovarian-dt-web
"""

import sys
import shutil
import argparse
import logging
import json
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def sync_to_web(
    timeline_path: Path,
    web_dir: Path,
    source_data_dir: Path
):
    """
    同步数据到前端

    Args:
        timeline_path: timeline.json 路径
        web_dir: 前端目录
        source_data_dir: 源数据目录
    """
    logger = logging.getLogger(__name__)

    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline.json 不存在: {timeline_path}")

    # 创建前端 data 目录
    web_data_dir = web_dir / 'data'
    web_data_dir.mkdir(parents=True, exist_ok=True)

    # 复制 timeline.json
    target_timeline = web_data_dir / 'timeline.json'
    shutil.copy2(timeline_path, target_timeline)
    logger.info(f"复制 timeline.json -> {target_timeline}")

    # 加载 timeline
    with open(timeline_path, 'r') as f:
        timeline = json.load(f)

    # 同步每个患者的数据
    total_files = 0
    for patient in timeline:
        patient_id = patient['patient_id']

        for visit in patient['visits']:
            visit_num = visit['visit']
            source_visit_dir = source_data_dir / patient_id / f"visit_{visit_num}"
            target_visit_dir = web_data_dir / patient_id / f"visit_{visit_num}"

            if not source_visit_dir.exists():
                logger.warning(f"源目录不存在: {source_visit_dir}")
                continue

            target_visit_dir.mkdir(parents=True, exist_ok=True)

            # 复制文件
            for ext in ['*.ply', '*.vtp', '*.stl']:
                for file in source_visit_dir.glob(ext):
                    target_file = target_visit_dir / file.name
                    shutil.copy2(file, target_file)
                    logger.debug(f"复制 {file.name} -> {target_file}")
                    total_files += 1

            logger.info(f"同步 {patient_id}/visit_{visit_num}")

    logger.info(f"同步完成: {total_files} 个文件")


def main():
    parser = argparse.ArgumentParser(description="同步数据到前端")
    parser.add_argument(
        "--timeline",
        type=Path,
        default=Path("ovarian-dt-web/data/timeline.json"),
        help="timeline.json 路径 (默认: ovarian-dt-web/data/timeline.json)"
    )
    parser.add_argument(
        "--source-data",
        type=Path,
        default=Path("ovarian-dt-web/data"),
        help="源数据目录 (默认: ovarian-dt-web/data)"
    )
    parser.add_argument(
        "--web-dir",
        type=Path,
        default=Path("ovarian-dt-web"),
        help="前端目录 (默认: ovarian-dt-web)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    print("\n" + "=" * 60)
    print("同步数据到前端")
    print("=" * 60)
    print(f"Timeline:   {args.timeline}")
    print(f"源数据:     {args.source_data}")
    print(f"前端目录:   {args.web_dir}")
    print("=" * 60 + "\n")

    try:
        sync_to_web(args.timeline, args.web_dir, args.source_data)
        print("\n✓ 同步完成！")
        print(f"\n前端地址: file://{args.web_dir.absolute()}/index.html")
    except Exception as e:
        logger.error(f"同步失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
