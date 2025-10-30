"""
TCIA Client Module

与 The Cancer Imaging Archive (TCIA) API 交互，下载 DICOM 影像数据。

支持功能：
- 按 collection / patient 下载 DICOM
- 断点续传
- 最小重试机制
- API Key 认证

参考：https://wiki.cancerimagingarchive.net/x/fILTB
"""

import os
import json
import time
import logging
import requests
import requests.exceptions
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TCIAClient:
    """TCIA REST API 客户端"""

    BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 TCIA 客户端

        Args:
            api_key: TCIA API Key (可选，部分数据集需要)
        """
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"api_key": api_key})

    def get_collections(self) -> List[str]:
        """获取可用的数据集合列表"""
        url = f"{self.BASE_URL}/getCollectionValues"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            collections = [item["Collection"] for item in response.json()]
            logger.info(f"获取到 {len(collections)} 个数据集合")
            return collections
        except Exception as e:
            logger.error(f"获取数据集合列表失败: {e}")
            return []

    def get_patients(self, collection: str) -> List[str]:
        """
        获取指定数据集中的患者列表

        Args:
            collection: 数据集名称

        Returns:
            患者 ID 列表
        """
        url = f"{self.BASE_URL}/getPatient"
        params = {"Collection": collection, "format": "json"}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            # 尝试解析 JSON
            try:
                data = response.json()
            except ValueError as e:
                logger.error(f"无法解析 JSON 响应: {e}")
                logger.debug(f"响应内容: {response.text[:200]}")
                return []

            # 处理不同的响应格式
            if isinstance(data, list):
                # 尝试多种可能的键名
                possible_keys = ['PatientID', 'patientId', 'Subject ID', 'SubjectID']
                for key in possible_keys:
                    try:
                        patients = [item[key] for item in data if key in item]
                        if patients:
                            logger.info(f"数据集 {collection} 包含 {len(patients)} 个患者")
                            return patients
                    except (KeyError, TypeError):
                        continue

                # 如果上述都失败，尝试直接返回列表项
                if data and isinstance(data[0], str):
                    logger.info(f"数据集 {collection} 包含 {len(data)} 个患者")
                    return data

            logger.error(f"无法从响应中提取患者列表，响应格式: {type(data)}")
            if data:
                logger.debug(f"响应示例: {str(data)[:200]}")
            return []

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error(f"认证失败: TCIA API 需要有效的 API Key")
                logger.info(f"请访问 https://wiki.cancerimagingarchive.net/x/X4ATBg 获取 API Key")
            else:
                logger.error(f"HTTP 错误 {e.response.status_code}: {e}")
            return []
        except Exception as e:
            logger.error(f"获取患者列表失败: {e}")
            return []

    def get_studies(self, collection: str, patient_id: str) -> List[Dict[str, Any]]:
        """
        获取患者的研究列表

        Args:
            collection: 数据集名称
            patient_id: 患者 ID

        Returns:
            研究信息列表
        """
        url = f"{self.BASE_URL}/getPatientStudy"
        params = {
            "Collection": collection,
            "PatientID": patient_id
        }
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            studies = response.json()
            logger.info(f"患者 {patient_id} 有 {len(studies)} 个研究")
            return studies
        except Exception as e:
            logger.error(f"获取研究列表失败: {e}")
            return []

    def get_series(self, study_uid: str) -> List[Dict[str, Any]]:
        """
        获取研究的序列列表

        Args:
            study_uid: 研究 UID

        Returns:
            序列信息列表
        """
        url = f"{self.BASE_URL}/getSeries"
        params = {"StudyInstanceUID": study_uid}
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            series = response.json()
            logger.info(f"研究 {study_uid} 有 {len(series)} 个序列")
            return series
        except Exception as e:
            logger.error(f"获取序列列表失败: {e}")
            return []

    def download_series(
        self,
        series_uid: str,
        output_dir: Path,
        max_retries: int = 3
    ) -> bool:
        """
        下载指定序列的 DICOM 文件

        Args:
            series_uid: 序列 UID
            output_dir: 输出目录
            max_retries: 最大重试次数

        Returns:
            是否下载成功
        """
        url = f"{self.BASE_URL}/getImage"
        params = {"SeriesInstanceUID": series_uid}

        output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = output_dir / f"{series_uid}.zip"

        # 检查是否已下载
        if zip_path.exists():
            logger.info(f"序列 {series_uid} 已存在，跳过下载")
            return True

        for attempt in range(max_retries):
            try:
                logger.info(f"下载序列 {series_uid} (尝试 {attempt + 1}/{max_retries})")
                response = self.session.get(url, params=params, stream=True, timeout=300)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))

                with open(zip_path, 'wb') as f, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=f"下载 {series_uid[:8]}..."
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

                logger.info(f"序列 {series_uid} 下载完成")

                # 解压 ZIP 文件
                import zipfile
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(output_dir)
                    logger.info(f"解压完成: {output_dir}")
                    # 删除 ZIP 文件
                    zip_path.unlink()
                except Exception as e:
                    logger.warning(f"解压失败: {e}，保留 ZIP 文件")

                return True

            except Exception as e:
                logger.warning(f"下载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    logger.error(f"序列 {series_uid} 下载失败")
                    return False

        return False

    def download_patient(
        self,
        collection: str,
        patient_id: str,
        output_dir: Path,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        下载患者的所有 DICOM 数据

        Args:
            collection: 数据集名称
            patient_id: 患者 ID
            output_dir: 输出目录
            start_date: 起始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            下载统计信息
        """
        logger.info(f"开始下载患者 {patient_id} 的数据")

        patient_dir = output_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        # 获取研究列表
        studies = self.get_studies(collection, patient_id)

        # 过滤日期
        if start_date or end_date:
            filtered_studies = []
            for study in studies:
                study_date = study.get("StudyDate", "")
                if start_date and study_date < start_date.replace("-", ""):
                    continue
                if end_date and study_date > end_date.replace("-", ""):
                    continue
                filtered_studies.append(study)
            studies = filtered_studies
            logger.info(f"日期过滤后剩余 {len(studies)} 个研究")

        stats = {
            "patient_id": patient_id,
            "total_studies": len(studies),
            "total_series": 0,
            "downloaded_series": 0,
            "failed_series": 0,
            "studies": []
        }

        for study in studies:
            study_uid = study["StudyInstanceUID"]
            study_dir = patient_dir / f"study_{study_uid}"

            # 获取序列列表
            series_list = self.get_series(study_uid)
            stats["total_series"] += len(series_list)

            study_info = {
                "study_uid": study_uid,
                "study_date": study.get("StudyDate", ""),
                "study_description": study.get("StudyDescription", ""),
                "series_count": len(series_list),
                "downloaded": 0,
                "failed": 0
            }

            for series in series_list:
                series_uid = series["SeriesInstanceUID"]
                series_dir = study_dir / f"series_{series_uid}"

                if self.download_series(series_uid, series_dir):
                    stats["downloaded_series"] += 1
                    study_info["downloaded"] += 1
                else:
                    stats["failed_series"] += 1
                    study_info["failed"] += 1

            stats["studies"].append(study_info)

        # 保存元数据
        metadata_path = patient_dir / "download_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"患者 {patient_id} 下载完成: "
                   f"{stats['downloaded_series']}/{stats['total_series']} 个序列成功")

        return stats


def main():
    """测试 TCIA 客户端"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    client = TCIAClient()

    # 获取数据集列表
    collections = client.get_collections()
    if collections:
        print(f"\n可用数据集 ({len(collections)}):")
        for i, col in enumerate(collections[:10], 1):
            print(f"  {i}. {col}")
        if len(collections) > 10:
            print(f"  ... 还有 {len(collections) - 10} 个")


if __name__ == "__main__":
    main()
