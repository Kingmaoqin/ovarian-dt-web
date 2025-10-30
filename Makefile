# 卵巢癌数字孪生项目 Makefile

.PHONY: help setup demo clean test

# Python 解释器
PYTHON := python3

# 项目目录
PROJECT_DIR := $(shell pwd)
SCRIPTS_DIR := $(PROJECT_DIR)/scripts
DATA_DIR := $(PROJECT_DIR)/ovarian-dt-web/data
WORK_DIR := $(PROJECT_DIR)/work

help:
	@echo "卵巢癌数字孪生项目 - Makefile"
	@echo ""
	@echo "可用命令:"
	@echo "  make setup       - 创建必要的目录结构"
	@echo "  make demo        - 运行最小演示案例"
	@echo "  make clean       - 清理生成的文件"
	@echo "  make test        - 运行测试"
	@echo ""
	@echo "完整管线示例:"
	@echo "  1. make setup"
	@echo "  2. 准备 DICOM 数据到 raw/tcia/"
	@echo "  3. 准备临床数据到 raw/clinical/"
	@echo "  4. make demo"

setup:
	@echo "创建项目目录结构..."
	mkdir -p dt_pipeline scripts ovarian-dt-web/{data,assets}
	mkdir -p raw/{tcia,clinical} work/{feats,models}
	mkdir -p examples/minicase
	@echo "✓ 目录结构创建完成"

demo:
	@echo "运行演示案例..."
	@echo ""
	@echo "注意: 演示需要准备测试数据"
	@echo "请参考 examples/minicase/README.md"
	@echo ""
	@echo "完整管线命令示例:"
	@echo ""
	@echo "# 1. 生成几何模型"
	@echo "$(PYTHON) $(SCRIPTS_DIR)/build_geometry.py \\"
	@echo "    --patient DEMO001 \\"
	@echo "    --dicom-dir raw/tcia/DEMO001/study_1/series_1 \\"
	@echo "    --visit 1 --date 2024-01-01 \\"
	@echo "    --threshold -200 300"
	@echo ""
	@echo "# 2. 数据融合"
	@echo "$(PYTHON) -m dt_pipeline.joiner \\"
	@echo "    --timeline $(DATA_DIR)/timeline.json \\"
	@echo "    --output $(WORK_DIR)/feats"
	@echo ""
	@echo "# 3. 训练分类模型"
	@echo "$(PYTHON) $(SCRIPTS_DIR)/run_classify.py \\"
	@echo "    --data $(WORK_DIR)/feats/tabular_X.csv \\"
	@echo "    --labels $(WORK_DIR)/feats/y.csv \\"
	@echo "    --model xgb --out $(WORK_DIR)/models/xgb.pkl"
	@echo ""
	@echo "# 4. 训练生存模型"
	@echo "$(PYTHON) $(SCRIPTS_DIR)/run_survival.py \\"
	@echo "    --x $(WORK_DIR)/feats/surv_X.csv \\"
	@echo "    --duration $(WORK_DIR)/feats/duration.csv \\"
	@echo "    --event $(WORK_DIR)/feats/event.csv \\"
	@echo "    --out $(WORK_DIR)/models/coxph.pkl"
	@echo ""
	@echo "# 5. 同步到前端"
	@echo "$(PYTHON) $(SCRIPTS_DIR)/sync_to_web.py"

clean:
	@echo "清理生成的文件..."
	rm -rf work/feats/*
	rm -rf work/models/*
	rm -f *.log
	rm -f build_geometry.log tcia_download.log run_classify.log run_survival.log
	@echo "✓ 清理完成"

test:
	@echo "运行测试..."
	$(PYTHON) -m pytest tests/ -v

# 列出所有可用脚本
list-scripts:
	@echo "可用脚本:"
	@ls -1 $(SCRIPTS_DIR)/*.py | xargs -I {} basename {}

# 检查依赖
check-deps:
	@echo "检查 Python 依赖..."
	@$(PYTHON) -c "import open3d; print('✓ open3d')"
	@$(PYTHON) -c "import numpy; print('✓ numpy')"
	@$(PYTHON) -c "import pandas; print('✓ pandas')"
	@$(PYTHON) -c "import sklearn; print('✓ scikit-learn')"
	@$(PYTHON) -c "import skimage; print('✓ scikit-image')"
	@$(PYTHON) -c "import xgboost; print('✓ xgboost')" || echo "⚠ xgboost 未安装（可选）"
	@$(PYTHON) -c "import lifelines; print('✓ lifelines')" || echo "⚠ lifelines 未安装（可选）"
	@$(PYTHON) -c "import pydicom; print('✓ pydicom')"
	@$(PYTHON) -c "import requests; print('✓ requests')"
	@$(PYTHON) -c "import tqdm; print('✓ tqdm')"
	@$(PYTHON) -c "import yaml; print('✓ pyyaml')"
	@echo ""
	@echo "✓ 依赖检查完成"
