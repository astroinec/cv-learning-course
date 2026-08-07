# 第 14 章数据说明

Notebook 使用内存合成输入，只验证端到端接口、导出一致性和计时方法，不包含生产数据。

## 可选官方来源

- COCO 官方页面：<https://cocodataset.org/>
- Open Images 官方页面：<https://storage.googleapis.com/openimages/web/index.html>
- Open Images 下载说明：<https://storage.googleapis.com/openimages/web/download_v7.html>

下载前应核对数据版本、图片级许可、标注许可、引用要求和再分发条件。COCO 与 Open Images 中的图片可能继承原作者许可，不能只看数据集总页就假定所有内容权利一致。

## 项目数据清单至少记录

- 数据版本、来源、许可和校验值；
- 类别表、标注规范、采集设备和日期；
- 按来源分组的 train/val/test 清单；
- 隐私复核、删除请求和已知偏差；
- 部署基准样例是否允许随模型一起分发。

本章不自动下载，也不建议把大型原始数据或模型权重提交到课程目录。
