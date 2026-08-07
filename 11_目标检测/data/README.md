# 第 11 章数据说明

Notebook 只使用代码中定义的合成框、分数与类别，不需要图像或标注文件。

## 可选官方来源

- COCO 官方页面：<https://cocodataset.org/>
- COCO 下载与许可说明：<https://cocodataset.org/#download>
- Pascal VOC 官方页面：<http://host.robots.ox.ac.uk/pascal/VOC/>

COCO 标注与图片许可并不一定相同：标注有公开许可说明，图片通常继承原 Flickr 作者许可。VOC 也有自己的使用条款。使用前应核对当前页面、图片级许可、引用要求和是否允许再分发。

本章不自动下载。若自行准备小子集，应保留原始 image id、annotation id、许可字段和划分清单，避免同一视频或近重复图片跨 train/val/test。
