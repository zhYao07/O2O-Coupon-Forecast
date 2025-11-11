本项目用来记录自己参加天池新人实战赛O2O优惠券使用的方案，也算是入门了数据挖掘。

## 主要工作
详细请参考文件目录下的"O2O方案介绍.pptx"

## 文件目录结构
```bash
├── Fusion/                         # 三种模型融合相关代码与结果（最高rank融合。auc：0.8056）
│   ├── fusion_model/               # 模型融合脚本与实现
│   ├── model/                      # 各单模型的训练
│   └── model_result/               # 各单模型预测结果
│
├── O2O_dataset/                    # 存放原始O2O数据集
│
├── dataset/                        # 提取特征后生成的数据集
│
├── extract_history_feature.py      # 提取历史行为特征
├── extract_label_feature.py        # 提取标签相关特征
├── extract_other_feature.py        # 提取其他辅助特征
├── get_dataset.py                  # 划分数据集，提取历史，标签上的特征，得到最后的训练集，验证集和测试集
├── model.py                        # 模型训练与预测代码
│
├── feature_importance_gain.png     # 模型特征重要性可视化结果
├── result_0.8049.csv               # 模型预测结果文件，xgboost单模型（0.8049）
│
├── O2O方案介绍.pptx                # 项目方案介绍PPT
└── README.md                       # 项目说明文件
```
