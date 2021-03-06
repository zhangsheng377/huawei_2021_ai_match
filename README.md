# huawei_2021_ai_match
## 简介
文章质量判别是信息流领域的核心问题，提升文章质量判别的准确率是提升信息流质量和精准推送的核心技术点。在本次大赛中，主办方提供匿名化的文章质量数据，参赛选手基于给定的数据构建文章质量判别模型。希望通过本次大赛挖掘nlp算法领域的人才，推动 nlp算法的发展。

## 赛题说明
本题目将为选手提供文章数据，参赛选手基于给定的数据构建文章质量判别模型。所提供的数据经过脱敏处理，保证数据安全。

基础数据集包含两部分：训练集和测试集。其中训练集给定了该样本的文章质量的相关标签；测试集用于计算参赛选手模型的评分指标，参赛选手需要计算出测试集中每个样本文章质量判断及优质文章的类型。

## 数据列表
2021_2_data.zip
zip
2021-6-10

## 数据说明
### 数据概况

基础数据集包含两部分：训练集和测试集。其中训练集给定了该样本的文章质量的相关标签；测试集用于计算参赛选手模型的评分指标，参赛选手需要计算出测试集中每个样本文章质量判断及优质文章的类型。

#### train_data.json 训练数据

训练数据中没有预置质量类型为“其他”的数据，训练数据中包含大量未标注数据，即“doctype”为“”。

训练数据共五个字段，字段说明如下：

id
样本（文章）的唯一性id
title
样本（文章）的标题
body
样本（文章）的正文
category
文章分类编码
doctype
文章质量类别信息

#### test_data.json 测试数据

测试数据中包含大量文章质量类型为“其他”的数据，测试数据共四个字段，字段说明如下：

id
样本（文章）的唯一性id
title
样本（文章）的标题
body
样本（文章）的正文
category
文章分类编码
