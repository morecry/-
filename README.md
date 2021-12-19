# <i>语诗</i>
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

临近期末，本学期信息检索课程需要完成一个大作业，受清华大学孙茂松、刘志远老师团队开发的<a href='https://wantwords.thunlp.org'>万词王</a><i>(一个根据给定描述来检索相关词语表达的工具，也叫做<b>反向词典</b>)</i>启发，想到可以用同样的方式来检索诗歌。首先，我使用BERT-chinese预训练模型，在一个<a href='https://github.com/snowtraces/poetry-source'>诗歌数据集</a>下基于两个自监督任务来进行post-train，之后在一个<a href='https://github.com/THUNLP-AIPoet/CCPM'>诗歌翻译数据集</a>下进行fine-tune，使得相关语义的口语描述与诗歌的相似度能够更高。

最后，在`inference.py`脚本中，我提供了两种调用方式，第一种是由诗到诗的检索，即根据一句诗返回语义近似的诗句，第二种是由口语描述到诗的检索，即根据自然的口语表达返回语义近似的诗句，受数据集规模限制，第二种调用方式的效果一般，但是也算是基本实现了我最初的想法，为保证结果质量，目前只能检索到七言律诗，下图是两种调用方式返回的example：

<div align=center><img src="https://github.com/morecry/With-Poetry/blob/main/fig/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20211216222031.png" width="50%"></div>
<div align=center><img src="https://github.com/morecry/With-Poetry/blob/main/fig/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20211216222042.png" width="50%"></div>

----------------------------------------------------------------------------------------------------------------------------------------------------------------------
<b>按照下面的提示操作来训练模型或是直接复现结果</b>

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 首先安装依赖库
  * torch=1.4.0
  * transformers=2.7.0

## 直接测试结果
> 1. 下载预处理后的诗歌数据集文件<a href='https://cnbj1.fds.api.xiaomi.com/imagebucket/user/tuquan/all_pair_7.txt'>all_pair_7.txt</a>放入`data`文件夹中
> 2. 下载预训练模型文件<a href='https://cnbj1.fds.api.xiaomi.com/imagebucket/user/tuquan/bert_ch.bin'>bert_ch.bin</a>和<a href='https://cnbj1.fds.api.xiaomi.com/imagebucket/user/tuquan/bert_poem.bin'>bert_poem.bin</a>放入`output`文件夹中
> 3. 运行`inference.py` (如果你想使用其他query测试，修改该文件即可)，第一次运行可能时间较长，因为需要建立索引向量文件

## 重新训练
> 1. 下载预处理后的诗歌数据集文件<a href='https://cnbj1.fds.api.xiaomi.com/imagebucket/user/tuquan/all_pair_7.txt'>all_pair_7.txt</a>放入`data`文件夹中
> 2. 运行`split_data.py`
> 3. 下载bert-base-chinese预训练模型<a href='https://huggingface.co/bert-base-chinese/resolve/main/pytorch_model.bin'>pytorch_model.bin</a>放入`output`文件夹中
> 4. 使用如下命令添加后台训练任务 (多卡训练，如果是单卡将CUDA_VISIBLE_DEVICES后面的数字改成0即可)
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run.py --train_batch_size 2048 --eval_batch_size 2048 >train.log 2>&1 &
```
> 5. 训练结束后可以运行`inference.py`进行测试


如果您在运行本项目过程中遇到问题、有什么新的想法，或是您发现有更好的诗歌翻译数据集(且规模够大)，均可直接和我沟通, 下面是我的联系方式

 * <b>邮箱:</b> quantu@ruc.edu.cn
