# 与诗 (语诗)
临近期末，本学期信息检索课程需要完成一个大作业，受清华大学孙茂松、刘志远老师团队开发的<a href='https://wantwords.thunlp.org'>万词王</a><i>(一个根据给定描述来检索相关词语表达的工具，也叫做<b>反向词典</b>)</i>启发，想到可以用同样的方式来检索诗歌。首先，我使用BERT-chinese预训练模型，在一个<a href='https://github.com/snowtraces/poetry-source'>诗歌数据集</a>下基于两个自监督任务来进行post-train，之后在一个<a href='https://github.com/THUNLP-AIPoet/CCPM'>诗歌翻译数据集</a>下进行fine-tune，使得相关语义的口语与诗歌的相似度能够更高。

最后，在`inference.py`脚本中，我提供了两种调用方式，第一种是由诗到诗的检索，即根据一句诗返回语义近似的诗句，第二种是由口语描述到诗的检索，即根据自然的口语表达返回语义近似的诗句，受数据集规模限制，第二种调用方式的效果一般，但是也算是基本实现了我最初的想法，下图是两种调用方式的example：

<div align=center><img src="https://github.com/morecry/With-Poetry/blob/main/fig/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20211216222031.png" width="50%"></div>
<div align=center><img src="https://github.com/morecry/With-Poetry/blob/main/fig/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20211216222042.png" width="50%"></div>


  * [And a table of contents](#and-a-table-of-contents)
  * [On the right](#on-the-right)

<small><i></i></small>
