# 与诗 
临近期末，本学期信息检索课程需要完成一个大作业，受清华大学孙茂松、刘志远老师团队开发的万词王工具(能根据描述来query到相关的词语的反向词典)启发，想到可以用描述来检索诗歌。首先，我使用BERT-chinese预训练模型，在诗歌数据集下基于两个自监督任务来进行post-train，之后在一个诗歌翻译数据集下进行fine-tune，使得相关的口语与诗歌在语义上能够更加接近。
  * [And a table of contents](#and-a-table-of-contents)
  * [On the right](#on-the-right)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>
