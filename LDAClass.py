#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   LDAClass.py
@Time    :   2021/09/23 09:49:51
@Author  :   WuYifan 
@Version :   1.0
@Contact :   wyf3510@126.com
@Desc    :   LDA类
'''

# here put the import lib

import tomotopy as tp
import pandas as pd
import re
import jieba



class LDAClass(object):
    def __init__(self, file, file_model, k, dict, stop_dict, iter, log_file, alpha=0.1, eta=0.01, gamma=0.1,
                 rm_top=0, cf=0, df=0, tw=tp.TermWeight.ONE) -> None:
        super().__init__()

        self.file = file
        self.file_model = file_model
        self.k = k
        self.dict = dict
        self.stop_dict = stop_dict
        self.iter = iter
        self.log_file = log_file
        self.rm_top = rm_top
        self.cf = cf
        self.df = df
        self.tw = tw
        self.alpha = alpha
        self.eta = eta
        self.gamma = gamma 

    def Preprocess(self):
        """
        数据预处理
        """
        # data = pd.read_excel("data/1921-2021(百年作风建设).xlsx")
        # data = pd.read_excel(self.file)
        
        # documents = pd.read_excel(self.file, usecols=['正文','归类'])

        # documents = documents.rename(columns={'正文':'text','归类':'labels'})
       
        documents = pd.read_excel(self.file, usecols=['项目简介'])

        documents = documents.rename(columns={'项目简介':'text'})


        # pattern = re.compile('[a-zA-Z0-9]+^[1-9]\d*\.\d*|^[A-Za-z0-9]+$|^[0-9]*$|^(-?\d+)(\.\d+)?$|^[A-Za-z0-9]{4,40}.*?.（）')

        # labeled_documents = [(re.sub("[\s+\!\/_,$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+", "", row.text.strip().replace('\n', ''))) for index, row in documents.iterrows()]
        labeled_documents = [row.text.strip().replace('\n', '') for index, row in documents.iterrows()]

        return labeled_documents
    # print(labeled_documents)


    def make_corpus(self, mdl):
        """
        @description  : 词库构建处理
        @param  : mdl:模型
        @Returns  : None
        """
        
        # 分词

        # 导入自定义词库
        # jieba.load_userdict("data/word.txt")
        jieba.load_userdict(self.dict)

        # 导入停用词库
        # stop_words = [line.strip().lstrip() for line in open("data/stop_words.txt", 'r', encoding='utf-8').readlines()]
        stop_words = [line.strip().lstrip() for line in open(self.stop_dict, 'r', encoding='utf-8').readlines()]

        doc_corpus = []

        labeled_documents = self.Preprocess()

        # 设置词的优先级

        # 民航与无人机
        mdl.set_word_prior('无人机', [1.0 if k == 0 else 0.0 for k in range(self.k)])
        # 1-4月
        # mdl.set_word_prior('患者', [1.0 if k == 0 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('医疗队', [1.0 if k == 0 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('救治', [1.0 if k == 0 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('中医', [1.0 if k == 0 else 0.1 for k in range(self.k)])

        # mdl.set_word_prior('社区', [1.0 if k == 2 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('志愿者', [1.0 if k == 2 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('隔离', [1.0 if k == 2 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('一线', [0.7 if k == 2 else 0.2 for k in range(self.k)])


        # mdl.set_word_prior('打赢', [1.0 if k == 3 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('疫情防控阻击战', [1.0 if k == 3 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('党中央', [1.0 if k == 3 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('干部', [1.0 if k == 3 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('担当', [1.0 if k == 3 else 0.1 for k in range(self.k)])

        # mdl.set_word_prior('平台', [1.0 if k == 4 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('心理', [1.0 if k == 4 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('线上', [1.0 if k == 4 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('科技', [1.0 if k == 4 else 0.1 for k in range(self.k)])


        # mdl.set_word_prior('企业', [1.0 if k == 6 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('贷款', [1.0 if k == 6 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('脱贫攻坚', [1.0 if k == 6 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('农业', [1.0 if k == 6 else 0.1 for k in range(self.k)])

        # mdl.set_word_prior('物资', [0.5 if k == 8 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('捐赠', [0.7 if k == 8 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('防护服', [1.0 if k == 8 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('复工复产', [1.0 if k == 8 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('供应', [1.0 if k == 8 else 0.1 for k in range(self.k)])

        # mdl.set_word_prior('物资', [0.5 if k == 10 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('中方', [1.0 if k == 10 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('援助', [1.0 if k == 10 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('捐赠', [0.7 if k == 10 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('交流', [1.0 if k == 10 else 0.1 for k in range(self.k)])

        # mdl.set_word_prior('医护人员', [1.0 if k == 12 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('一线', [0.7 if k == 12 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('英雄', [1.0 if k == 12 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('加油', [1.0 if k == 12 else 0.1 for k in range(self.k)])

        # mdl.set_word_prior('药物', [1.0 if k == 13 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('疫苗', [1.0 if k == 13 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('研究', [1.0 if k == 13 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('信息', [1.0 if k == 13 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('临床', [1.0 if k == 13 else 0.1 for k in range(self.k)])


        # 5-12月
        # 加大权重
        # mdl.set_word_prior('医院', [1.0 if k == 0 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('党支部', [1.0 if k == 0 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('医学科', [1.0 if k == 0 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('街道', [1.0 if k == 0 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('社区', [1.0 if k == 0 else 0.1 for k in range(self.k)])
        # # # 降低权重
        # # mdl.set_word_prior('主任', [0.001 if k == 0 else 0.7 for k in range(self.k)])
        # # mdl.set_word_prior('主任医师', [0.001 if k == 0 else 0.7 for k in range(self.k)])
        # # mdl.set_word_prior('书记', [0.001 if k == 0 else 0.7 for k in range(self.k)])
        # # mdl.set_word_prior('党委', [0.001 if k == 0 else 0.7 for k in range(self.k)])
        # # mdl.set_word_prior('疾病', [0.001 if k == 0 else 0.7 for k in range(self.k)])
        # # mdl.set_word_prior('预防', [0.001 if k == 0 else 0.7 for k in range(self.k)])
        # # mdl.set_word_prior('院长', [0.001 if k == 0 else 0.7 for k in range(self.k)])
        # # mdl.set_word_prior('重症', [0.001 if k == 0 else 0.7 for k in range(self.k)])
        # # mdl.set_word_prior('感染', [0.001 if k == 0 else 0.7 for k in range(self.k)])

        # mdl.set_word_prior('救治', [1.0 if k == 1 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('中医', [1.0 if k == 1 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('治疗', [1.0 if k == 1 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('生命', [1.0 if k == 1 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('方舱', [1.0 if k == 1 else 0.1 for k in range(self.k)])

        # mdl.set_word_prior('人类命运共同体', [1.0 if k == 6 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('团结合作', [1.0 if k == 6 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('国际合作', [1.0 if k == 6 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('中方', [0.7 if k == 6 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('携手', [1.0 if k == 6 else 0.1 for k in range(self.k)])

        # mdl.set_word_prior('国家', [1.0 if k == 13 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('中华民族', [1.0 if k == 13 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('抗疫精神', [1.0 if k == 13 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('生命至上', [1.0 if k == 13 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('团结', [1.0 if k == 13 else 0.1 for k in range(self.k)])


        # mdl.set_word_prior('一带一路', [1.0 if k == 7 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('项目', [1.0 if k == 7 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('防疫', [1.0 if k == 7 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('共建', [1.0 if k == 7 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('生产', [1.0 if k == 7 else 0.1 for k in range(self.k)])

        # mdl.set_word_prior('疫情防控疫情防控', [1.0 if k == 12 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('常态化', [1.0 if k == 12 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('经济社会', [1.0 if k == 12 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('公共卫生', [1.0 if k == 12 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('健康', [1.0 if k == 12 else 0.1 for k in range(self.k)])

        # mdl.set_word_prior('就业', [1.0 if k == 10 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('农业', [1.0 if k == 10 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('合作', [1.0 if k == 10 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('复工复产', [1.0 if k == 10 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('服务', [1.0 if k == 10 else 0.1 for k in range(self.k)])

        # mdl.set_word_prior('美国', [1.0 if k == 4 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('人权', [1.0 if k == 4 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('死亡', [1.0 if k == 4 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('政客', [1.0 if k == 4 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('种族', [1.0 if k == 4 else 0.1 for k in range(self.k)])

        # mdl.set_word_prior('学生', [1.0 if k == 8 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('网络', [1.0 if k == 8 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('在线', [1.0 if k == 8 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('老师', [1.0 if k == 8 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('教学', [1.0 if k == 8 else 0.1 for k in range(self.k)])

        # mdl.set_word_prior('疫苗', [1.0 if k == 2 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('新冠肺炎', [1.0 if k == 2 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('世卫', [1.0 if k == 2 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('疫苗研发', [1.0 if k == 2 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('临床试验', [1.0 if k == 2 else 0.1 for k in range(self.k)])

        # mdl.set_word_prior('物资', [1.0 if k == 5 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('捐赠', [1.0 if k == 5 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('中方', [1.0 if k == 5 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('援助', [1.0 if k == 5 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('交流', [1.0 if k == 5 else 0.1 for k in range(self.k)])
        # mdl.set_word_prior('合作', [1.0 if k == 5 else 0.1 for k in range(self.k)])


        for document in labeled_documents:

            # 分词构建词袋，去掉停用词
            doc_words = [k for k in jieba.lcut(document, cut_all=False) if ((k not in stop_words)and(len(k)>1))]
            # 去掉标点符号
            labeled_documents = [row for row in doc_words if row not in "[\s+\!\/_,$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+"]
            # print(doc_words)  # 精确模式
            # doc_words = document.split()
            # print("doc_word: %s"%doc_words)
            if (len(doc_words) != 0):
                doc_corpus.append(doc_words)
                mdl.add_doc(doc_words)


        with open("data/doc_corpus.txt", 'a') as f:
            for i in doc_corpus:
                f.write(",".join(i) + '\n')
        print("词库构建完毕！", file=self.log_file)

        
        
    def TrainModel(self):

        mdl = tp.LDAModel(k=self.k, min_df=0)


        self.make_corpus(mdl)

        print("开始模型训练！")
        print("开始模型训练！", file=self.log_file)
        for i in range(0, self.iter, 10):
            mdl.train(10)
            # ll_per_word:模型每个单词的似然概率
            print('Iteration: {}\tLog-likelihood: {}\tPerplexity: {}'.format(i, mdl.ll_per_word, mdl.perplexity))
            # print('Iteration: {}\tLog-likelihood: {}\tPerplexity: {}'.format(i, mdl.ll_per_word, mdl.perplexity), file=self.log_file)

        print("模型训练完毕!开始存储模型！")
        print("模型训练完毕!开始存储模型！", file=self.log_file)
        # save into file
        mdl.save(self.file_model + 'LDA_K' + str(self.k) + '_it' + str(self.iter)+ '.bin', True)
        print("模型存储完毕！")
        print("模型存储完毕！", file=self.log_file)

        print("输出每个主题的前10个主题词：")
        # print("输出每个主题的前10个主题词：", file=self.log_file)

        result_topic = []
        for k in range(mdl.k):
            result_topic.append(mdl.get_topic_words(k, top_n=20))

            print('Top 10 words of topic #{}'.format(k))
            print(mdl.get_topic_words(k, top_n=20))

            # print('Top 10 words of topic #{}'.format(k), file=self.log_file)
            # print(mdl.get_topic_words(k, top_n=20), file=self.log_file)

        # result_df = pd.DataFrame(result_topic)
        # result_df.to_excel("data/result_"+str(mdl.k) + "min_df0ci.xlsx", index=False)

        print("模型summary:")
        print("模型summary:", file=self.log_file)
        mdl.summary(topic_word_top_n=20, file=self.log_file)
        mdl.summary(topic_word_top_n=20)


