import base64
import os
import pickle

import cohere
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer

from utils import get_embeddings, seed_everything, streamlit_header_and_footer_setup

# 在机器学习中，很多操作都涉及随机性（比如数据集划分、模型初始化等）
seed_everything(3777)

# 设置页面布局为宽屏模式，让内容可以占用更多水平空间
st.set_page_config(layout="wide")

# 设置页面的页眉和页脚（具体实现在 utils.py 中）
streamlit_header_and_footer_setup()

# 使用 markdown 添加一个二级标题 "Sentiment Analysis"，并带有表情符号
st.markdown("## Sentiment Analysis 🥺")

model_name: str = 'multilingual-22-12'
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)


def train_and_save():
    full_df = pd.read_json("./data/xed_with_embeddings.json", orient='index')
    df = full_df

    # MultiLabelBinarizer用于多标签分类问题，可以将文本标签转换为二进制格式
    # 比如['开心','悲伤'] -> [0,0,0,1,0,1,0,0]
    mlb = MultiLabelBinarizer()

    # X是特征矩阵(输入),将每个文本的embeddings(词向量)转换为numpy数组
    X = np.array(df.embeddings.tolist())

    # y是标签矩阵(输出),将文本情感标签转换为二进制格式
    # 一个文本可能同时包含多种情感,所以用多标签格式
    y = mlb.fit_transform(df.labels_text)

    # 获取所有可能的情感类别
    classes = mlb.classes_

    # 创建情感标签的索引映射字典
    classes_mapping = {index: emotion for index, emotion in enumerate(mlb.classes_)}

    # 将数据集分为训练集和测试集
    # test_size=0.01表示1%用于测试,99%用于训练
    # random_state设定随机种子,确保结果可复现
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    # 创建基础分类器:逻辑回归
    # 逻辑回归是一种基础的分类算法，尽管名字带"回归"，但它是用来做分类的
    # 它的工作原理是：
    # 1. 计算输入特征的加权和
    # 2. 通过sigmoid函数将结果压缩到0-1之间，得到概率值
    # 3. 如果概率>0.5判定为正类，否则为负类
    #
    # 优点：
    # 1. 训练速度快，计算简单
    # 2. 可以输出概率值，而不是仅仅给出分类结果
    # 3. 不容易过拟合
    # 4. 模型可解释性强
    #
    # solver='lbfgs'是优化算法,适用于小型数据集
    # random_state确保结果可复现
    base_lr = LogisticRegression(solver='lbfgs', random_state=0)

    # 创建分类器链
    # ClassifierChain用于处理多标签分类问题,考虑了标签之间的相关性
    # 比如"悲伤"和"恐惧"可能经常一起出现
    # order='random'表示随机排序标签
    chain = ClassifierChain(base_lr, order='random', random_state=0)

    # 训练模型
    chain.fit(X_train, y_train)

    # 在测试集上评估模型性能
    print(chain.score(X_test, y_test))

    # 保存训练好的模型到文件
    pickle.dump(chain, open("./data/models/emotion_chain.pkl", 'wb'))



classes_mapping = {
    0: 'Anger',
    1: 'Anticipation',
    2: 'Disgust',
    3: 'Fear',
    4: 'Joy',
    5: 'Sadness',
    6: 'Surprise',
    7: 'Trust'
}
# Pickle 文件是一种序列化和保存 Python 对象（如训练好的机器学习模型）的方式。
# 通过将训练好的模型保存为 pickle 文件，我们可以在以后重复使用该模型，
# 而无需从头开始重新训练。
model_path = "./data/models/emotion_chain.pkl"

@st.cache
def setup():
    emotions2image_mapping = {
        'Anger': './data/emotions/anger.gif',
        'Anticipation': './data/emotions/anticipation.gif',
        'Disgust': './data/emotions/disgust.gif',
        'Fear': './data/emotions/fear.gif',
        'Joy': './data/emotions/joy.gif',
        'Sadness': './data/emotions/sadness.gif',
        'Surprise': './data/emotions/surprise.gif',
        'Trust': './data/emotions/trust.gif',
    }
    for key, value in emotions2image_mapping.items():
        with open(value, "rb") as f:
            emotions2image_mapping[key] = f.read()

    chain_model = pickle.load(open(model_path, 'rb'))
    return emotions2image_mapping, chain_model


emotions2image_mapping, chain_model = setup()

# Streamlit 功能便于用户输入。
# 调用 st.text_input 创建一个页面对象，询问用户“您感觉如何？”
# 并捕获用户的文本响应。随后，通过 st.slider 向用户展示一个滑块，
# 用户可用其选择希望展示的前 k 种情绪数量。
feeling_text = st.text_input("How are you feeling?", "")
top_k = st.slider("Top Emotions", min_value=1, max_value=len(classes_mapping), value=1, step=1)


def score_sentence(text: str, top_k: int = 5):
    # 获取输入文本的词向量表示
    embeddings = torch.as_tensor(get_embeddings(co=co, model_name=model_name, texts=[text]), dtype=torch.float32)
    
    # 使用模型预测每个情感标签的概率
    outputs = torch.as_tensor(chain_model.predict_proba(embeddings), dtype=torch.float32)
    
    # 对概率进行排序,获取最可能的k个情感
    probas, indices = torch.sort(outputs)
    
    # 转换为numpy数组并反转顺序(从大到小)
    probas = probas.cpu().numpy()[0][::-1]
    indices = indices.cpu().numpy()[0][::-1]

    cols = st.columns(top_k, gap="large")
    for i, (index, p) in enumerate(zip(indices[:top_k], probas[:top_k])):
        if i % 3 == 0:
            cols = st.columns(3, gap="large")

        emotion = classes_mapping[index]

        i = i % 3
        image_file = emotions2image_mapping.get(emotion, None)
        if image_file:
            image_gif = base64.b64encode(image_file).decode("utf-8")
            cols[i].markdown(
                f'<img src="data:image/gif;base64,{image_gif}" style="width:250px;height:250px;border-radius: 25%;">',
                unsafe_allow_html=True,
            )
            cols[i].markdown("---")
        cols[i].markdown(f"**{emotion}**: {p * 100:.2f}%")

        print(f"Predicted emotion: {emotion}, with probability: {p}")


if feeling_text:
    score_sentence(feeling_text, top_k=top_k)
