import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 获取数据
df = pd.read_table('D:\文本挖掘\实验6\data\天猫各类商品评论.text', names=['type','content'], encoding='utf-8')
df = df[['type', 'content']]
print('数据总量：%d.' % len(df))
print(df.sample(100))

# 数据集预处理
print("在type列中总共有%d个空值." % df['type'].isnull().sum())
print("在content列中总共有%d个空值." % df['content'].isnull().sum())
df[df.isnull().values == True]
df = df[pd.notnull(df['content'])]
d = {'type': df['type'].value_counts().index, 'count': df['type'].value_counts()}
df_class = pd.DataFrame(data=d).reset_index(drop=True)
print(df_class)

df['type_id'] = df['type'].factorize()[0]
class_id_df = df[['type', 'type_id']].drop_duplicates().sort_values('type_id').reset_index(drop=True)
class_to_id = dict(class_id_df.values)
id_to_class = dict(class_id_df[['type_id', 'type']].values)
# df.sample(10)

# 定义删除字母，数字，汉字以外的所有符号函数
import re

def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    relu = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = relu.sub('', line)
    return line

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 加载停用词
stopwords = stopwordslist('D:\文本挖掘\实验6\data\中文停用词表.txt')

df['clean_content'] = df['content'].apply(remove_punctuation)
# df.sample(10)
# 分词，并过滤停用词

import jieba

df['cut_content'] = df['clean_content'].apply(lambda x: "".join([w for w in list(jieba.cut(x)) if w not in stopwords]))
df.head()

# LSTM建模(数据预处理完成以后，要开始进行LSTM的建模工作)：
from keras.preprocessing.text import Tokenizer

MAX_NB_WORDS = 50000
MAX_SEQUUENCE_LENGTH = 200  #每条cut_content最大长度
EMBEDDING_DIM = 100            #设置embeddingceng层的维度
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True)  # fileters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
tokenizer.fit_on_texts(df['cut_content'].values)
word_index = tokenizer.word_index
print('共有%s个不同的词语.' % len(word_index))

from keras.preprocessing.sequence import pad_sequences
X = tokenizer.texts_to_sequences(df['cut_content'].values)
X = pad_sequences(X, maxlen=MAX_SEQUUENCE_LENGTH)
Y = pd.get_dummies(df['type_id']).values
print(X.shape)
print(Y.shape)

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# 定义模型
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(9, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# 训练数据
# 定义好LSTM模型后，开始训练数据:设置5个训练周期（epochs），batch_size为64训练数据
from keras.callbacks import EarlyStopping

history = model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)
                               ])

# LSTM模型的评估:
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)
Y_test = Y_test.argmax(axis=1)
print('accuracy %s' % accuracy_score(y_pred, Y_test))

# 生成混淆矩阵
conf_mat = confusion_matrix(Y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=class_id_df.type.values, yticklabels=class_id_df.type.values)
plt.ylabel('实际结果', fontsize=16)
plt.ylabel('预测结果', fontsize=16)
plt.show()


# LSTM模型的测试:
def predict(text):
    txt = remove_punctuation(text)
    txt = [" ".join([w for w in list(jieba.cut(txt)) if w not in stopwords])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=MAX_SEQUUENCE_LENGTH)
    pred = model.predict(padded)
    class_id = pred.argmax(axis=1)[0]
    return class_id_df[class_id_df.type_id == class_id]['type'].values[0]
