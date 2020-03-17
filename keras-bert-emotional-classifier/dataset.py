# -*- coding: utf-8 -*-
# @File    : dataset.py
# @Author  : AaronJny
# @Time    : 2020/01/04
# @Desc    :
import chardet
from collections import Counter
import time
from bert4keras.tokenizer import Tokenizer, load_vocab
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.bert import build_bert_model
import numpy as np
import keras

# 预训练的模型参数
CONFIG_PATH = '/home/aaron/tools/uncased_L-12_H-768_A-12/bert_config.json'
CHECKPOINT_PATH = '/home/aaron/tools/uncased_L-12_H-768_A-12/bert_model.ckpt'
DICT_PATH = '/home/aaron/tools/uncased_L-12_H-768_A-12/vocab.txt'
# 负例样本路径
NEG_PATH = 'rt-polarity.neg'
# 正例样本路径
POS_PATH = 'rt-polarity.pos'
# 评论最大长度
MAX_LEN = 256
# 允许的最小词频
MIN_WORD_FREQUENCY = 5
# 数据集划分比例
TRAIN_SPLIT, DEV_SPLIT = 0.8, 0.1


def load_data(path, label):
    """
    从给定路径加载数据
    :param path: 数据路径
    :param label: 数据标签
    """
    # 从文本中读取数据
    with open(path, 'rb') as f:
        text_bin = f.read()
    # 获取文本编码格式
    encoding = chardet.detect(text_bin)['encoding']
    # 解码文本
    text = text_bin.decode(encoding=encoding).lower()
    # 按行切分数据，并移除超出长度的数据
    lines = [line for line in text.splitlines()]
    print(len(lines))
    _data = []
    for line in lines:
        if len(line) <= MAX_LEN - 2:
            _data.append(line)
    print(len(_data))
    _labels = [label] * len(_data)
    return _data, _labels


# 读取全部数据
all_data, all_labels = [], []
neg_data, neg_labels = load_data(NEG_PATH, 0)
all_data.extend(neg_data)
all_labels.extend(neg_labels)
pos_data, pos_labels = load_data(POS_PATH, 1)
all_data.extend(pos_data)
all_labels.extend(pos_labels)
# 混洗数据
seed = int(time.time())
np.random.seed(seed)
np.random.shuffle(all_data)
np.random.seed(seed)
np.random.shuffle(all_labels)
# 划分数据集
samples = len(all_data)
train_samples = int(samples * TRAIN_SPLIT)
dev_samples = int(samples * DEV_SPLIT)
train_data, train_labels = all_data[:train_samples], all_labels[:train_samples]
dev_data, dev_labels = all_data[train_samples:train_samples + dev_samples], all_labels[
                                                                            train_samples:train_samples + dev_samples]
test_data, test_labels = all_data[train_samples + dev_samples:], all_labels[train_samples + dev_samples:]

# 加载预训练模型的词典
_token_dict = load_vocab(DICT_PATH)
_tokenizer = Tokenizer(_token_dict, do_lower_case=True)
print(all_data[0])
print(_tokenizer.encode(all_data[0]))
print(_tokenizer.tokenize(all_data[0]))
print([_tokenizer.id_to_token(21934)])
print(_tokenizer.token_to_id('[PAD]'))

# 统计数据集中的词频
counter = Counter()
for line in all_data:
    _tokens = _tokenizer.tokenize(line)
    # 统计词频时，移除[CLS]和[SEP]字符
    counter.update(_tokens[1:-1])
print(len(counter))
# 移除词频较低的词
_tokens = [token for token, cnt in counter.items() if cnt >= MIN_WORD_FREQUENCY]
print(len(_tokens))
# 构建新词典
_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + _tokens
keep_words = []
token_dict = {}
for token in _tokens:
    token_dict[token] = len(token_dict)
    keep_words.append(_token_dict[token])
# 使用新词典构建分词器
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class MyDataGenerator(DataGenerator):

    def __init__(self, data, labels, batch_size=32):
        super().__init__(data, batch_size)
        self.labels = labels

    def __iter__(self, random=True):
        # 混洗数据
        if random:
            seed = int(time.time())
            np.random.seed(seed)
            np.random.shuffle(self.data)
            np.random.seed(seed)
            np.random.shuffle(self.labels)
        total = len(self.data)
        # 迭代整个数据集，每次返回一个mini batch
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch_token_ids = []
            batch_segment_ids = []
            for line in self.data[start:end]:
                _token_ids, _segment_ids = tokenizer.encode(line)
                batch_token_ids.append(_token_ids)
                batch_segment_ids.append(_segment_ids)
            batch_token_ids = sequence_padding(batch_token_ids)
            batch_segment_ids = sequence_padding(batch_segment_ids)
            batch_labels = np.reshape(np.array(self.labels[start:end]), (-1, 1))
            yield [batch_token_ids, batch_segment_ids], batch_labels
            del batch_labels, batch_token_ids, batch_segment_ids


bert_model = build_bert_model(config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH, keep_words=keep_words)
# bert_model.trainable = False
output = keras.layers.Lambda(lambda x: x[:, 0])(bert_model.output)
output = keras.layers.Dropout(rate=0.5)(output)
output = keras.layers.Dense(1, activation=keras.activations.sigmoid)(output)
model = keras.Model(bert_model.input, output)
model.summary()

train_data_generator = MyDataGenerator(train_data, train_labels)
dev_data_generator = MyDataGenerator(dev_data, dev_labels)
test_data_generator = MyDataGenerator(test_data, test_labels)

model.compile(keras.optimizers.Adam(1e-6, decay=1e-4), keras.losses.binary_crossentropy, metrics=['accuracy'])
model.fit_generator(train_data_generator.forfit(), steps_per_epoch=train_data_generator.steps, epochs=10,
                    validation_data=dev_data_generator.forfit(), validation_steps=dev_data_generator.steps)
print(model.evaluate_generator(test_data_generator.forfit(), steps=test_data_generator.steps))
