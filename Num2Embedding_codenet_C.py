####把数字对应的code转换成对应的嵌入保存到本地，用的时候 直接加载本地的embedding进行计算时间的优化，不然 太慢了
import pickle
import numpy as np
import pickle
import torch
import ast
from tqdm import  tqdm
from functools import lru_cache
from transformers import AutoTokenizer, AutoModel
#
tokenizer = AutoTokenizer.from_pretrained("graphcodebert-base")  # 使用的是本地的codebert-base 或 graphcodebert-base
model = AutoModel.from_pretrained("graphcodebert-base")
# # 从本地文件加载字典
with open('./data/Afterprocess/problems_codenet_C.pkl', 'rb') as f:
    loaded_problems = pickle.load(f)

with open('data/Afterprocess/questions_codenet_C.pkl', 'rb') as f:
    loaded_questions = pickle.load(f)


#处理学生作答源代码的编码
# 创建一个新的字典，将（num, code）转换为（num, context_embeddings）
new_embeddings_dict = {}
for num, code in tqdm(loaded_problems.items()):
    # 截断数据，确保不超过模型要求的最大长度
    truncated_code = code[:512]
    # 使用 tokenizer 处理数据，并填充到最大长度
    inputs = tokenizer(truncated_code, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
    # inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        context_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # 转换为 NumPy 数组
    new_embeddings_dict[num] = context_embeddings

# 保存新的字典
np.save('embeddings_problems_codebert_codenet_C.npy', new_embeddings_dict)

##处理知识点题目
# 创建一个新的字典，将（num, code）转换为（num, context_embeddings）
new_question_embeddings_dict = {}
for num, skill in tqdm(loaded_questions.items()):
    # inputs = tokenizer(skill, return_tensors="pt", padding=True, truncation=False)
    truncated_code = skill[1][:100]
    # 使用 tokenizer 处理数据，并填充到最大长度
    inputs = tokenizer(truncated_code, return_tensors="pt", padding="max_length", max_length=100, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        context_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # 转换为 NumPy 数组
    new_question_embeddings_dict[num] = context_embeddings

# 保存新的字典
np.save('embeddings_questions_codebert_codenet_C.npy', new_question_embeddings_dict)

# #处理知识点对应的那个简写的提示
# new_question_embeddings_hint_dct = {}
# for num, skill in tqdm(loaded_questions.items()):
#     # inputs = tokenizer(skill, return_tensors="pt", padding=True, truncation=False)
#     truncated_code = skill[0][:20]
#     # 使用 tokenizer 处理数据，并填充到最大长度
#     inputs = tokenizer(truncated_code, return_tensors="pt", padding="max_length", max_length=20, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         context_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # 转换为 NumPy 数组
#     new_question_embeddings_hint_dct[num] = context_embeddings
#
# # 保存新的字典
# np.save('embeddings_questions_codenet_C_hint.npy', new_question_embeddings_hint_dct)


######上面的代码是生成对应的embedding向量的
######下面的代码是加载这些向量进行使用的 一般从其他模块引入下面的函数进行使用





# num = 1  # 例如，假设要访问数字为1的嵌入向量
# result = num2codeemb(num)
# if result is not None:
#     print("Embedding for num {}: {}".format(num, result))
# else:
#     print("Embedding for num {} not found".format(num))

