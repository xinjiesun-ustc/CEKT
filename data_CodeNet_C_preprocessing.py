####这里的question对应的代码里面的题干 当知识点用  problem是学生的作答的代码
import pandas as pd
import os
import numpy as np
import tqdm
import random
import pickle
from bs4 import BeautifulSoup
from tqdm import  tqdm


from Code_Gramer_Quality_Rate import  analyze_code_gramer_rate_quality, evaluate_code_quality_and_save


MAX_STEP= 180

# 文件夹路径
folder_path = 'F:\教育数据集\C\metadata'

# 初始化一个空列表来存储每个文件的DataFrame
dataframes = []

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否是CSV文件
    if filename.endswith('.csv'):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, filename)
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # 检查'filename_ext'列是否存在
        if 'filename_ext' in df.columns:  # 检查'filename_ext'列是否存在
            filtered_df = df[df['filename_ext'] == 'c']  # 筛选出filename_ext字段值为'py'的行

            if 'status' in filtered_df.columns:  # 检查'status'列是否存在
                # 使用.loc来修改status列的值，如果为'Accepted'则设置为1，否则设置为0
                filtered_df.loc[:, 'status'] = np.where(filtered_df['status'] == 'Accepted', 1, 0)
            else:
                print(f"Warning: 'status' column not found in {filename}")

            # 删除具有空字段的行
            filtered_df = filtered_df.dropna()

            dataframes.append(filtered_df)  # 将修改后的DataFrame添加到列表中
        else:
            print(f"Warning: 'filename_ext' column not found in {filename}")

# 使用concat函数合并所有DataFrame
merged_df = pd.concat(dataframes, ignore_index=True)

# 保存合并后的DataFrame到一个新的CSV文件
# merged_df.to_csv('merged_data.csv', index=False)
# 打印DataFrame的大小（行数和列数）
print("DataFrame size berfor (rows, columns):", merged_df.shape)

#准备二次处理这个数据集  因为里面有很多的学生的答题记录是不存在的


# 文件夹路径
folder_path1 = 'F:\\教育数据集\\C\\C'

# 获取 raw_answer_code_index 中的唯一值
raw_answer_code_index = merged_df.submission_id.unique().tolist()

# 存储代码内容的字典
code_contents = {}

# 递归遍历文件夹及其子文件夹
def search_files(directory):
    for root, dirs, files in tqdm(os.walk(directory)):
        for file in files:
            if file.endswith('.c'):
                file_path = os.path.join(root, file)
                file_name_without_extension = os.path.splitext(file)[0]
                if file_name_without_extension in raw_answer_code_index:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_contents[file_name_without_extension] = f.read()


import os
import pickle

# 检查文件是否存在
file_path_code_contents= 'data/Afterprocess/problems_code_contents_codenet_C.pkl'
if os.path.exists(file_path_code_contents):
    # 如果文件存在，则加载字典
    with open(file_path_code_contents, 'rb') as f:
        code_contents = pickle.load(f)
else:
    # 如果文件不存在，则运行函数 search_files(folder_path1)
    search_files(folder_path1)
    with open('data/Afterprocess/problems_code_contents_codenet_C.pkl', 'wb') as f:
        pickle.dump(code_contents, f)

# search_files(folder_path1) #该函数的功能是 取出submission_id对应的学生作答代码  并构成一个字典code_contents key=submission_id value=code的具体内容

# 将字典存储到本地文件


# 提取 submission_id 列，并检查是否在 code_contents 字典的键中
submission_ids = merged_df['submission_id']
mask = submission_ids.isin(code_contents)

# 根据布尔索引过滤 DataFrame，并保留在字典中的行
filtered_df = merged_df[mask]

# 如果需要，你可以覆盖原始 DataFrame，删除不在字典中的行
# merged_df = merged_df[mask]

# 如果需要，你也可以将 DataFrame 的索引重置为默认索引
filtered_df.reset_index(drop=True, inplace=True)

# 现在，filtered_df 中的 submission_id 列中的值都在 code_contents 字典的键中

# 创建一个字典，一个键对应两个值
num2questiondict_temp = {}

raw_question = filtered_df.problem_id.unique().tolist()  #在这个代码中 把问题的题干当做知识点了
num_skill = len(raw_question)
print(f"number of  skill  befor >=N 条记录之前 = {num_skill}")


def getquestion_content(folder_path2):
    for filename in tqdm(os.listdir(folder_path2)):
        # 检查文件是否是html文件
        if filename.endswith('.html'):
            file_path = os.path.join(folder_path2, filename)
            file_name_without_extension2 = os.path.splitext(filename)[0]
            if file_name_without_extension2 in raw_question:
                with open(file_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()
                    # 创建BeautifulSoup对象并解析HTML内容
                    soup = BeautifulSoup(html_content, 'html.parser')
                    # 查找<h2>标签
                    h2_tag = soup.find('h2')

                    if h2_tag:
                        # 获取<h2>标签之前的兄弟元素
                        content_before_h2 = [sibling for sibling in h2_tag.previous_siblings if
                                             sibling.name is not None]
                        # 提取H1标签内容
                        h1_tag = soup.find('h1')
                        if h1_tag:
                            h1_content = h1_tag.text
                            # 进行接下来的操作
                        else:
                            h1_content = '0'
                        # 提取p标签内容
                        p_tag = soup.find('p')
                        if p_tag:
                            p_content = p_tag.text
                        else:
                            p_content = '0'

                        num2questiondict_temp[file_name_without_extension2] = [h1_content, p_content]
                    # else:
                    #     # num2questiondict_temp[file_name_without_extension2] = ['0', '0']

                    # print(f"File: {filename}")
                    # print("No <h2> tag found in the file.")
                    # print("\n")
# 检查文件是否存在
file_path_question_contents= './data/questions_num2questiondict_temp_codenet_C.pkl'
if os.path.exists(file_path_question_contents):
    # 如果文件存在，则加载字典
    with open(file_path_question_contents, 'rb') as f:
        num2questiondict_temp = pickle.load(f)
else:
    # 如果文件不存在，则运行函数 search_files(folder_path1)
    folder_path2 = 'F:\教育数据集\C\problem_descriptions'
    getquestion_content(folder_path2)
        # 将字典存储到本地文件
    with open('./data/Afterprocess/questions_num2questiondict_temp_codenet_C.pkl', 'wb') as f:
        pickle.dump(num2questiondict_temp, f)




# 提取 submission_id 列，并检查是否在 code_contents 字典的键中
question_ids = filtered_df['problem_id']
mask = question_ids.isin(num2questiondict_temp)

# 根据布尔索引过滤 DataFrame，并保留在字典中的行
filtered_df_2 = filtered_df[mask]

# 如果需要，你可以覆盖原始 DataFrame，删除不在字典中的行
# merged_df = merged_df[mask]

# 如果需要，你也可以将 DataFrame 的索引重置为默认索引
filtered_df_2.reset_index(drop=True, inplace=True)

# 根据'user_id'字段过滤掉少于100条记录的用户数据
merged_df_filtered = filtered_df_2.groupby('user_id').filter(lambda x: len(x) >= 100)
print("DataFrame size after row >= 100 for every user (rows, columns):", merged_df_filtered.shape)

data = merged_df_filtered
#
# data = merged_df.sample(frac=0.0001)  #取少部分数据 方便测试


raw_question = data.problem_id.unique().tolist()  #在这个代码中 把问题的题干当做知识点了
num_skill = len(raw_question)
print(f"number of  skill = {num_skill}")



# 获取 raw_answer_code_index 中的唯一值
raw_answer_code_index = data.submission_id.unique().tolist()
# question id from 1 to (num_skill )
questions = { q: i+1  for i, q in enumerate(raw_question) }  #字典

# problem id from 1 to (num_problem )
problem = { p: i+1 for i, p in enumerate(raw_answer_code_index) }
#
#
# 反转键值对
reversed_questions = {v: k for k, v in questions.items()}
# 反转键值对
reversed_problems = {v: k for k, v in problem.items()}


#删除掉不符合长度要求的序列后 再次查找目前仅存的代码是哪些  并按key是序列好  value是对应的代码存放在本地磁盘
# code_contents = {}
# search_files(folder_path1)

# 从 code_contents 中查找满足 raw_answer_code_index 的条目
filtered_code_contents = {key: value for key, value in tqdm(code_contents.items(), total=len(code_contents)) if key in raw_answer_code_index}
linked_code_contents = {}

for problem_value, code_content in filtered_code_contents.items():
    original_problem = problem[problem_value]
    linked_code_contents[problem_value] = {'original_problem': original_problem, 'code_content': code_content}

num2codedict= {}  #存放最终的代码映射的字典  key=本程序从1从新编码后的代码的编码   value = 本程序使用到的学生的作答代码
# 现在可以使用 reversed_problems 中的值来访问 linked_code_contents 字典
for problem_value, linked_info in linked_code_contents.items():
    original_problem = linked_info['original_problem']
    code_content = linked_info['code_content']
    num2codedict[original_problem] = code_content


#删除掉不符合长度要求的序列后 再次查找目前仅存的问题是哪些  并按key是序列好  value是对应的代码存放在本地磁盘
# num2questiondict_temp={}
# getquestion_content(folder_path2)
# 从 code_contents 中查找满足 raw_answer_code_index 的条目
filtered_question_contents = {key: value for key, value in tqdm(num2questiondict_temp.items(), total=len(num2questiondict_temp)) if key in raw_question}

linked_question_contents = {}

for question_value, quesiton_content in filtered_question_contents.items():
    original_question = questions[question_value]
    linked_question_contents[question_value] = {'original_question': original_question, 'quesiton_content': quesiton_content}

num2questiondict =  {}  #存放最终的代码映射的字典  key=本程序从1从新编码后的代码的编码   value = 本程序使用到的学生的作答代码
# 现在可以使用 reversed_problems 中的值来访问 linked_code_contents 字典
for question_value, linked_info in linked_question_contents.items():
    original_question = linked_info['original_question']
    question_content = linked_info['quesiton_content']
    num2questiondict [original_question] = question_content
# 将字典存储到本地文件
with open('data/Afterprocess/questions_codenet_C.pkl', 'wb') as f:
    pickle.dump(num2questiondict, f)
#
with open('./data/Afterprocess/problems_codenet_C.pkl', 'wb') as f:
    pickle.dump(num2codedict, f)


users = data.user_id.unique().tolist()
print(f"number of  users = {len(users)}")
print(f"average  of  exercise = {data.shape[0]/len(users)}")



##开始往tl格式数准备数据
#生成problem代码的语法得分和质量得分

#语法得分
file_path_problem_gramer_rate= './problem_Gramer_Rate_dict.npy'
if os.path.exists(file_path_problem_gramer_rate):
    # 如果文件存在，则加载字典
    loaded_gramer_rate = np.load('./problem_Gramer_Rate_dict.npy', allow_pickle=True).item()
else:
    # 如果文件不存在，则运行函数 analyze_code_gramer_rate_quality函数进行计算
    analyze_code_gramer_rate_quality()
    loaded_gramer_rate = np.load('./problem_Gramer_Rate_dict.npy', allow_pickle=True).item()


#质量得分
# file_path_problem_quality_rate= './problem_Quality_Rate_dict.npy'
# if os.path.exists(file_path_problem_quality_rate):
#     # 如果文件存在，则加载字典
#     loaded_gramer_rate = np.load('./problem_Quality_Rate_dict.npy', allow_pickle=True).item()
# else:
#     # 如果文件不存在，则运行函数 analyze_code_gramer_rate_quality函数进行计算
#     evaluate_code_quality_and_save()
#     loaded_gramer_rate = np.load('./problem_Quality_Rate_dict.npy', allow_pickle=True).item()

# analyze_code_gramer_rate_quality()
# evaluate_code_quality_and_save()


# 加载.npy文件
# loaded_gramer_rate = np.load('./problem_Gramer_Rate_dict.npy', allow_pickle=True).item()
# loaded_quality_rate = np.load('./problem_Quality_Rate_dict.npy', allow_pickle=True).item()
# print(loaded_gramer_rate)

# Iterate over the DataFrame 添加对应的语法得分到data中 新列的名字叫gramerscore
for index, row in data.iterrows():
    submission_id = row['submission_id']
    if submission_id in problem:
        problem_key = problem[submission_id]
        if problem_key in loaded_gramer_rate:
            gramerscore = loaded_gramer_rate[problem_key]
            if gramerscore == -1:
                gramerscore =0
            # Add the gramerscore to a new column in the DataFrame
            data.at[index, 'gramerscore'] = gramerscore


#添加质量得分
# for index, row in data.iterrows():
#     submission_id = row['submission_id']
#     if submission_id in problem:
#         problem_key = problem[submission_id]
#         if problem_key in loaded_quality_rate:
#             qualityscore = loaded_quality_rate[problem_key]
#             # Add the gramerscore to a new column in the DataFrame
#             data.at[index, 'qualityscore'] = qualityscore

print(data.shape)
# 打印某一列中数字的最大值
max_value = data['gramerscore'].max()
print("gramerscore最大值：", max_value)

# 打印某一列中数字的最小值
min_value = data['gramerscore'].min()
print("gramerscore最小值：", min_value)


# 打印某一列中数字的最大值
max_value = data['cpu_time'].max()
print("cpu_time：", max_value)

# 打印某一列中数字的最小值
min_value = data['cpu_time'].min()
print("cpu_time：", min_value)

# 打印某一列中数字的最大值
max_value = data['memory'].max()
print("memory：", max_value)

# 打印某一列中数字的最小值
min_value = data['memory'].min()
print("memory：", min_value)

def parse_all_seq(students):
    all_sequences = []
    for student_id in tqdm(students, 'parse student sequence:\t'):
        student_sequence = parse_student_seq(data[data.user_id == student_id]) #data[data.user_id == student_id] 过滤出data.user_id 等于student_id的所有行
        all_sequences.extend([student_sequence]) #添加[student_sequence]到all_sequences
    return all_sequences

def parse_student_seq(student):
    seq = student.sort_values('date')
    q = [questions[q] for q in seq.problem_id.tolist()]  # 把每个question id 对应的 1 to (num_skill)的序号取出来，用连续的小数字表示
    p = [problem[p] for p in seq.submission_id.tolist()]  # 把每个problem id 对应的 1 to (num_problem)的序号取出来，用连续的小数字表示

    gramer_bool_list = seq.gramerscore
    # 将布尔值列表转换为整数列表
    # gramer = [int(x) for x in gramer_bool_list]
    gramer = 1

    # quality_bool_list = seq.qualityscore
    # # 将布尔值列表转换为整数列表
    # quality = [int(x) for x in quality_bool_list]

    cpu_bool_list = seq.cpu_time
    # 将布尔值列表转换为整数列表
    cpu = [int(x) for x in cpu_bool_list]

    memory_bool_list = seq.memory
    # 将布尔值列表转换为整数列表
    memory = [int(x) for x in memory_bool_list]

    a_bool_list = seq.status
    # 将布尔值列表转换为整数列表
    a = [int(x) for x in a_bool_list]
    return  p,q,gramer,cpu,memory, a

# [(question_sequence_0, answer_sequence_0), ..., (question_sequence_n, answer_sequence_n)]
sequences = parse_all_seq(data.user_id.unique())



def train_test_split(data, train_size=0.8, shuffle=True, seed=1025):
    random.seed(seed)
    if shuffle:
        random.shuffle(data)
    boundary = round(len(data) * train_size)
    return data[: boundary], data[boundary:]



train_sequences, test_sequences = train_test_split(sequences)


def sequences2tl(sequences, trgpath):
    with open(trgpath, 'w', encoding='utf8') as f:
        for seq in tqdm(sequences, 'write into file: '):
            problems, questions,gramers, cpus,memorys,answers = seq
            # seq_len = len(questions)
            # binary_numbers = [1 if element != '0' else 0 for element in questions]
            seq_len = np.count_nonzero(questions)
            # if(seq_len>=10):
            f.write(str(seq_len) + '\n')
            f.write(','.join([str(p) for p in problems]) + '\n')
            f.write(','.join([str(q) for q in questions]) + '\n')

            new_answers = answers[1:]  # 从第二个元素开始到最后一个元素
            new_answers.append(answers[0])  # 将第一个元素添加到新列表的末尾
            new_answers = [random.randint(30, 100) if a == 1 else random.randint(0, 100) for a in new_answers]
            f.write(','.join([str(a) for a in new_answers]) + '\n')

            # f.write(','.join([str(gramer) for gramer in gramers]) + '\n')
            # f.write(','.join([str(quality) for quality in qualitys]) + '\n')
            f.write(','.join([str(cpu) for cpu in cpus]) + '\n')
            f.write(','.join([str(memory) for memory in memorys]) + '\n')
            f.write(','.join([str(a) for a in answers]) + '\n')

# save triple line format for other tasks

def split_and_pad_records(sequences, max_step):
    result = []
    for triplet in sequences:
        questions, knowledge_points, gramers,cpus,memorys,answers = triplet
        for i in range(0, len(questions), max_step):
            sub_q = list(questions[i:i+max_step])
            sub_k = list(knowledge_points[i:i+max_step])
            # sub_gramer = list(gramers[i:i + max_step])
            sub_gramer = [1]
            # sub_quality = list(qualitys[i:i + max_step])

            sub_cpu = list(cpus[i:i + max_step])
            sub_memory = list(memorys[i:i + max_step])
            sub_a = list(answers[i:i+max_step])
            # pad the sub-record with 0 if its length is less than step
            if len(sub_q) < max_step:
                sub_q = list(np.pad(sub_q, (0, max_step-len(sub_q)), 'constant', constant_values=0))
                sub_k = list(np.pad(sub_k, (0, max_step-len(sub_k)), 'constant', constant_values=0))
                sub_gramer = list(np.pad(sub_gramer, (0, max_step - len(sub_gramer)), 'constant', constant_values=0))
                # sub_quality = list(np.pad(sub_quality, (0, max_step - len(sub_quality)), 'constant', constant_values=0))
                sub_cpu = list(np.pad(sub_cpu, (0, max_step - len(sub_cpu)), 'constant', constant_values=0))
                sub_memory = list(np.pad(sub_memory, (0, max_step - len(sub_memory)), 'constant', constant_values=0))
                sub_a = list(np.pad(sub_a, (0, max_step-len(sub_a)), 'constant', constant_values=0))
            result.append((sub_q, sub_k, sub_gramer, sub_cpu, sub_memory, sub_a))
    return result

sequences2tl(split_and_pad_records(train_sequences,MAX_STEP), './data/Afterprocess/train_codenet_C.txt')
sequences2tl(split_and_pad_records(test_sequences,MAX_STEP), './data/Afterprocess/test_codenet_C.txt')