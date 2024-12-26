from Num2CodeBert import num2Codevec
# str=num2Codevec(1726)
# print(str)

#
# import torch
#
# # # 假设您的 num2Codevec 函数如下，返回一个形状为 (768,) 的张量
# # def num2Codevec(num):
# #     # 假设这里是您的 num2Codevec 函数的实现
# #     # 返回一个形状为 (768,) 的张量
# #     return torch.randn(768)
#
# # 假设您有一个形状为 batch_size * seq_len 的张量 tensor
# tensor = torch.tensor([[1761, 1762, 1763, 1785, 1786, 1787],
#                        [1549, 1550, 1551, 0, 0, 0],
#                        [1916, 1917, 1911, 1932, 1911, 1933],
#                        [112, 29, 1088, 1108, 1109, 1110],
#                        [135, 136, 137, 0, 0, 0],
#                        [269, 270, 271, 294, 295, 296]])
#
# # 使用 torch.map() 将 num2Codevec 函数应用于 tensor 中的每个元素并直接堆叠
# codevecs = torch.stack([torch.stack([num2Codevec(int(num)) for num in row]) for row in tensor])
# print(codevecs)
# # 确保 codevecs 的形状正确
# print(codevecs.shape)

# from radon.complexity import cc_visit
# code = """
# print("hello")
# """
#
# cc = cc_visit(code)
# print(cc)
# complexity = max(cc)
# print(complexity[1])
import torch

import networkx as nx
import pickle

# 加载原始图
with open("questions_graph.gpickle", "rb") as f:
    G = pickle.load(f)

# 给定的知识点列表
q = [1405, 1408, 1409, 1410, 1406, 1411, 1412]

# 从列表q中提取唯一的知识点
unique_q = set(q)

# 构建子图：只包含列表q中的知识点及它们之间的关系
# 使用G.subgraph方法，这将返回一个子图，该子图包含unique_q中所有知识点及它们之间的边
subG = G.subgraph(unique_q).copy()
# 打印图中的节点和边
print("Nodes:", subG.nodes())
print("Edges:", subG.edges())

# 如果你需要，可以保存这个子图
# with open("sub_questions_graph.gpickle", "wb") as f:
#     pickle.dump(subG, f)

# 可以使用networkx的各种功能对subG进行分析或可视化等操作
in_vec = []
out_vec = []
# 计算除去0之外的元素个数
non_zero_sequence = [x for x in q if x != 0]
for i, node in enumerate(q):
    # 确保自己到自己也有一条边
    if node in G.nodes:
        in_vec.append(i)
        out_vec.append(i)
    for j in range(i + 1, len(non_zero_sequence)):
        if q[j] in G.nodes and G.has_edge(node, q[j]):
            in_vec.append(i)
            out_vec.append(j)

print(in_vec)
print(out_vec)
