import torch
import torch.nn as nn
import numpy as np

# 假设有以下知识点集合和Q矩阵
knowledge_points = ["calculus", "trigonometry", "geography"]
knowledge_vectors = {kp: i for i, kp in enumerate(knowledge_points)}

# 创建Q矩阵
Q_matrix = np.array([
    [1, 1, 0],  # 对应第一个题目的知识点向量
    [0, 0, 1]   # 对应第二个题目的知识点向量
])

# 将Q矩阵转换为Tensor
Q_tensor = torch.tensor(Q_matrix, dtype=torch.long)

# 定义嵌入层
embedding_dim = 50  # 嵌入维度，可以根据需要调整
embedding_layer = nn.Embedding(num_embeddings=len(knowledge_points), embedding_dim=embedding_dim)

# 获取知识点嵌入
knowledge_embeddings = embedding_layer(Q_tensor)

# 将嵌入层的输出展平
knowledge_embeddings_flat = knowledge_embeddings.view(Q_tensor.size(0), -1)

# 输出结果
print(f"Q矩阵:\n{Q_tensor}")
print(f"嵌入后的知识点向量:\n{knowledge_embeddings_flat}")

# 如果需要，可以将结果转回numpy
knowledge_embeddings_numpy = knowledge_embeddings_flat.detach().numpy()
print(f"嵌入后的知识点向量（numpy格式）:\n{knowledge_embeddings_numpy}")
