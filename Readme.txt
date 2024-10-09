运行步骤：
1.Find_file_from_alldata.py(如果没有从Project_codenet中提取所需要语言类型的所有数据，就要先运行这个 记得该对应的文件夹名字)

2 .data_CodeNet_Python_preprocessing.py  (根据数据量的大小，在代码质量得分评价阶段会很慢，功能很多 ，最后生成tl格式的文件)

3.Num2Embedding_codenet_python.py  （根据需要 选择不同的地方开启注释  根据数据量的大小，在代码质量得分评价阶段会很慢，负责生成源代码对应的embedding表示，并存储在硬盘上，方便调用，不然训练的时候在生产，太慢了）

3.1  Findembedding.py  主要负责从生成好的embedding中查找需要的 目的是为了快速   不用单独运行 会被调用执行num2codeemb, num2skillemb

4.Question_Similarity_Graph.py   寻找知识点之间的相似性，并构建知识点相似关系图

5.DKT_emb_main.py (模型调用主函数)
