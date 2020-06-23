# -hadoop-KNN-KMEANS
KNN-KMEANS-中山大学云计算实践课程设计
文件包含KNN文件夹以及KMEANS文件夹

1、KNN文件夹包含实现KNN算法的java代码（Knn_new.java），数据集（sick.csv），测试集（KnnTest_new.txt)<br>
   代码思路为：<br>
   （1）计算测试数据与各个训练数据之间的距离；<br>
   （2）按照距离的递增关系进行排序；<br>
   （3）选取距离最小的K个点；<br>
   （4）确定前K个点所在类别的出现频率；<br>
   （5）返回前K个点中出现频率最高的类别作为测试数据的预测分类。<br>
   详细过程参考代码注释<br>

2、KMEANS文件夹包含实现kmeans算法的java代码（KMeans.java），数据集（data.txt），初始中心点（oldcenter.txt），结果集（result.txt）<br>
   代码思路为：<br>
   （1） 从 n个数据对象任意选择 k 个对象作为初始聚类中心；<br>
   （2） 根据每个聚类对象的均值（中心对象），计算每个对象与这些中心对象的距离；并根据最小距离重新对相应对象进行划分（如下图中，我们可以看到A,B属于上面的种子点，C,D,E属于下面中部的种子点）；<br>
   （3） 重新计算每个（有变化）聚类的均值（中心对象）；<br>
   （4） 循环（2）到（3）直到每个聚类不再发生变化为止（如下图中，我们可以看到最终结果是上面的种子点聚合了A,B,C，下面的种子点聚合了D，E）<br>
    代码包含具体注释<br>
