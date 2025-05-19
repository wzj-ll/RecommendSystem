## Week1

### 数据处理

使用pandas加载数据时遇到的问题：

- 从.txt文件中加载数据时使用`pd.read_csv()`函数，并且由于数据本身没有表头，所以指定columns作为数据的表头，并且在参数中设置`header=None, names=columns`，以及一行数据中的分隔符`sep=\t`

  ```python
  columns1 = ['user_id', 'item_id', 'rating', 'timestamp']
  df1 = pd.read_csv("ml-100k/u.data", header=None, sep='\t', names=columns)
  ```

- 在处理ml-100k的u.item文件时，文件中的数据列较多，不知道每一列代表什么意思，查找数据集官方文档可以获取相关信息，或者直接设置为`unknown`，设置方式为：

  ```python
  columns3 = ['item_id', 'name', 'showtime', 'unknown1', 'link'] + [f"unknown2_{i}" for i in range(19)]
  ```

- 无法使用utf-8进行u.item文件的解码，解决方法是使用chardet库中的detect函数来解析文件使用的是什么编码方式

  ```python
  with open("ml-100k/u.item", 'rb') as f:
      result = chardet.detect(f.read())
  print(result)
  
  df3 = pd.read_csv("ml-100k/u.item", encoding=f'{result["encoding"]}', sep='|', header=None, names=columns3)
  ```

- 构建用户-物品评分矩阵时，可以从已有的u.data文件中提取评分的数据，使用`pd.pivot()`方法提取相关数据

  ```Python
  # 创建物品-用户评分矩阵
  item_user_matrix = ratings.pivot(index='item_id', columns='user_id', values='rating')
  ```

### 基于用户的协同过滤流程

首先计算用户的相似度，找出和目标用户最相似的top-N用户

然后根据最相似的用户对目标用户未交互过的物品的评分情况、以及用户相似度来计算目标用户对物品的评分，并最终生成推荐