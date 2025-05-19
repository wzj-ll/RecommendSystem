import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from utils import compute_user_similarity

def user_based_cf(ratings:pd.DataFrame, target_user_id, n_neighbors:int=20, n_recommendations:int=10):
    # 1. 检查目标用户是否存在
    if target_user_id not in ratings['user_id'].unique():
        return "The target user is not exist"
    # 2. 构建用户-物品评分矩阵，以及用户相似度矩阵
    user_item_matrix = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    # user_similarity_df = compute_user_similarity(ratings)
    # print(user_similarity_df)

    # 3. 找到目标用户的n个最相似的用户
    similar_users = user_similarity_df.loc[target_user_id].sort_values(ascending=False)
    similar_users = similar_users.drop(target_user_id)
    print(similar_users)
    top_neighbors = similar_users.head(n_neighbors)

    # 4. 计算用户平均评分
    user_means = ratings.groupby('user_id')['rating'].mean()
    
    # 5. 获取目标用户已评分的物品
    target_user_rated_items = set(ratings[ratings['user_id'] == target_user_id]['item_id'])
    target_user_mean = user_means[target_user_id]

    # 所有物品的集合
    all_items = set(ratings['item_id'].unique())
    # 目标用户未评分的物品
    unrated_items = all_items - target_user_rated_items

    # 存储预测评分
    predictions = {}

    for item_id in unrated_items:
        if item_id not in user_item_matrix.columns:
            continue

        numerator = 0
        denominator = 0
        # 遍历邻居用户
        for neighbor_id, similarity in top_neighbors.items():
            # 如果相似度为0或者邻居用户未评价过该物品，则跳过
            if similarity <= 0 or user_item_matrix.loc[neighbor_id, item_id] == 0:
                continue
            # 计算邻居用户的评分偏差
            neighbor_mean = user_means[neighbor_id]
            rating_deviation = user_item_matrix.loc[neighbor_id, item_id] - neighbor_mean

            # 累加分子和分母
            numerator += similarity * rating_deviation
            denominator += abs(similarity)
        if denominator == 0:
            continue
        prediction = target_user_mean + numerator / denominator
        predictions[item_id] = prediction
    
    # key代表排序的键
    recommended_items = heapq.nlargest(n_recommendations, predictions.items(), key=lambda x : x[1])
    return recommended_items


columns = ['user_id', 'item_id', 'rating', 'timestamp']
interaction = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=columns)

