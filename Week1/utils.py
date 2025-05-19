import pandas as pd
import numpy as np

def compute_user_similarity(ratings:pd.DataFrame, min_common_items:int=5) -> pd.DataFrame:
    # 创建用户-物品评分矩阵
    user_item_matrix = ratings.pivot(index='user_id', columns='item_id', values='rating')
    print(f"The shape of user_item_matrix: {user_item_matrix.shape}")

    # 初始化用户相似度矩阵
    n_users = user_item_matrix.shape[0]
    user_similarity = np.zeros((n_users, n_users))
    user_ids = user_item_matrix.index.to_list()

    # 计算用户之间的相似度
    for i in range(n_users):
        for j in range(i+1, n_users):
            user_id1 = user_ids[i]
            user_id2 = user_ids[j]

            # 获取两个用户的物品评分集合
            user1_ratings = user_item_matrix.loc[user_id1]
            user2_ratings = user_item_matrix.loc[user_id2]
            common_items = user1_ratings.dropna().index.intersection(user2_ratings.dropna().index)

            if len(common_items) < min_common_items:
                similarity = 0
            else:
                user1_common = user1_ratings[common_items].values
                user2_common = user2_ratings[common_items].values
                # 计算范数（计算平方和，然后开方）
                norm1 = np.linalg.norm(user1_common)
                norm2 = np.linalg.norm(user2_common)
                if norm1 == 0 or norm2 == 0:
                    similarity = 0
                else:
                    similarity = np.dot(user1_common, user2_common) / (norm1 * norm2)
            user_similarity[user_id1 - 1][user_id2 - 1] = similarity
            user_similarity[user_id2 - 1][user_id1 - 1] = similarity
    # 创建用户相似度的 DataFrame
    user_similarity_df = pd.DataFrame(user_similarity, index=user_ids, columns=user_ids)
    return user_similarity_df

def compute_item_similarity(ratings:pd.DataFrame, min_common_users:int=5) -> pd.DataFrame:
    # 创建物品-用户评分矩阵
    item_user_matrix = ratings.pivot(index='item_id', columns='user_id', values='rating')
    print(f"The shape of item_user_matrix: {item_user_matrix.shape}")

    # 初始化物品相似度矩阵
    n_items = item_user_matrix.shape[0]
    item_similarity = np.zeros((n_items, n_items))
    item_ids = item_user_matrix.index.to_list()

    for i in range(n_items):
        for j in range(i+1, n_items):
            item1_id = item_ids[i]
            item2_id = item_ids[j]

            # 获取两个物品的用户评分
            item1_ratings = item_user_matrix.loc[item1_id]
            item2_ratings = item_user_matrix.loc[item2_id]

            # 找到对两个物品都评分的用户
            common_users = item1_ratings.dropna().index.intersection(item2_ratings.dropna().index)

            if len(common_users) < min_common_users:
                similarity = 0
            else:
                # 提取共同评分
                item1_common = item1_ratings[common_users].values
                item2_common = item2_ratings[common_users].values

                norm1 = np.linalg.norm(item1_common)
                norm2 = np.linalg.norm(item2_common)
                if norm1 == 0 or norm2 == 0:
                    similarity = 0
                else:
                    similarity = np.dot(item1_ratings, item2_ratings) / (norm1 * norm2)
            item_similarity[item1_id][item2_id] = similarity
            item_similarity[item2_id][item1_id] = similarity

    item_similarity_df = pd.DataFrame(item_similarity, index=item_ids, columns=item_ids)
    return item_similarity_df

