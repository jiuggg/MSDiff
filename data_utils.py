import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.sparse as sp


class MashupFeaturesDataset(Dataset):
    """
    为PWA数据集创建PyTorch数据集。
    每个样本包含一个Mashup的交互向量和所有相关的指导特征。
    """

    def __init__(self, train_interactions_dense,
                 mashup_mashup_structural_features,
                 contextual_api_api_similarity_features,
                 mashup_api_complementarity_features,
                 mashup_text_similarity_features,
                 num_mashups, num_apis):
        self.train_interactions_dense = train_interactions_dense
        self.mashup_mashup_structural_features = mashup_mashup_structural_features
        self.contextual_api_api_similarity_features = contextual_api_api_similarity_features
        self.mashup_api_complementarity_features = mashup_api_complementarity_features
        self.mashup_text_similarity_features = mashup_text_similarity_features

        # 存储维度以备不时之需
        self.num_mashups = num_mashups
        self.num_apis = num_apis

    def __len__(self):
        # 数据集的长度等于Mashup的数量
        return self.num_mashups

    def __getitem__(self, idx):
        # 对于给定的mashup_id (idx)，返回其所有特征
        mashup_id = idx

        interactions = self.train_interactions_dense[mashup_id]

        # 特征 1: Mashup-Mashup 结构化 (二跳)
        # 输入给模型前，每个Mashup需要一个(num_mashups,)维度的向量
        f1_mashup_struct = self.mashup_mashup_structural_features[mashup_id].cpu()

        # 特征 2: 上下文API-API相似度
        # 输入给模型前，每个Mashup需要一个(num_apis,)维度的向量
        f2_contextual_api_sim = self.contextual_api_api_similarity_features[mashup_id].cpu()

        # 特征 3: Mashup-API互补性
        # 输入给模型前，每个Mashup需要一个(num_apis,)维度的向量
        f3_api_compl = self.mashup_api_complementarity_features[mashup_id].cpu()

        # 特征 4: Mashup文本相似度
        # 输入给模型前，每个Mashup需要一个(num_mashups,)维度的向量
        f4_mashup_text_sim = self.mashup_text_similarity_features[mashup_id].cpu()

        return (
            torch.FloatTensor(interactions),  # (num_apis,)
            f1_mashup_struct,  # (num_mashups,)
            f2_contextual_api_sim,  # (num_apis,)
            f3_api_compl,  # (num_apis,)
            f4_mashup_text_sim  # (num_mashups,)
        )


def load_processed_mashup_data(path):
    """
    从指定路径加载所有PWA预处理好的数据文件。
    """
    print(f"从以下路径加载处理后的PWA Mashup数据: {path}")

    # 1. 加载映射和统计信息
    with open(os.path.join(path, "mappings_and_stats_PWA.json"), 'r') as f:
        mappings = json.load(f)
    n_mashup = mappings["num_mashups"]
    n_api = mappings["num_apis"]
    print(f"Mashup数量 (n_mashup): {n_mashup}, API数量 (n_api): {n_api}")

    # 2. 加载交互列表
    train_list = np.load(os.path.join(path, "train_list_PWA.npy"))
    valid_list = np.load(os.path.join(path, "valid_list_PWA.npy"))
    test_list = np.load(os.path.join(path, "test_list_PWA.npy"))
    print(f"加载的PWA交互数量: 训练集={len(train_list)}, 验证集={len(valid_list)}, 测试集={len(test_list)}")

    # 3. 构建稀疏和稠密交互矩阵
    # 训练数据 (稠密，用于输入模型)
    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]), (train_list[:, 0], train_list[:, 1])), dtype='float32',
                               shape=(n_mashup, n_api))
    train_data_dense = train_data.toarray()
    print(f"PWA训练集交互矩阵 (稠密) 构建完毕，形状: {train_data_dense.shape}")

    # 验证数据 (稀疏，用于评估)
    vad_data_sparse = sp.csr_matrix((np.ones_like(valid_list[:, 0]), (valid_list[:, 0], valid_list[:, 1])),
                                    dtype='float32', shape=(n_mashup, n_api))
    print(f"PWA验证集交互矩阵 (稀疏) 构建完毕，形状: {vad_data_sparse.shape}, 非零元素: {vad_data_sparse.nnz}")

    # 测试数据 (稀疏，用于评估)
    test_data_sparse = sp.csr_matrix((np.ones_like(test_list[:, 0]), (test_list[:, 0], test_list[:, 1])),
                                     dtype='float32', shape=(n_mashup, n_api))
    print(f"PWA测试集交互矩阵 (稀疏) 构建完毕，形状: {test_data_sparse.shape}, 非零元素: {test_data_sparse.nnz}")

    # 4. 加载特征文件
    feature_files = mappings["feature_files"]

    # 特征1: Mashup-Mashup Structural
    feature1_path = os.path.join(path, feature_files["mashup_mashup_structural"])
    feature1_user_user_struct_tensor = torch.load(feature1_path, map_location='cpu')
    print(
        f"成功加载特征 'mashup_mashup_structural': {feature_files['mashup_mashup_structural']}, 形状: {feature1_user_user_struct_tensor.shape}")

    # 特征2: Contextual API-API Similarity
    feature2_path = os.path.join(path, feature_files["contextual_api_structural_similarity"])
    feature2_item_item_struct_ctx_tensor = torch.load(feature2_path, map_location='cpu')
    print(
        f"成功加载特征 'contextual_api_structural_similarity': {feature_files['contextual_api_structural_similarity']}, 形状: {feature2_item_item_struct_ctx_tensor.shape}")

    # 特征3: Mashup-API Complementarity (在PWA中是类别共现或语义相似度)
    # 根据您最新的修改，这个文件现在存储的是 Mashup-API 语义相似度
    feature3_path = os.path.join(path, feature_files.get("mashup_api_semantic_similarity") or feature_files.get(
        "contextual_api_category_complementarity"))
    feature3_item_item_cat_ctx_tensor = torch.load(feature3_path, map_location='cpu')
    print(
        f"成功加载特征 'mashup_api_semantic_similarity': {os.path.basename(feature3_path)}, 形状: {feature3_item_item_cat_ctx_tensor.shape}")

    # 特征4: Mashup Text Similarity
    feature4_path = os.path.join(path, feature_files["mashup_text_similarity"])
    feature4_user_user_semantic_tensor = torch.load(feature4_path, map_location='cpu')
    print(
        f"成功加载特征 'mashup_text_similarity': {feature_files['mashup_text_similarity']}, 形状: {feature4_user_user_semantic_tensor.shape}")

    return (
        train_data_dense,
        feature1_user_user_struct_tensor,
        feature2_item_item_struct_ctx_tensor,
        feature3_item_item_cat_ctx_tensor,
        feature4_user_user_semantic_tensor,
        train_data,  # 训练数据的稀疏形式
        vad_data_sparse,
        test_data_sparse,
        n_mashup,
        n_api
    )


class AppFeaturesDataset(Dataset):
    """
    为HGA数据集创建PyTorch数据集。
    每个样本包含一个App的交互向量和所有相关的指导特征。
    """

    def __init__(self, train_interactions_dense,
                 app_app_structural_features,  # 对应特征1
                 contextual_api_api_structural_features,  # 对应特征2
                 contextual_api_api_category_cooccurrence_features,  # 对应特征3
                 app_category_similarity_features,  # 对应特征4
                 num_apps, num_apis):
        self.train_interactions_dense = train_interactions_dense
        self.app_app_structural_features = app_app_structural_features
        self.contextual_api_api_structural_features = contextual_api_api_structural_features
        self.contextual_api_api_category_cooccurrence_features = contextual_api_api_category_cooccurrence_features
        self.app_category_similarity_features = app_category_similarity_features
        self.num_apps = num_apps
        self.num_apis = num_apis

    def __len__(self):
        return self.num_apps

    def __getitem__(self, idx):
        app_id = idx

        interactions = self.train_interactions_dense[app_id]

        app_app_structural = self.app_app_structural_features[app_id].cpu() \
            if self.app_app_structural_features is not None else torch.zeros(self.num_apps)

        app_category_similarity = self.app_category_similarity_features[app_id].cpu() \
            if self.app_category_similarity_features is not None else torch.zeros(self.num_apps)

        contextual_api_api_structural = self.contextual_api_api_structural_features[app_id].cpu() \
            if self.contextual_api_api_structural_features is not None else torch.zeros(self.num_apis)

        contextual_api_api_category_cooccurrence = self.contextual_api_api_category_cooccurrence_features[app_id].cpu() \
            if self.contextual_api_api_category_cooccurrence_features is not None else torch.zeros(self.num_apis)

        return (
            torch.FloatTensor(interactions),
            app_app_structural,
            contextual_api_api_structural,
            contextual_api_api_category_cooccurrence,
            app_category_similarity
        )


def load_processed_hga_data(path):
    print(f"从以下路径加载处理后的HGA App数据: {path}")

    # 1. 加载映射和统计信息
    with open(os.path.join(path, "mappings_and_stats_HGA.json"), 'r') as f:
        mappings = json.load(f)
    n_app = mappings["num_apps"]
    n_api = mappings["num_apis"]
    print(f"App数量 (n_app): {n_app}, API数量 (n_api): {n_api}")

    # 2. 加载交互列表
    train_list = np.load(os.path.join(path, "train_list_HGA.npy"))
    valid_list = np.load(os.path.join(path, "valid_list_HGA.npy"))
    test_list = np.load(os.path.join(path, "test_list_HGA.npy"))
    print(f"加载的HGA交互数量: 训练集={len(train_list)}, 验证集={len(valid_list)}, 测试集={len(test_list)}")

    # 3. 构建稀疏和稠密交互矩阵
    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]), (train_list[:, 0], train_list[:, 1])), dtype='float32',
                               shape=(n_app, n_api))
    train_data_dense = train_data.toarray()
    print(f"HGA训练集交互矩阵 (稠密) 构建完毕，形状: {train_data_dense.shape}")

    vad_data_sparse = sp.csr_matrix((np.ones_like(valid_list[:, 0]), (valid_list[:, 0], valid_list[:, 1])),
                                    dtype='float32', shape=(n_app, n_api))
    print(f"HGA验证集交互矩阵 (稀疏) 构建完毕，形状: {vad_data_sparse.shape}, 非零元素: {vad_data_sparse.nnz}")

    test_data_sparse = sp.csr_matrix((np.ones_like(test_list[:, 0]), (test_list[:, 0], test_list[:, 1])),
                                     dtype='float32', shape=(n_app, n_api))
    print(f"HGA测试集交互矩阵 (稀疏) 构建完毕，形状: {test_data_sparse.shape}, 非零元素: {test_data_sparse.nnz}")

    # 4. 安全地加载特征文件
    feature_files = mappings.get("feature_files", {})

    def safe_load_feature(key, p, default_shape):
        if key in feature_files:
            try:
                feature_path = os.path.join(p, feature_files[key])
                tensor = torch.load(feature_path, map_location='cpu')
                print(f"成功加载特征 '{key}': {feature_files[key]}, 形状: {tensor.shape}")
                return tensor
            except FileNotFoundError:
                print(f"警告: 特征文件 '{feature_files[key]}' 在mappings.json中定义，但文件未找到。将使用零张量替代。")
                return torch.zeros(default_shape)
            except Exception as e:
                print(f"警告: 加载特征 '{key}' 时出错: {e}。将使用零张量替代。")
                return torch.zeros(default_shape)
        else:
            print(f"信息: 在mappings.json中未找到特征 '{key}' 的定义。将使用None。")
            return None

    # App-App Structural (Branch 1)
    feature1_user_user_struct_tensor = safe_load_feature(
        "app_app_structural", path, (n_app, n_app)
    )

    # Contextual API-API Structural (Branch 2)
    feature2_item_item_struct_ctx_tensor = safe_load_feature(
        "contextual_api_api_structural", path, (n_app, n_api)
    )

    # App-API Semantic Similarity (Branch 3)
    feature3_item_item_cat_ctx_tensor = safe_load_feature(
        "app_api_semantic_similarity", path, (n_app, n_api)
    )

    # App-App Text Similarity (Branch 4)
    feature4_user_user_semantic_tensor = safe_load_feature(
        "app_app_text_similarity", path, (n_app, n_app)
    )

    return (
        train_data_dense,
        feature1_user_user_struct_tensor,
        feature2_item_item_struct_ctx_tensor,
        feature3_item_item_cat_ctx_tensor,
        feature4_user_user_semantic_tensor,
        train_data,
        vad_data_sparse,
        test_data_sparse,
        n_app,
        n_api
    )
