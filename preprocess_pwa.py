import json
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util as sbert_util
import scipy.sparse as sp
from collections import defaultdict
import random


# 移除了 networkx, node2vec, 和 torch.nn.functional

# --- 新增种子设置函数 ---
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- 全局配置 ---
MASHUP_FILE_PATH = r"dataset1/PWA/mashupData.json"
API_FILE_PATH = r"dataset1/PWA/apiData.json"
OUTPUT_DIR = "dataset1/PWA_processed/"
RANDOM_STATE = 42
TEST_SIZE = 0.1
VAL_SIZE = 0.1
SBERT_MODEL_NAME = r"/home/528lab/wh/CF_Diff-main2/all-MiniLM-L6-v2"
CATEGORY_SIM_THRESHOLD = 0.5


# --- 1. 加载和解析原始数据 ---
def load_and_parse_data(mashup_file_path, api_file_path):
    """
    加载并解析Mashup和API的JSON数据文件。
    """
    print("开始加载和解析数据 (两阶段优化流程)...")
    with open(mashup_file_path, 'r', encoding='utf-8') as f:
        mashup_data_json = json.load(f)
    with open(api_file_path, 'r', encoding='utf-8') as f:
        api_data_json = json.load(f)

    print("阶段1: 预扫描Mashup，收集活跃API名称...")
    all_api_name_to_id_temp = {item.get("Name"): i for i, item in enumerate(api_data_json) if item.get("Name")}
    active_api_names = set()
    qualified_mashup_names = set()

    for mashup_item in mashup_data_json:
        name = mashup_item.get("Name")
        if not name: continue
        related_apis_str = mashup_item.get("Related APIs")
        if not related_apis_str: continue
        related_api_names = [api_name.strip() for api_name in related_apis_str.split(',') if api_name.strip()]
        unique_api_names_in_mashup = list(
            set([api_name for api_name in related_api_names if api_name in all_api_name_to_id_temp]))
        if len(unique_api_names_in_mashup) >= 3:
            qualified_mashup_names.add(name)
            active_api_names.update(unique_api_names_in_mashup)

    print(f"预扫描完成。找到 {len(qualified_mashup_names)} 个合格的Mashup。")
    print(f"这些Mashup共计使用了 {len(active_api_names)} 个唯一的活跃API。")

    print("阶段2: 正式加载数据并创建ID映射...")
    sorted_active_api_names = sorted(list(active_api_names))
    api_to_id = {name: i for i, name in enumerate(sorted_active_api_names)}
    id_to_api_name = {i: name for i, name in enumerate(sorted_active_api_names)}
    num_apis = len(api_to_id)

    sorted_qualified_mashup_names = sorted(list(qualified_mashup_names))
    mashup_to_id = {name: i for i, name in enumerate(sorted_qualified_mashup_names)}
    id_to_mashup_name = {i: name for i, name in enumerate(sorted_qualified_mashup_names)}
    num_mashups = len(mashup_to_id)

    api_descriptions = {}
    api_categories = {}
    for api_item in api_data_json:
        name = api_item.get("Name")
        if name in api_to_id:
            api_id = api_to_id[name]
            api_descriptions[api_id] = api_item.get("Description", "")
            categories = set()
            primary_cat = api_item.get("Primary Category")
            if primary_cat and primary_cat.strip(): categories.add(primary_cat.strip().lower())
            secondary_cats_str = api_item.get("Secondary Categories")
            if secondary_cats_str:
                for cat in secondary_cats_str.split(','):
                    if cat.strip(): categories.add(cat.strip().lower())
            api_categories[api_id] = categories

    mashup_descriptions, mashup_tags, interactions, mashup_api_relations_by_id = {}, {}, [], defaultdict(list)
    for mashup_item in mashup_data_json:
        name = mashup_item.get("Name")
        if name in mashup_to_id:
            mashup_id = mashup_to_id[name]
            mashup_descriptions[mashup_id] = mashup_item.get("Description", "")
            mashup_tags[mashup_id] = mashup_item.get("Tags", "")
            related_apis_str = mashup_item.get("Related APIs", "")
            related_api_names = [api_name.strip() for api_name in related_apis_str.split(',') if api_name.strip()]
            api_ids_for_this_mashup = list(
                set([api_to_id[api_name] for api_name in related_api_names if api_name in api_to_id]))
            for api_id in api_ids_for_this_mashup:
                interactions.append((mashup_id, api_id))
                mashup_api_relations_by_id[mashup_id].append(api_id)

    print(f"数据加载完成: {num_mashups} 个 Mashups, {num_apis} 个 APIs (只包含活跃实体).")
    print(f"共计 {len(interactions)} 个 Mashup-API 交互.")

    for m_id in range(num_mashups):
        if m_id not in mashup_descriptions: mashup_descriptions[m_id] = ""
        if m_id not in mashup_tags: mashup_tags[m_id] = ""
    for a_id in range(num_apis):
        if a_id not in api_descriptions: api_descriptions[a_id] = ""
        if a_id not in api_categories: api_categories[a_id] = set()

    return (mashup_to_id, id_to_mashup_name, mashup_descriptions, mashup_tags,
            mashup_api_relations_by_id,
            api_to_id, id_to_api_name, api_descriptions, api_categories,
            num_mashups, num_apis, interactions)


# --- 2. 数据集划分 ---
def split_interactions_data(interactions, num_mashups, test_size_ratio=0.1, val_size_ratio=0.1, random_state=42):
    print(
        f"开始按Mashup分层划分数据集 (目标比例 - 训练:{1 - test_size_ratio - val_size_ratio:.1f}, 验证:{val_size_ratio:.1f}, 测试:{test_size_ratio:.1f})...")
    if not interactions: raise ValueError("交互列表为空，无法划分。")
    # random.seed(random_state) # 已经在 main 中全局设置

    # 按Mashup分组
    mashup_groups = defaultdict(list)
    for mashup_id, api_id in interactions:
        mashup_groups[mashup_id].append(api_id)

    train_interactions, val_interactions, test_interactions = [], [], []

    # 由于预处理时已过滤，所有Mashup都至少有3个API
    for mashup_id, api_list in mashup_groups.items():
        random.shuffle(api_list)
        total_apis = len(api_list)

        # 按8:1:1比例分配，但保证测试和验证至少各有1个
        test_count = max(1, round(total_apis * test_size_ratio))
        val_count = max(1, round(total_apis * val_size_ratio))
        train_count = total_apis - test_count - val_count

        if train_count < 1:
            test_count = 1
            val_count = 1
            train_count = total_apis - 2

        # 分配交互
        for _ in range(test_count):
            test_interactions.append((mashup_id, api_list.pop()))
        for _ in range(val_count):
            val_interactions.append((mashup_id, api_list.pop()))
        for api_id in api_list:  # 剩余的给训练
            train_interactions.append((mashup_id, api_id))

    print(f"所有Mashup都有≥3个API（预处理时已过滤）")
    print(
        f"数据集划分完成: 训练集 {len(train_interactions)}, 验证集 {len(val_interactions)}, 测试集 {len(test_interactions)}")

    # 计算实际比例
    total_interactions = len(train_interactions) + len(val_interactions) + len(test_interactions)
    actual_train_ratio = len(train_interactions) / total_interactions
    actual_val_ratio = len(val_interactions) / total_interactions
    actual_test_ratio = len(test_interactions) / total_interactions
    print(f"实际比例 - 训练:{actual_train_ratio:.3f}, 验证:{actual_val_ratio:.3f}, 测试:{actual_test_ratio:.3f}")

    train_list_npy = np.array(train_interactions, dtype=np.int32) if train_interactions else np.array([],
                                                                                                      dtype=np.int32)
    valid_list_npy = np.array(val_interactions, dtype=np.int32) if val_interactions else np.array([], dtype=np.int32)
    test_list_npy = np.array(test_interactions, dtype=np.int32) if test_interactions else np.array([], dtype=np.int32)

    # 注意：这些邻接表在PWA中未使用，但在HGA中用于构建Adamic-Adar（旧版）
    # 为保持一致性，我们在此处返回空字典，因为新版CF-Diff特征在内部构建邻接表
    train_mashup_api_adj, train_api_mashup_adj = defaultdict(set), defaultdict(set)

    return train_mashup_api_adj, train_api_mashup_adj, train_list_npy, valid_list_npy, test_list_npy


# ==============================================================================
# === 新增：基于 CF-Diff 论文 (2404.14240v1) 的高阶特征构建函数 ===
# ==============================================================================
def build_cf_diff_structural_features(train_list_npy, num_users, num_items):
    """
    根据 CF-Diff 论文 (arXiv:2404.14240v1) 的 Section 3.1 逻辑，
    构建2跳 (U-U) 和 3跳 (U-I) 结构特征。

    这一个函数将同时生成替换分支一和分支二的特征。

    Args:
        train_list_npy (np.array): 训练集交互列表 (N, 2)。 (N_mashup, N_api)
        num_users (int): 用户 (Mashup) 的总数。
        num_items (int): 物品 (API) 的总数。

    Returns:
        torch.Tensor: u_u_2hop_tensor (分支一, shape: [num_users, num_users])
        torch.Tensor: u_i_3hop_tensor (分支二, shape: [num_users, num_items])
    """
    print("\n--- 开始构建 CF-Diff 论文的高阶结构特征 (2-hop U-U & 3-hop U-I) ---")

    if train_list_npy.size == 0:
        print("警告: 训练列表为空，返回全零特征。")
        u_u_2hop_tensor = torch.zeros((num_users, num_users), dtype=torch.float32)
        u_i_3hop_tensor = torch.zeros((num_users, num_items), dtype=torch.float32)
        return u_u_2hop_tensor, u_i_3hop_tensor

    # 1. 构建训练集的稀疏交互矩阵 R (User-Item)
    # R[i, j] = 1 如果 user i 和 item j 交互过
    R = sp.csr_matrix(
        (np.ones(len(train_list_npy)), (train_list_npy[:, 0], train_list_npy[:, 1])),
        shape=(num_users, num_items),
        dtype=np.float32
    )
    # R_T (Item-User)
    R_T = R.transpose().tocsr()

    # --- 2. 构建分支一 (Branch 1): 2-hop U-U 结构特征 (Mashup -> API -> Mashup) ---
    # 论文 逻辑: S_uu[i, k] = user i 和 user k 共同交互过的 item 数量
    print("1. 正在计算 S_uu = R.dot(R_T) ...")
    S_uu = R.dot(R_T).tocsr()

    # 论文 逻辑: 移除自环 (u^(2) 中不包含自己)
    S_uu.setdiag(0)
    S_uu.eliminate_zeros()

    # 论文 逻辑: 归一化 (Eq. 6 中 u^(2) 的归一化)
    # S_uu[i, :] 的行和 (Row sum) 即为 user i 的 2-hop 路径总数
    print("2. 正在对 S_uu (U-U 2-hop) 进行行归一化...")
    row_sums = S_uu.sum(axis=1)
    row_sums[row_sums == 0] = 1e-8  # 避免除以零

    # 创建 D_inv (对角矩阵)
    diag_indices = np.arange(num_users)
    D_inv_vals = 1.0 / np.array(row_sums).flatten()
    D_inv = sp.csr_matrix((D_inv_vals, (diag_indices, diag_indices)), shape=(num_users, num_users))

    # S_uu_normalized = D_inv * S_uu
    S_uu_normalized = D_inv.dot(S_uu)

    # 转换为 torch.Tensor
    u_u_2hop_tensor = torch.from_numpy(S_uu_normalized.toarray()).float()
    print(f"分支一 (U-U 2-hop) 构建完成, shape: {u_u_2hop_tensor.shape}")

    # --- 3. 构建分支二 (Branch 2): 3-hop U-I 结构特征 (Mashup -> API -> Mashup -> API) ---
    # 论文 逻辑: S_ui = u^(2) * R
    print("3. 正在计算 S_ui = S_uu_normalized.dot(R) ...")

    # S_ui_3hop[i, j] = sum_k( S_uu_norm[i, k] * R[k, j] )
    # 含义: user i 对 item j 的分数 = sum( user i 与 user k 的(2-hop)相似度 * user k 对 item j 的交互)
    S_ui_3hop = S_uu_normalized.dot(R)  # 已经是归一化的分数

    # 关键：屏蔽掉训练集中已有的交互，我们只关心对 *未交互* API 的推荐分数
    S_ui_3hop[R.nonzero()] = 0.0

    # 转换为 torch.Tensor
    u_i_3hop_tensor = torch.from_numpy(S_ui_3hop.toarray()).float()
    print(f"分支二 (U-I 3-hop) 构建完成, shape: {u_i_3hop_tensor.shape}")

    print("--- CF-Diff 高阶特征构建完毕 ---")

    return u_u_2hop_tensor, u_i_3hop_tensor


# ==============================================================================
# === 结束新增函数 ===
# ==============================================================================


# --- 3.B (分支四) 构建Mashup-Mashup文本相似性 (基于Description) ---
def build_mashup_text_similarity_features(mashup_descriptions_dict, num_mashups, sbert_model):
    print("开始构建Mashup-Mashup文本相似性特征...")
    ordered_descriptions = [mashup_descriptions_dict.get(i, "") for i in range(num_mashups)]
    print(f"使用SBERT模型 ({SBERT_MODEL_NAME}) 编码Mashup描述...")
    mashup_embeddings = sbert_model.encode(ordered_descriptions, show_progress_bar=True, convert_to_tensor=True)
    print(f"Mashup描述编码完成。嵌入维度: {mashup_embeddings.shape[1]}")
    similarity_matrix = sbert_util.pytorch_cos_sim(mashup_embeddings, mashup_embeddings)
    similarity_matrix = torch.nan_to_num(similarity_matrix, nan=0.0)
    for i in range(num_mashups):
        similarity_matrix[i, i] = 0.0
    print("Mashup-Mashup文本相似性矩阵构建完成。")
    return similarity_matrix.float()


# --- 3.D (分支三) 构建Mashup-API语义相似性 (基于Description) ---
def build_mashup_api_semantic_similarity_features(mashup_descriptions_dict, api_descriptions_dict, num_mashups,
                                                  num_apis, sbert_model):
    print("开始构建Mashup-API语义相似性特征...")
    ordered_mashup_descriptions = [mashup_descriptions_dict.get(i, "") for i in range(num_mashups)]
    ordered_api_descriptions = [api_descriptions_dict.get(i, "") for i in range(num_apis)]
    print(f"使用SBERT模型 ({SBERT_MODEL_NAME}) 编码Mashup描述...")
    mashup_embeddings = sbert_model.encode(ordered_mashup_descriptions, show_progress_bar=True, convert_to_tensor=True)
    print("Mashup描述编码完成。")
    print(f"使用SBERT模型 ({SBERT_MODEL_NAME}) 编码API描述...")
    api_embeddings = sbert_model.encode(ordered_api_descriptions, show_progress_bar=True, convert_to_tensor=True)
    print("API描述编码完成。")
    similarity_matrix = sbert_util.pytorch_cos_sim(mashup_embeddings, api_embeddings)
    similarity_matrix = torch.nan_to_num(similarity_matrix, nan=0.0)
    print("Mashup-API语义相似性矩阵构建完成。")
    return similarity_matrix.float()


# --- 主流程 ---
def main():
    # --- 新增：在main函数开头设置所有种子 ---
    set_seeds(RANDOM_STATE)

    print("--- 运行在完整预处理模式 (PWA) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    (mashup_to_id, id_to_mashup_name, mashup_descriptions, mashup_tags,
     mashup_api_relations_by_id, api_to_id, id_to_api_name, api_descriptions, api_categories,
     num_mashups, num_apis, interactions) = load_and_parse_data(MASHUP_FILE_PATH, API_FILE_PATH)

    train_mashup_api_adj, train_api_mashup_adj, train_list_npy, valid_list_npy, test_list_npy = \
        split_interactions_data(interactions, num_mashups, TEST_SIZE, VAL_SIZE, RANDOM_STATE)

    np.save(os.path.join(OUTPUT_DIR, f"train_list_PWA.npy"), train_list_npy)
    np.save(os.path.join(OUTPUT_DIR, f"valid_list_PWA.npy"), valid_list_npy)
    np.save(os.path.join(OUTPUT_DIR, f"test_list_PWA.npy"), test_list_npy)
    print(f"训练/验证/测试交互列表已保存到 {OUTPUT_DIR}")

    print(f"全局加载SBERT模型: {SBERT_MODEL_NAME}...")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    print("SBERT模型加载完毕。")

    print("\n--- 开始构建4个分支的特征 (CF-Diff 结构 + SBERT 语义) ---")

    # ==============================================================================
    # === 核心修改：用一个函数调用替换旧的分支1和分支2构建代码 ===
    # ==============================================================================

    # --- 新 B1 和 新 B2 (基于 CF-Diff 论文的 2-hop/3-hop 路径) ---
    # 替换掉旧的 build_node2vec_features(...)
    mashup_mashup_structural_tensor, contextual_api_structural_similarity_tensor = build_cf_diff_structural_features(
        train_list_npy, num_mashups, num_apis
    )

    # ==============================================================================
    # === 结束核心修改 ===
    # ==============================================================================

    # --- B4 (U-U 语义) (不变) ---
    mashup_text_similarity_tensor = build_mashup_text_similarity_features(mashup_descriptions, num_mashups, sbert_model)

    # --- B3 (U-I 语义) (不变) ---
    mashup_api_semantic_similarity_tensor = build_mashup_api_semantic_similarity_features(
        mashup_descriptions, api_descriptions, num_mashups, num_apis, sbert_model)

    # 转换并保存 (文件名保持不变，这样你的 main.py 不用改)
    tensors_to_save = {
        "features_2hop_mashup_PWA.pt": mashup_mashup_structural_tensor.cpu().float(),  # 新B1 (U-U 2-hop)
        "features_mashup_text_similarity_PWA.pt": mashup_text_similarity_tensor.cpu().float(),  # B4
        "features_contextual_api_api_similarity_PWA.pt": contextual_api_structural_similarity_tensor.cpu().float(),
        # 新B2 (U-I 3-hop)
        "features_mashup_api_complementarity_PWA.pt": mashup_api_semantic_similarity_tensor.cpu().float()  # B3
    }

    print("\n--- 特征维度和保存信息 (新版B1, B2) ---")
    for filename, tensor in tensors_to_save.items():
        print(f"Shape of {filename}: {tensor.shape}")
        path = os.path.join(OUTPUT_DIR, filename)
        torch.save(tensor, path)
        print(f"特征已保存到: {path}")

    # 保存映射和统计信息
    mappings = {
        "mashup_to_id": mashup_to_id, "id_to_mashup_name": id_to_mashup_name,
        "api_to_id": api_to_id, "id_to_api_name": id_to_api_name,
        "num_mashups": num_mashups, "num_apis": num_apis,
        "sbert_model_name": SBERT_MODEL_NAME,
        "feature_files": {
            "mashup_mashup_structural": "features_2hop_mashup_PWA.pt",
            "mashup_text_similarity": "features_mashup_text_similarity_PWA.pt",
            "contextual_api_structural_similarity": "features_contextual_api_api_similarity_PWA.pt",
            "mashup_api_semantic_similarity": "features_mashup_api_complementarity_PWA.pt"
        },
        # --- 修改：更新特征描述 ---
        "feature_descriptions": {
            "features_2hop_mashup_PWA.pt": "NEW (CF-Diff): U-U 2-hop Path Feature (Mashup -> API -> Mashup)",
            "features_contextual_api_api_similarity_PWA.pt": "NEW (CF-Diff): U-I 3-hop Path Feature (Mashup -> API -> Mashup -> API)",
            "features_mashup_api_complementarity_PWA.pt": "ORIGINAL: Mashup-API SBERT (U-I Semantic)",
            "features_mashup_text_similarity_PWA.pt": "ORIGINAL: Mashup-Mashup SBERT (U-U Semantic)"
        }
    }
    with open(os.path.join(OUTPUT_DIR, "mappings_and_stats_PWA.json"), 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=4)
    print(f"ID 映射关系和统计信息已更新并保存到 mappings_and_stats_PWA.json 于 {OUTPUT_DIR}")

    print("\nPWA 数据预处理全部完成！已生成4个分支的特征并保存：")
    print("  - 分支1: NEW (CF-Diff) Mashup-Mashup 2-hop Path")
    print("  - 分支2: NEW (CF-Diff) Mashup-API 3-hop Path")
    print("  - 分支3: Mashup-API SBERT (U-I Semantic)")
    print("  - 分支4: Mashup-Mashup SBERT (U-U Semantic)")


if __name__ == "__main__":
    main()