import json
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util as sbert_util
import scipy.sparse as sp
from collections import defaultdict
import random


# --- 新增种子设置函数 ---
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- 全局配置 ---
APP_FILE_PATH = r"dataset1/HGA/app.json"
API_FILE_PATH = r"dataset1/HGA/api.json"
OUTPUT_DIR = "dataset1/HGA_processed/"
RANDOM_STATE = 42
TEST_SIZE = 0.1
VAL_SIZE = 0.1

# SBERT模型 (中文)
SBERT_MODEL_NAME = r"/home/528lab/wh/CF_Diff-main2/all-MiniLM-L6-v2"


# --- 1. 加载和解析原始数据 ---
def load_and_parse_data(app_file_path, api_file_path):
    """
    加载并解析App和API的JSON数据文件 (兼容 Dict 和 List 格式)。
    """
    print(f"正在加载数据文件:\n  APP: {app_file_path}\n  API: {api_file_path}")

    # 1. 加载 APP 数据
    with open(app_file_path, 'r', encoding='utf-8') as f:
        app_data_raw = json.load(f)

    # 兼容性处理：List 直接用，Dict 取 rows
    if isinstance(app_data_raw, list):
        app_data_json_rows = app_data_raw
        print(f"  识别到 App 数据为 List 格式，共 {len(app_data_json_rows)} 条")
    elif isinstance(app_data_raw, dict):
        app_data_json_rows = app_data_raw.get("rows", app_data_raw.get("data", list(app_data_raw.values())))
        print(f"  识别到 App 数据为 Dict 格式，提取后共 {len(app_data_json_rows)} 条")
    else:
        raise ValueError(f"App 数据格式错误: {type(app_data_raw)}")

    # 2. 加载 API 数据
    with open(api_file_path, 'r', encoding='utf-8') as f:
        api_data_raw = json.load(f)

    # 兼容性处理
    if isinstance(api_data_raw, list):
        api_data_json_rows = api_data_raw
        print(f"  识别到 API 数据为 List 格式，共 {len(api_data_json_rows)} 条")
    elif isinstance(api_data_raw, dict):
        api_data_json_rows = api_data_raw.get("rows", api_data_raw.get("data", list(api_data_raw.values())))
        print(f"  识别到 API 数据为 Dict 格式，提取后共 {len(api_data_json_rows)} 条")
    else:
        raise ValueError(f"API 数据格式错误: {type(api_data_raw)}")

    # --- 阶段 1: 预扫描，筛选App，并收集活跃API ---
    print("阶段1: 预扫描App，收集活跃API名称...")

    all_api_names_in_dataset = {item.get("name").strip() for item in api_data_json_rows if
                                item.get("name") and item.get("name").strip()}

    temp_app_info = defaultdict(lambda: {"apis": set(), "categories": set(), "description": ""})
    for app_item in app_data_json_rows:
        name = app_item.get("name")
        if not name or not name.strip(): continue
        name = name.strip()

        category = app_item.get("category")
        if category and category.strip():
            temp_app_info[name]["categories"].add(category.strip())

        description = app_item.get("Description")
        if description and description.strip() and not temp_app_info[name]["description"]:
            temp_app_info[name]["description"] = description.strip()

        related_apis_str = app_item.get("relatedAPIs")
        if not related_apis_str or not related_apis_str.strip():
            continue

        related_api_names = {api_name.strip() for api_name in related_apis_str.split(',') if api_name.strip()}
        valid_api_names = {api_name for api_name in related_api_names if api_name in all_api_names_in_dataset}
        temp_app_info[name]["apis"].update(valid_api_names)

    active_api_names = set()
    qualified_app_names = set()

    for name, data in temp_app_info.items():
        if len(data["apis"]) >= 3:
            qualified_app_names.add(name)
            active_api_names.update(data["apis"])

    print(f"预扫描完成。找到 {len(qualified_app_names)} 个合格的App。")
    print(f"这些App共计使用了 {len(active_api_names)} 个唯一的活跃API。")

    # --- 阶段 2: 基于活跃实体，正式加载所有数据和创建映射 ---
    print("阶段2: 正式加载数据并创建ID映射...")

    sorted_active_api_names = sorted(list(active_api_names))
    api_to_id = {name: i for i, name in enumerate(sorted_active_api_names)}
    id_to_api_name = {i: name for i, name in enumerate(sorted_active_api_names)}
    num_apis = len(api_to_id)

    sorted_qualified_app_names = sorted(list(qualified_app_names))
    app_to_id = {name: i for i, name in enumerate(sorted_qualified_app_names)}
    id_to_app_name = {i: name for i, name in enumerate(sorted_qualified_app_names)}
    num_apps = len(app_to_id)

    app_categories = {}
    app_descriptions = {}
    for name, data in temp_app_info.items():
        if name in app_to_id:
            app_id = app_to_id[name]
            app_categories[app_id] = " ".join(sorted(data["categories"])) if data["categories"] else ""
            app_descriptions[app_id] = data.get("description", "")

    api_categories = {}
    api_descriptions = {}
    for api_item in api_data_json_rows:
        name = api_item.get("name")
        if name and name.strip() in api_to_id:
            api_id = api_to_id[name.strip()]
            category = api_item.get("category", "")
            api_categories[api_id] = category.strip() if category else ""
            description = api_item.get("Description", "")
            api_descriptions[api_id] = description.strip() if description else ""

    interactions = []
    app_api_relations_by_id = defaultdict(list)

    for name, data in temp_app_info.items():
        if name in app_to_id:
            app_id = app_to_id[name]
            api_ids_for_this_app = {api_to_id[api_name] for api_name in data["apis"] if api_name in api_to_id}
            for api_id in api_ids_for_this_app:
                interactions.append((app_id, api_id))
                app_api_relations_by_id[app_id].append(api_id)

    for app_id in range(num_apps):
        if app_id not in app_categories:
            app_categories[app_id] = ""
        if app_id not in app_descriptions:
            app_descriptions[app_id] = ""
    for api_id in range(num_apis):
        if api_id not in api_categories:
            api_categories[api_id] = ""
        if api_id not in api_descriptions:
            api_descriptions[api_id] = ""

    print(f"HGA数据加载完成: {num_apps} 个 Apps, {num_apis} 个 APIs (只包含活跃实体).")
    print(f"共计 {len(interactions)} 个 App-API 交互.")

    return (app_to_id, id_to_app_name, app_categories, app_descriptions,
            app_api_relations_by_id,
            api_to_id, id_to_api_name, api_categories, api_descriptions,
            num_apps, num_apis, interactions)


# --- 2. 数据集划分 (采用分层策略) ---
def split_interactions_data(interactions, num_entities, test_size_ratio=0.1, val_size_ratio=0.1, random_state=42,
                            entity_name="App"):
    """
    将交互数据划分为训练集、验证集、测试集。
    采用按App分层策略，按8:1:1比例分配，但保证测试集和验证集都有数据。
    """
    print(
        f"开始按{entity_name}分层划分数据集 (目标比例 - 训练:{1 - test_size_ratio - val_size_ratio:.1f}, 验证:{val_size_ratio:.1f}, 测试:{test_size_ratio:.1f})...")
    if not interactions:
        raise ValueError("交互列表为空，无法划分。")

    entity_groups = defaultdict(list)
    for entity_id, api_id in interactions:
        entity_groups[entity_id].append(api_id)

    train_interactions, val_interactions, test_interactions = [], [], []

    for entity_id, api_list in entity_groups.items():
        random.shuffle(api_list)
        total_apis = len(api_list)

        test_count = max(1, round(total_apis * test_size_ratio))
        val_count = max(1, round(total_apis * val_size_ratio))
        train_count = total_apis - test_count - val_count

        if train_count < 1:
            test_count = 1
            val_count = 1
            train_count = total_apis - 2

        for _ in range(test_count):
            test_interactions.append((entity_id, api_list.pop()))
        for _ in range(val_count):
            val_interactions.append((entity_id, api_list.pop()))
        for api_id in api_list:
            train_interactions.append((entity_id, api_id))

    print(f"所有{entity_name}都有≥3个API（预处理时已过滤）")
    print(
        f"HGA数据集划分完成: 训练集 {len(train_interactions)}, 验证集 {len(val_interactions)}, 测试集 {len(test_interactions)}")

    total_interactions = len(train_interactions) + len(val_interactions) + len(test_interactions)
    actual_train_ratio = len(train_interactions) / total_interactions
    actual_val_ratio = len(val_interactions) / total_interactions
    actual_test_ratio = len(test_interactions) / total_interactions
    print(f"实际比例 - 训练:{actual_train_ratio:.3f}, 验证:{actual_val_ratio:.3f}, 测试:{actual_test_ratio:.3f}")

    train_list_npy = np.array(train_interactions, dtype=np.int32) if train_interactions else np.array([],
                                                                                                      dtype=np.int32)
    valid_list_npy = np.array(val_interactions, dtype=np.int32) if val_interactions else np.array([], dtype=np.int32)
    test_list_npy = np.array(test_interactions, dtype=np.int32) if test_interactions else np.array([], dtype=np.int32)

    train_app_api_adj = defaultdict(set)
    train_api_app_adj = defaultdict(set)

    return train_app_api_adj, train_api_app_adj, train_list_npy, valid_list_npy, test_list_npy


# ==============================================================================
# === 新增：基于 CF-Diff 论文 (2404.14240v1) 的高阶特征构建函数 ===
# ==============================================================================
def build_cf_diff_structural_features(train_list_npy, num_users, num_items):
    """
    根据 CF-Diff 论文 (arXiv:2404.14240v1) 的 Section 3.1 逻辑，
    构建2跳 (U-U) 和 3跳 (U-I) 结构特征。

    这一个函数将同时生成替换分支一和分支二的特征。

    Args:
        train_list_npy (np.array): 训练集交互列表 (N, 2)。 (N_app, N_api)
        num_users (int): 用户 (Mashup/App) 的总数。
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

    # --- 2. 构建分支一 (Branch 1): 2-hop U-U 结构特征 (App -> API -> App) ---
    # 论文 逻辑: S_uu[i, k] = user i 和 user k 共同交互过的 item 数量
    print("1. 正在计算 S_uu = R.dot(R_T) ...")
    S_uu = R.dot(R_T).tocsr()

    # 论文 逻辑: 移除自环 (u^(2) 中不包含自己)
    S_uu.setdiag(0)
    S_uu.eliminate_zeros()

    # 论文 逻辑: 归一化 (Eq. 6 中 u^(2) 的归一化)
    # N_h-1,h 是 (h-1)-hop 和 h-hop 间的总交互数
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

    # --- 3. 构建分支二 (Branch 2): 3-hop U-I 结构特征 (App -> API -> App -> API) ---
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


# --- 3.C (分支三) 构建App-API语义相似性 (基于Description的BERT) ---
def build_app_api_semantic_similarity_features(app_descriptions_dict, api_descriptions_dict, num_apps, num_apis,
                                               sbert_model):
    """
    基于App和API的Description，使用BERT构建App-API语义相似性特征。
    """
    print("开始构建App-API语义相似性特征 (基于Description的BERT)...")

    ordered_app_descriptions = [app_descriptions_dict.get(i, "") for i in range(num_apps)]
    ordered_api_descriptions = [api_descriptions_dict.get(i, "") for i in range(num_apis)]

    print(f"使用SBERT模型 ({SBERT_MODEL_NAME}) 编码App Descriptions...")
    app_embeddings = sbert_model.encode(ordered_app_descriptions, show_progress_bar=True, convert_to_tensor=True)
    print("App Descriptions编码完成。")

    print(f"使用SBERT模型 ({SBERT_MODEL_NAME}) 编码API Descriptions...")
    api_embeddings = sbert_model.encode(ordered_api_descriptions, show_progress_bar=True, convert_to_tensor=True)
    print("API Descriptions编码完成。")

    similarity_matrix = sbert_util.pytorch_cos_sim(app_embeddings, api_embeddings)
    similarity_matrix = torch.nan_to_num(similarity_matrix, nan=0.0)

    print("App-API语义相似性矩阵构建完成。")
    return similarity_matrix.float()


# --- 3.D (分支四) 构建App-App文本相似性 (基于Description的BERT) ---
def build_app_app_text_similarity_features(app_descriptions_dict, num_apps, sbert_model):
    """
    基于App的Description，使用BERT构建App-App文本相似性特征。
    """
    print("开始构建App-App文本相似性特征 (基于Description的BERT)...")

    ordered_app_descriptions = [app_descriptions_dict.get(i, "") for i in range(num_apps)]

    print(f"使用SBERT模型 ({SBERT_MODEL_NAME}) 编码App Descriptions...")
    app_embeddings = sbert_model.encode(ordered_app_descriptions, show_progress_bar=True, convert_to_tensor=True)
    print(f"App Descriptions编码完成。嵌入维度: {app_embeddings.shape[1]}")

    similarity_matrix = sbert_util.pytorch_cos_sim(app_embeddings, app_embeddings)
    similarity_matrix = torch.nan_to_num(similarity_matrix, nan=0.0)

    for i in range(num_apps):
        similarity_matrix[i, i] = 0.0

    print("App-App文本相似性矩阵构建完成。")
    return similarity_matrix.float()


# --- 主流程 ---
def main():
    set_seeds(RANDOM_STATE)

    print("--- 运行在完整预处理模式 (HGA) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载和解析HGA数据
    (app_to_id, id_to_app_name, app_categories, app_descriptions,
     _app_api_relations_by_id,
     api_to_id, id_to_api_name, api_categories, api_descriptions,
     num_apps, num_apis, interactions) = load_and_parse_data(APP_FILE_PATH, API_FILE_PATH)

    if num_apps == 0 or not interactions:
        print("警告: 过滤后没有App或交互数据。将生成空的或占位符的特征文件。")
        train_list_npy = np.array([], dtype=np.int32)
        valid_list_npy = np.array([], dtype=np.int32)
        test_list_npy = np.array([], dtype=np.int32)
    else:
        # 2. 划分数据集
        _, _, train_list_npy, valid_list_npy, test_list_npy = \
            split_interactions_data(interactions, num_apps, TEST_SIZE, VAL_SIZE, RANDOM_STATE, entity_name="App")

    np.save(os.path.join(OUTPUT_DIR, f"train_list_HGA.npy"), train_list_npy)
    np.save(os.path.join(OUTPUT_DIR, f"valid_list_HGA.npy"), valid_list_npy)
    np.save(os.path.join(OUTPUT_DIR, f"test_list_HGA.npy"), test_list_npy)
    print(f"HGA训练/验证/测试交互列表已保存到 {OUTPUT_DIR}")

    # 加载SBERT模型（用于分支3和分支4）
    print(f"加载SBERT模型: {SBERT_MODEL_NAME}...")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    print("SBERT模型加载完毕。")

    print("\n--- 开始构建4个分支的特征 (CF-Diff 结构 + SBERT 语义) ---")

    # ==============================================================================
    # === 核心修改：用一个函数调用替换旧的分支1和分支2构建代码 ===
    # ==============================================================================

    # --- 新 B1 和 新 B2 (基于 CF-Diff 论文的 2-hop/3-hop 路径) ---
    app_app_structural_tensor, contextual_api_structural_similarity_tensor = build_cf_diff_structural_features(
        train_list_npy, num_apps, num_apis
    )

    # ==============================================================================
    # === 结束核心修改 ===
    # ==============================================================================

    # --- B4 (U-U 语义) (不变) ---
    app_app_text_similarity_tensor = build_app_app_text_similarity_features(
        app_descriptions, num_apps, sbert_model)

    # --- B3 (U-I 语义) (不变) ---
    app_api_semantic_similarity_tensor = build_app_api_semantic_similarity_features(
        app_descriptions, api_descriptions, num_apps, num_apis, sbert_model)

    # --- CPU转换和维度打印 ---
    app_app_structural_tensor = app_app_structural_tensor.cpu().float()
    contextual_api_structural_similarity_tensor = contextual_api_structural_similarity_tensor.cpu().float()
    app_api_semantic_similarity_tensor = app_api_semantic_similarity_tensor.cpu().float()
    app_app_text_similarity_tensor = app_app_text_similarity_tensor.cpu().float()

    # 保存所有特征文件 (文件名保持不变，兼容下游代码)
    tensors_to_save = {
        "features_app_app_structural_HGA.pt": app_app_structural_tensor,
        "features_contextual_api_api_structural_HGA.pt": contextual_api_structural_similarity_tensor,
        "features_app_api_semantic_similarity_HGA.pt": app_api_semantic_similarity_tensor,
        "features_app_app_text_similarity_HGA.pt": app_app_text_similarity_tensor
    }

    print("\n--- HGA特征维度和保存信息 (新版B1, B2) ---")
    for filename, tensor in tensors_to_save.items():
        print(f"Shape of {filename}: {tensor.shape}")
        path = os.path.join(OUTPUT_DIR, filename)
        torch.save(tensor, path)
        print(f"特征已保存到: {path}")

    # 保存映射和统计信息
    mappings = {
        "app_to_id": app_to_id,
        "id_to_app_name": id_to_app_name,
        "api_to_id": api_to_id,
        "id_to_api_name": id_to_api_name,
        "num_apps": num_apps,
        "num_apis": num_apis,
        "sbert_model_name": SBERT_MODEL_NAME,
        "feature_files": {
            "app_app_structural": "features_app_app_structural_HGA.pt",
            "contextual_api_api_structural": "features_contextual_api_api_structural_HGA.pt",
            "app_api_semantic_similarity": "features_app_api_semantic_similarity_HGA.pt",
            "app_app_text_similarity": "features_app_app_text_similarity_HGA.pt"
        },
        # --- 修改：更新特征描述 ---
        "feature_descriptions": {
            "features_app_app_structural_HGA.pt": "NEW (CF-Diff): U-U 2-hop Path Feature (App -> API -> App)",
            "features_contextual_api_api_structural_HGA.pt": "NEW (CF-Diff): U-I 3-hop Path Feature (App -> API -> App -> API)",
            "features_app_api_semantic_similarity_HGA.pt": "ORIGINAL: App-API SBERT on Description (U-I Semantic)",
            "features_app_app_text_similarity_HGA.pt": "ORIGINAL: App-App SBERT on Description (U-U Semantic)"
        },
        "interactions_stats": {
            "final_total_interactions_after_app_filtering": len(interactions),
            "train_interactions": len(train_list_npy),
            "valid_interactions": len(valid_list_npy),
            "test_interactions": len(test_list_npy)
        }
    }

    with open(os.path.join(OUTPUT_DIR, "mappings_and_stats_HGA.json"), 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=4, ensure_ascii=False)
    print(f"HGA的ID映射关系和统计信息已更新并保存到 mappings_and_stats_HGA.json 于 {OUTPUT_DIR}")

    print("\nHGA数据预处理全部完成！已生成4个分支的特征并保存：")
    print("  - 分支1: NEW (CF-Diff) App-App 2-hop Path")
    print("  - 分支2: NEW (CF-Diff) App-API 3-hop Path")
    print("  - 分支3: App-API SBERT on Description (U-I Semantic)")
    print("  - 分支4: App-App SBERT on Description (U-U Semantic)")
    print(f"使用的SBERT模型: {SBERT_MODEL_NAME}")


if __name__ == "__main__":
    main()