import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import models.gaussian_diffusion as gd
from models.CAM_AE_multihops import CAM_AE_multihops
import evaluate_utils
import data_utils
import random
import sys
from datetime import datetime

# --- 新增：导入 AUC 计算库 ---
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("警告: 未找到 scikit-learn (sklearn)。AUC 指标将不可用。")
    print("请运行 'pip install scikit-learn' 来安装它。")
    roc_auc_score = None


# ---------------------------


# --- Tee类：同时输出到控制台和文件 ---
class Tee:
    """将输出同时重定向到控制台和文件"""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # 立即写入文件，避免缓冲

    def flush(self):
        for f in self.files:
            f.flush()


# --- Helper for boolean argparse ---
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# --- 基本设置 ---
random_seed = 1
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_early_stop_metric(valid_results, metric_mode='combined_four', metric_weights=None):
    """
    计算早停指标值

    Args:
        valid_results: 验证集结果 [precision_list, recall_list, ndcg_list, mrr_list, f1_list]
        metric_mode: 'recall@20', 'ndcg@20', 'combined_two', 或 'combined_four'
        metric_weights: 权重列表 [w_r10, w_r20, w_n10, w_n20]

    Returns:
        metric_value: 计算得到的指标值
        metric_info: 指标详细信息字符串
        metrics_dict: 各指标的字典 {'recall_10', 'recall_20', 'ndcg_10', 'ndcg_20'}
    """
    try:
        topN_list = eval(args.topN)

        # 获取各指标的索引
        idx_10 = topN_list.index(10) if 10 in topN_list else None
        idx_20 = topN_list.index(20) if 20 in topN_list else None

        if idx_10 is None or idx_20 is None:
            raise ValueError("topN列表必须包含10和20")

        # 提取四个关键指标
        recall_10 = valid_results[1][idx_10]
        recall_20 = valid_results[1][idx_20]
        ndcg_10 = valid_results[2][idx_10]
        ndcg_20 = valid_results[2][idx_20]

        metrics_dict = {
            'recall_10': recall_10,
            'recall_20': recall_20,
            'ndcg_10': ndcg_10,
            'ndcg_20': ndcg_20
        }

        if metric_mode == 'recall@20':
            metric_value = recall_20
            metric_info = f"Recall@20={recall_20:.4f}"

        elif metric_mode == 'ndcg@20':
            metric_value = ndcg_20
            metric_info = f"NDCG@20={ndcg_20:.4f}"

        elif metric_mode == 'combined_two':
            # 综合Recall@20和NDCG@20
            if metric_weights is None:
                metric_weights = [0.0, 0.5, 0.0, 0.5]
            w_r20, w_n20 = metric_weights[1], metric_weights[3]
            total_weight = w_r20 + w_n20
            w_r20_norm, w_n20_norm = w_r20 / total_weight, w_n20 / total_weight

            metric_value = w_r20_norm * recall_20 + w_n20_norm * ndcg_20
            metric_info = (f"Combined_Two={metric_value:.4f} "
                           f"(R@20={recall_20:.4f}×{w_r20_norm:.2f} + N@20={ndcg_20:.4f}×{w_n20_norm:.2f})")

        else:  # combined_four
            # 综合四个指标
            if metric_weights is None:
                metric_weights = [0.25, 0.25, 0.25, 0.25]

            w_r10, w_r20, w_n10, w_n20 = metric_weights

            # 归一化权重
            total_weight = sum(metric_weights)
            if total_weight > 0:
                w_r10, w_r20, w_n10, w_n20 = [w / total_weight for w in metric_weights]

            metric_value = (w_r10 * recall_10 + w_r20 * recall_20 +
                            w_n10 * ndcg_10 + w_n20 * ndcg_20)

            metric_info = (f"Combined_Four={metric_value:.4f}\n"
                           f"    R@10={recall_10:.4f}×{w_r10:.2f} + R@20={recall_20:.4f}×{w_r20:.2f} + "
                           f"N@10={ndcg_10:.4f}×{w_n10:.2f} + N@20={ndcg_20:.4f}×{w_n20:.2f}")

        return metric_value, metric_info, metrics_dict

    except (ValueError, IndexError) as e:
        print(f"警告: 无法计算早停指标 - {e}")
        return -1.0, "N/A", {'recall_10': -1.0, 'recall_20': -1.0, 'ndcg_10': -1.0, 'ndcg_20': -1.0}


# --- 参数解析 ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str, default='PWA', choices=['PWA', 'HGA'],
                    help='Type of dataset to use: PWA or HGA.')
parser.add_argument('--dataset_name', type=str, default='PWA_processed',
                    help='name of the processed dataset folder (e.g., PWA_processed or HGA_processed)')
parser.add_argument('--processed_data_path', type=str, default='./dataset1/',
                    help='base path to the processed dataset folder')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[10, 20, 50]', help='for evaluation')
parser.add_argument('--cuda', action='store_true', help='use CUDA (如果可用)')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='base save model path')
parser.add_argument('--log_name', type=str, default='diffusion_log', help='the log name suffix')
parser.add_argument('--norm', type=bool, default=False, help='Normalize input/attention in CAM_AE_multihops')
parser.add_argument('--emb_size', type=int, default=64, help='timestep embedding size (d_time_embed)')
parser.add_argument('--d_model', type=int, default=128, help='model dimension for CAM_AE_multihops (d_model in paper)')
parser.add_argument('--hidden_dims', type=str, default='[256]',
                    help='hidden dimension list for the intermediate projection layer, e.g., "[512, 256]"')
parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--num_layers', type=int, default=1, help='number of layers in each attention branch')
parser.add_argument('--dropout', type=float, default=0.5, help="dropout rate for CAM_AE_multihops")
parser.add_argument('--softmax_temp', type=float, default=1.0, help='Temperature for softmax in dynamic fusion weights')
parser.add_argument('--dim_inters', type=int, default=256,
                    help='Intermediate dimension for the main and branch encoders.')  # 新增参数
parser.add_argument('--fusion_mode', type=str, default='attention',
                    choices=['attention', 'mean', 'max', 'concat'],
                    help='融合指导信号的机制: attention (动态注意力), mean (平均池化), max (最大池化), concat (拼接-线性)')
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps (START_X or EPSILON)')
parser.add_argument('--steps', type=int, default=50, help='diffusion steps T')
parser.add_argument('--noise_schedule', type=str, default='linear', help='noise schedule (linear, cosine, etc.)')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for beta generation')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for linear schedule')
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for linear schedule')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not during inference')
parser.add_argument('--sampling_steps', type=int, default=50, help='effective steps for inference (<= T)')
parser.add_argument('--reweight', type=bool, default=True,
                    help='assign different weight to different timestep for loss')
parser.add_argument('--use_feature_branch1', type=str_to_bool, nargs='?', const=True, default=True,
                    help='Use User-User structural features (e.g., Mashup-Mashup 2-hop / App-App structural).')
parser.add_argument('--use_feature_branch2', type=str_to_bool, nargs='?', const=True, default=True,
                    help='Use Item-Item structural contextual features (e.g., Contextual API-API Similarity / Contextual API-API Structural).')
parser.add_argument('--use_feature_branch3', type=str_to_bool, nargs='?', const=True, default=True,
                    help='Use Item-Item category/complementarity contextual features (e.g., API complementarity / API Category Co-occurrence).')
parser.add_argument('--use_feature_branch4', type=str_to_bool, nargs='?', const=True, default=True,
                    help='Use User-User semantic/text features (e.g., Mashup text similarity / App category similarity).')

# ================= 新增：用于实例分析的权重保存开关 =================
parser.add_argument('--save_weights_for_analysis', type=str_to_bool, nargs='?', const=True, default=False,
                    help='在训练结束后，加载最佳模型并为实例分析保存权重。')
parser.add_argument('--early_stop_metric', type=str, default='combined_four',
                    choices=['recall@20', 'ndcg@20', 'combined_two', 'combined_four'],
                    help='早停指标: recall@20, ndcg@20, combined_two (R@20+N@20), combined_four (R@10+R@20+N@10+N@20)')
parser.add_argument('--metric_weights', type=str, default='[0.25, 0.25, 0.25, 0.25]',
                    help='综合评估模式下各指标权重 [R@10, R@20, N@10, N@20]，总和应为1.0')
# =================================================================

args = parser.parse_args()

# 只在用户没有明确指定dataset_name时，才使用默认值
if args.dataset_name == 'PWA_processed':  # 检查是否还是默认值
    if args.dataset_type == 'HGA':
        args.dataset_name = 'HGA_processed'
    elif args.dataset_type == 'PWA':
        args.dataset_name = 'PWA_processed'  # 保持默认
    else:
        raise ValueError(f"不支持的数据集类型: {args.dataset_type}")
# 如果用户明确指定了dataset_name (如PWA_processed1)，则保留用户的选择

args.save_path = os.path.join(args.save_path, args.dataset_type)

print("PyTorch 版本:", torch.__version__)
print("参数详情:", args)

if args.cuda and torch.cuda.is_available():
    # 直接指定GPU设备号（而不是通过环境变量）
    device = torch.device(f"cuda:{args.gpu}")
    print(f"使用 GPU: {args.gpu}")
else:
    device = torch.device("cpu")
    print("使用 CPU")

print("开始时间: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

processed_dataset_path = os.path.join(args.processed_data_path, args.dataset_name)
n_user_or_app = 0
n_item_or_api = 0

if args.dataset_type == 'PWA':
    (train_interactions_dense, feature1_user_user_struct, feature2_item_item_struct_ctx,
     feature3_item_item_cat_ctx, feature4_user_user_semantic, train_interactions_sparse,
     valid_interactions_sparse, test_interactions_sparse, n_user_or_app,
     n_item_or_api) = data_utils.load_processed_mashup_data(processed_dataset_path)
    print(f"PWA数据加载完成: Mashups={n_user_or_app}, APIs={n_item_or_api}")
    train_dataset = data_utils.MashupFeaturesDataset(
        train_interactions_dense, feature1_user_user_struct, feature2_item_item_struct_ctx,
        feature3_item_item_cat_ctx, feature4_user_user_semantic, num_mashups=n_user_or_app,
        num_apis=n_item_or_api)
    eval_dataset = data_utils.MashupFeaturesDataset(
        train_interactions_dense, feature1_user_user_struct, feature2_item_item_struct_ctx,
        feature3_item_item_cat_ctx, feature4_user_user_semantic, num_mashups=n_user_or_app,
        num_apis=n_item_or_api)

elif args.dataset_type == 'HGA':
    (train_interactions_dense, feature1_user_user_struct, feature2_item_item_struct_ctx,
     feature3_item_item_cat_ctx, feature4_user_user_semantic, train_interactions_sparse,
     valid_interactions_sparse, test_interactions_sparse, n_user_or_app,
     n_item_or_api) = data_utils.load_processed_hga_data(processed_dataset_path)
    print(f"HGA数据加载完成: Apps={n_user_or_app}, APIs={n_item_or_api}")
    train_dataset = data_utils.AppFeaturesDataset(
        train_interactions_dense, feature1_user_user_struct, feature2_item_item_struct_ctx,
        feature3_item_item_cat_ctx, feature4_user_user_semantic, num_apps=n_user_or_app,
        num_apis=n_item_or_api)
    eval_dataset = data_utils.AppFeaturesDataset(
        train_interactions_dense, feature1_user_user_struct, feature2_item_item_struct_ctx,
        feature3_item_item_cat_ctx, feature4_user_user_semantic, num_apps=n_user_or_app,
        num_apis=n_item_or_api)
else:
    raise ValueError(f"不支持的数据集类型: {args.dataset_type}")

train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=0, worker_init_fn=worker_init_fn, pin_memory=True if device.type == 'cuda' else False)

eval_loader = DataLoader(
    eval_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=0, worker_init_fn=worker_init_fn, pin_memory=True if device.type == 'cuda' else False)

print('数据准备完毕。')

model = CAM_AE_multihops(
    d_model=args.d_model, num_heads=args.num_heads, num_layers=args.num_layers,
    in_dims=n_item_or_api, emb_size=args.emb_size, num_mashups_dim=n_user_or_app,
    hidden_dims=json.loads(args.hidden_dims), use_2hop=args.use_feature_branch1,
    use_contextual_api_api_sim=args.use_feature_branch2, use_api_compl=args.use_feature_branch3,
    use_mashup_text_sim=args.use_feature_branch4, norm=args.norm, dropout=args.dropout,
    fusion_mode=args.fusion_mode, softmax_temperature=args.softmax_temp,
    dim_inters=args.dim_inters  # 传递新参数
).to(device)

if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError(f"未实现的 mean_type: {args.mean_type}")

diffusion = gd.GaussianDiffusion(
    denoise_fn=model, mean_type=mean_type, noise_schedule=args.noise_schedule,
    noise_scale=args.noise_scale, noise_min=args.noise_min, noise_max=args.noise_max,
    steps=args.steps, device=device
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("模型和Diffusion框架构建完毕。")


# ================= 修改：evaluate函数增加了 F1 计算 =================
def evaluate(data_loader_for_eval, ground_truth_sparse, history_mask_sparse, topN_list, current_n_users_or_apps,
             save_weights_path=None):
    eval_start_time = time.time()
    model.eval()

    all_user_indices = list(range(current_n_users_or_apps))
    predict_items_all_users = [[] for _ in range(current_n_users_or_apps)]

    # --- 用于保存权重的新增变量 ---
    all_weights_data = {}  # key: user_id, value: weights_numpy_array
    active_branch_names = []  # 保存分支名称
    # ---------------------------

    # --- 新增：用于全局指标的变量 ---
    all_labels_for_auc = []
    all_scores_for_auc = []

    tp_count = 0.0
    fp_count = 0.0
    fn_count = 0.0
    f1_threshold = 0.0  # 假设阈值为0.0，用于计算全局F1
    # ---------------------------

    processed_users = 0
    with torch.no_grad():
        for batch_data_tuple in data_loader_for_eval:
            x_b, f1_b, f2_b, f3_b, f4_b = batch_data_tuple
            current_batch_size = x_b.shape[0]
            start_idx = processed_users
            end_idx = processed_users + current_batch_size
            current_user_batch_indices = all_user_indices[start_idx:end_idx]

            x_b = x_b.to(device)
            eval_f1 = f1_b.to(device) if args.use_feature_branch1 and f1_b is not None else None
            eval_f2 = f2_b.to(device) if args.use_feature_branch2 and f2_b is not None else None
            eval_f3 = f3_b.to(device) if args.use_feature_branch3 and f3_b is not None else None
            eval_f4 = f4_b.to(device) if args.use_feature_branch4 and f4_b is not None else None

            prediction_scores, dynamic_weights, last_active_branch_names = diffusion.p_sample(
                x_b,
                x_mashup_2hop=eval_f1,
                x_contextual_api_api_sim=eval_f2,
                x_api_compl=eval_f3,
                x_mashup_text_sim=eval_f4,
                steps=args.sampling_steps if args.sampling_steps > 0 else args.steps,
                sampling_noise=args.sampling_noise
            )

            # --- 新增：如果需要，则收集权重 ---
            if save_weights_path is not None:
                if not active_branch_names and last_active_branch_names:
                    active_branch_names = last_active_branch_names  # 只需记录一次分支名称

                if dynamic_weights:  # 检查列表是否为空
                    weights_last_step = dynamic_weights[-1]  # Shape: (B, S, K)
                    for i in range(current_batch_size):
                        user_original_idx = current_user_batch_indices[i]
                        # S通常为1，用squeeze()去掉，转为numpy
                        weights_for_user = weights_last_step[i].squeeze().cpu().numpy()
                        all_weights_data[user_original_idx] = weights_for_user
                else:
                    if last_active_branch_names and len(last_active_branch_names) == 1:
                        if not active_branch_names:
                            active_branch_names = last_active_branch_names
                        for i in range(current_batch_size):
                            user_original_idx = current_user_batch_indices[i]
                            all_weights_data[user_original_idx] = np.array([1.0], dtype=np.float32)

            # --- 新增：为 AUC 和 F1 收集数据 ---
            # 获取当前批次的真实标签和历史记录 (转为numpy)
            his_data_for_batch_np = history_mask_sparse[current_user_batch_indices].A
            gt_data_for_batch_np = ground_truth_sparse[current_user_batch_indices].A
            prediction_scores_np = prediction_scores.cpu().numpy()

            # 1. 评估掩码：只在非历史记录的物品上评估
            eval_mask_np = (his_data_for_batch_np == 0)

            # 2. 真实标签：只看评估物品
            ground_truth_eval_np = (gt_data_for_batch_np == 1) & eval_mask_np

            # 3. 为 AUC 收集数据
            if roc_auc_score is not None:
                for i in range(current_batch_size):
                    user_gt_eval = ground_truth_eval_np[i]
                    # 只有当评估集中同时存在正负样本时，才为AUC收集数据
                    if np.sum(user_gt_eval) > 0 and np.sum(user_gt_eval) < np.sum(eval_mask_np[i]):
                        user_scores_eval = prediction_scores_np[i][eval_mask_np[i]]
                        user_labels_eval = user_gt_eval[eval_mask_np[i]]

                        all_scores_for_auc.extend(user_scores_eval.tolist())
                        all_labels_for_auc.extend(user_labels_eval.tolist())

            # 4. 为 全局F1 收集数据
            predicted_positives_np = (prediction_scores_np > f1_threshold) & eval_mask_np

            tp_count += np.sum(predicted_positives_np & ground_truth_eval_np)
            fp_count += np.sum(predicted_positives_np & ~ground_truth_eval_np)

            # ======================== BUG 修复 ========================
            # 错误的代码: fn_count += np.sum(~predicted_positives_np & ~predicted_positives_np)
            # 正确的代码:
            fn_count += np.sum(~predicted_positives_np & ground_truth_eval_np)
            # ==========================================================

            # --- 全局指标逻辑结束 ---

            # --- Top-K 指标逻辑 (不变) ---
            his_data_for_topk = history_mask_sparse[current_user_batch_indices].A
            prediction_scores_topk = prediction_scores.clone()  # 复制一份用于TopK
            prediction_scores_topk[his_data_for_topk.nonzero()] = -np.inf

            _, top_k_indices = torch.topk(prediction_scores_topk, topN_list[-1])
            top_k_indices = top_k_indices.cpu().numpy().tolist()

            for i in range(current_batch_size):
                user_original_idx = current_user_batch_indices[i]
                predict_items_all_users[user_original_idx].extend(top_k_indices[i])

            processed_users += current_batch_size

    # --- 新增：保存权重到文件 (不变) ---
    if save_weights_path is not None:
        if active_branch_names and all_weights_data:
            data_to_save = {
                'branch_names': active_branch_names,
                'user_fusion_weights': all_weights_data
            }
            torch.save(data_to_save, save_weights_path)
            print(f"动态融合权重已保存到: {save_weights_path}")
            # ... (省略了可读日志的导出代码，逻辑不变) ...
            try:
                if save_weights_path.endswith('.pt'):
                    log_save_path = save_weights_path[:-3] + 'log'
                else:
                    log_save_path = save_weights_path + '.log'
                lines = []
                lines.append(f"branch_names: {', '.join(active_branch_names)}\n")
                lines.append(f"num_users: {len(all_weights_data)}\n")
                lines.append("\n# weights_by_user (按 branch_names 顺序)\n")
                for uid in sorted(all_weights_data.keys()):
                    weights_arr = all_weights_data[uid]
                    try:
                        import numpy as _np
                        wa = _np.array(weights_arr)
                        if wa.ndim > 1:
                            wa = wa.mean(axis=0)
                        weights_fmt = ' '.join([f"{float(x):.4f}" for x in wa.tolist()])
                    except Exception:
                        weights_fmt = ' '.join([f"{float(x):.4f}" for x in list(weights_arr)])
                    lines.append(f"user {uid}: {weights_fmt}\n")
                with open(log_save_path, 'w', encoding='utf-8') as f_log:
                    f_log.writelines(lines)
                print(f"动态融合权重可读日志已保存到: {log_save_path}")
            except Exception as e:
                print(f"警告: 导出动态权重 .log 失败: {e}")
        else:
            print("警告: 未能收集到动态权重，无法保存。")

    # --- 新增：计算 全局F1 和 AUC ---
    auc_score = 0.0
    if roc_auc_score is not None and all_labels_for_auc:
        try:
            auc_score = roc_auc_score(all_labels_for_auc, all_scores_for_auc)
        except ValueError as e:
            print(f"警告: AUC 计算失败 (例如，评估集中只存在一个类别): {e}")
            auc_score = 0.0
    elif roc_auc_score is None:
        print("信息: 未安装sklearn，跳过AUC计算。")
    else:
        print("警告: 没有为AUC收集到任何有效数据。")

    f1_score = 0.0
    try:
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
    except Exception as e:
        print(f"警告: 全局F1 计算失败: {e}")
        f1_score = 0.0
    # ---------------------------

    # --- Top-K 结果计算 (不变) ---
    target_items_all_users = [[] for _ in range(current_n_users_or_apps)]
    for u_idx in range(current_n_users_or_apps):
        items_for_user = ground_truth_sparse[u_idx].nonzero()[1]
        if len(items_for_user) > 0:
            target_items_all_users[u_idx] = items_for_user.tolist()

    eval_user_indices_with_gt = [i for i, tgt_list in enumerate(target_items_all_users) if len(tgt_list) > 0]

    if not eval_user_indices_with_gt:
        print(f"警告: {args.dataset_type}评估集中没有找到任何真实标签，无法计算指标。")
        num_topN = len(topN_list)
        return [[0.0] * num_topN for _ in range(5)], 0.0, 0.0, 0.0  # 修改：返回 TopK, duration, AUC, F1

    target_items_to_eval = [target_items_all_users[i] for i in eval_user_indices_with_gt]
    predict_items_to_eval = [predict_items_all_users[i] for i in eval_user_indices_with_gt]

    test_results_topk = evaluate_utils.computeTopNAccuracy(target_items_to_eval, predict_items_to_eval, topN_list)
    eval_duration = time.time() - eval_start_time

    # --- 修改：返回 TopK, duration, AUC, F1 ---
    return test_results_topk, eval_duration, auc_score, f1_score
    # ---------------------------


# --- 训练主循环 ---
if __name__ == '__main__':
    # ... (日志设置不变) ...
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"{args.log_name}_{args.dataset_type}_{timestamp}.log")
    log_file = open(log_filename, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)
    print("=" * 80)
    print(f"日志文件已创建: {log_filename}")
    print("=" * 80)
    print()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    best_metric_value, best_epoch = -1.0, 0
    best_valid_results, best_test_results = None, None
    # --- 新增：记录最佳 全局 F1 和 AUC ---
    best_valid_auc, best_test_auc = 0.0, 0.0
    best_valid_f1, best_test_f1 = 0.0, 0.0
    # ---------------------------
    best_model_save_path = ""
    best_metrics_dict = {}

    all_epoch_dynamic_weights_collector = []
    current_epoch_active_branch_names = []

    metric_weights = json.loads(args.metric_weights)
    if len(metric_weights) != 4:
        raise ValueError(f"metric_weights必须包含4个值，当前为: {metric_weights}")

    patience = 300
    print(f"早停耐心值设置为: {patience} 轮")
    print(f"早停指标模式: {args.early_stop_metric}")
    if args.early_stop_metric in ['combined_two', 'combined_four']:
        print(f"指标权重: {metric_weights}")
        print(f"  - Recall@10: {metric_weights[0]:.2f}")
        print(f"  - Recall@20: {metric_weights[1]:.2f}")
        print(f"  - NDCG@10: {metric_weights[2]:.2f}")
        print(f"  - NDCG@20: {metric_weights[3]:.2f}")

    print(f"开始训练 {args.dataset_type} 数据集...")
    for epoch in range(1, args.epochs + 1):
        if epoch - best_epoch >= patience and best_epoch != 0:
            # ... (早停逻辑不变) ...
            metric_name_map = {
                'recall@20': 'Recall@20',
                'ndcg@20': 'NDCG@20',
                'combined_two': '综合指标(R@20+N@20)',
                'combined_four': '综合指标(R@10+R@20+N@10+N@20)'
            }
            metric_name = metric_name_map.get(args.early_stop_metric, '综合指标')
            print(
                f'从训练中提前退出 (Early stopping): 验证集 {metric_name} 在 {epoch - best_epoch} 个epoch内未改善 (自 epoch {best_epoch}以来).')
            break

        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0
        last_batch_weights = None
        last_batch_names = []

        # ... (训练循环不变) ...
        for batch_idx, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            current_x, current_f1, current_f2, current_f3, current_f4 = batch_data
            current_x, current_f1, current_f2, current_f3, current_f4 = \
                current_x.to(device), current_f1.to(device), current_f2.to(device), current_f3.to(
                    device), current_f4.to(device)
            train_input_f1 = current_f1 if args.use_feature_branch1 else None
            train_input_f2 = current_f2 if args.use_feature_branch2 else None
            train_input_f3 = current_f3 if args.use_feature_branch3 else None
            train_input_f4 = current_f4 if args.use_feature_branch4 else None
            loss_output_tuple = diffusion.training_losses(current_x,
                                                          x_mashup_2hop=train_input_f1,
                                                          x_contextual_api_api_sim=train_input_f2,
                                                          x_api_compl=train_input_f3,
                                                          x_mashup_text_sim=train_input_f4,
                                                          reweight=args.reweight)
            if len(loss_output_tuple) == 3:
                loss_terms, batch_weights, batch_names = loss_output_tuple
                loss = loss_terms["loss"]
                if batch_weights is not None and batch_names:
                    all_epoch_dynamic_weights_collector.append(batch_weights.detach().cpu())
                    if not current_epoch_active_branch_names and batch_names:
                        current_epoch_active_branch_names.extend(batch_names)
                    last_batch_weights = batch_weights.detach().cpu()
                    last_batch_names = batch_names
            else:
                loss_terms = loss_output_tuple
            loss = loss_terms["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch:03d} ({args.dataset_type}) | 训练损失: {avg_loss:.4f} | " +
              f"耗时: {time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_start_time))}")
        if last_batch_weights is not None and last_batch_names:
            avg_weights = last_batch_weights.mean(dim=[0, 1]).numpy()
            weights_str = " ".join([f"{w:.3f}" for w in avg_weights])
            names_str = ", ".join(last_batch_names)
            print(f"  └── 融合权重 (末批均值): [{names_str}] -> [{weights_str}]")

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"\n--- Epoch {epoch} ({args.dataset_type}) 评估开始 ---")
            # --- 修改：接收 全局F1 和 AUC ---
            valid_results, valid_eval_time, valid_auc, valid_f1 = evaluate(
                eval_loader, valid_interactions_sparse, train_interactions_sparse,
                eval(args.topN), n_user_or_app)

            test_history_mask = train_interactions_sparse + valid_interactions_sparse

            test_results, test_eval_time, test_auc, test_f1 = evaluate(
                eval_loader, test_interactions_sparse, test_history_mask,
                eval(args.topN), n_user_or_app)

            # --- 修改：传递 全局F1 和 AUC ---
            evaluate_utils.print_results(f"Epoch {epoch} ({args.dataset_type})",
                                         valid_results, test_results,
                                         valid_auc, test_auc,
                                         valid_f1, test_f1)
            print(f"验证集评估耗时: {time.strftime('%H:%M:%S', time.gmtime(valid_eval_time))}")
            print(f"测试集评估耗时: {time.strftime('%H:%M:%S', time.gmtime(test_eval_time))}")

            # 计算早停指标
            current_metric_value, metric_info, current_metrics_dict = compute_early_stop_metric(
                valid_results, args.early_stop_metric, metric_weights
            )
            print(f"早停指标: {metric_info}")

            if current_metric_value > best_metric_value:
                best_metric_value = current_metric_value
                best_epoch = epoch
                best_valid_results = valid_results
                best_test_results = test_results
                # --- 新增：保存最佳 全局F1 和 AUC ---
                best_valid_auc = valid_auc
                best_test_auc = test_auc
                best_valid_f1 = valid_f1
                best_test_f1 = test_f1
                # -------------------------
                best_metrics_dict = current_metrics_dict.copy()

                model_specific_params = (
                    f"lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}_"
                    f"dmodel{args.d_model}_diminters{args.dim_inters}_nh{args.num_heads}_nl{args.num_layers}_"
                    f"emb{args.emb_size}_drop{args.dropout}_steps{args.steps}_scale{args.noise_scale}_"
                    f"fusion{args.fusion_mode}"
                )
                feature_flags = f"_f1{int(args.use_feature_branch1)}_f2{int(args.use_feature_branch2)}_f3{int(args.use_feature_branch3)}_f4{int(args.use_feature_branch4)}"

                best_model_save_path = (
                    f"{args.save_path}/{args.dataset_name}_model_{model_specific_params}{feature_flags}_{args.log_name}_best.pth")
                torch.save(model.state_dict(), best_model_save_path)
                print(f"✓ 新的最佳模型! 已保存到: {best_model_save_path}")
                print(f"  综合指标值: {best_metric_value:.4f}")
                print(f"  R@10={best_metrics_dict['recall_10']:.4f}, R@20={best_metrics_dict['recall_20']:.4f}, "
                      f"N@10={best_metrics_dict['ndcg_10']:.4f}, N@20={best_metrics_dict['ndcg_20']:.4f}")
                print(f"  Val F1: {best_valid_f1:.4f}, Test F1: {best_test_f1:.4f}")  # 打印全局F1
                print(f"  Val AUC: {best_valid_auc:.4f}, Test AUC: {best_test_auc:.4f}")  # 打印AUC

        print('---' * 18)

    print('===' * 18)
    print(f"训练结束 ({args.dataset_type})。")
    print(f"  最佳 Epoch: {best_epoch:03d}")

    # ... (早停指标总结不变) ...
    if args.early_stop_metric == 'recall@20':
        print(f"  最佳验证集 Recall@20: {best_metric_value:.4f}")
        metric_desc = "Recall@20"
    elif args.early_stop_metric == 'ndcg@20':
        print(f"  最佳验证集 NDCG@20: {best_metric_value:.4f}")
        metric_desc = "NDCG@20"
    elif args.early_stop_metric == 'combined_two':
        print(f"  最佳综合指标: {best_metric_value:.4f}")
        print(f"    - Recall@20: {best_metrics_dict.get('recall_20', 0):.4f} (权重 {metric_weights[1]:.2f})")
        print(f"    - NDCG@20: {best_metrics_dict.get('ndcg_20', 0):.4f} (权重 {metric_weights[3]:.2f})")
        metric_desc = "综合指标(R@20+N@20)"
    else:  # combined_four
        print(f"  最佳综合指标: {best_metric_value:.4f}")
        print(f"    - Recall@10: {best_metrics_dict.get('recall_10', 0):.4f} (权重 {metric_weights[0]:.2f})")
        print(f"    - Recall@20: {best_metrics_dict.get('recall_20', 0):.4f} (权重 {metric_weights[1]:.2f})")
        print(f"    - NDCG@10: {best_metrics_dict.get('ndcg_10', 0):.4f} (权重 {metric_weights[2]:.2f})")
        print(f"    - NDCG@20: {best_metrics_dict.get('ndcg_20', 0):.4f} (权重 {metric_weights[3]:.2f})")
        metric_desc = "综合指标(R@10+R@20+N@10+N@20)"

    if best_valid_results and best_test_results:
        print()
        # --- 修改：传递最佳 全局F1 和 AUC ---
        evaluate_utils.print_results(f"最佳结果 ({args.dataset_type}, 基于验证集{metric_desc})",
                                     best_valid_results, best_test_results,
                                     best_valid_auc, best_test_auc,
                                     best_valid_f1, best_test_f1)
    else:
        print("未能获得有效的最佳结果。")
    print()
    print("结束时间: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # ... (权重保存逻辑不变) ...
    if args.save_weights_for_analysis and best_model_save_path:
        print("\n--- 开始为实例分析保存最佳模型的动态权重 ---")
        try:
            model.load_state_dict(torch.load(best_model_save_path, map_location=device))
            print(f"已成功加载最佳模型: {best_model_save_path}")
            model.eval()
            weights_save_path = os.path.join(args.save_path, f"{args.dataset_name}_dynamic_fusion_weights.pt")
            print("正在重新评估以捕获权重...")
            test_history_mask = train_interactions_sparse + valid_interactions_sparse
            # --- 修改：evaluate 现在返回4个值 ---
            _, _, _, _ = evaluate(eval_loader, test_interactions_sparse, test_history_mask, eval(args.topN),
                                  n_user_or_app,
                                  save_weights_path=weights_save_path)

        except FileNotFoundError:
            print(f"错误: 未找到最佳模型文件 {best_model_save_path}，无法保存权重。")
        except Exception as e:
            print(f"保存权重时发生错误: {e}")

    # ... (日志关闭逻辑不变) ...
    print()
    print("=" * 80)
    print(f"训练完成！日志已保存到: {log_filename}")
    print("=" * 80)
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()