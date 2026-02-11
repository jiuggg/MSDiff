import math


def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = []
    recall = []
    NDCG = []
    MRR = []
    F1 = []  # 这个F1是 F1@K

    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0 / math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0 / (j + 1.0))
                            mrrFlag = False
                        userHit += 1

                    if idcgCount > 0:
                        idcg += 1.0 / math.log2(j + 2)
                        idcgCount = idcgCount - 1

                if (idcg != 0):
                    ndcg += (dcg / idcg)

                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])
                sumForNdcg += ndcg
                sumForMRR += userMRR

        current_precision = round(sumForPrecision / len(predictedIndices), 4)
        current_recall = round(sumForRecall / len(predictedIndices), 4)

        precision.append(current_precision)
        recall.append(current_recall)
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        MRR.append(round(sumForMRR / len(predictedIndices), 4))

        # --- F1@K 计算 ---
        if (current_precision + current_recall) == 0:
            F1.append(0.0)
        else:
            f1_val = 2 * current_precision * current_recall / (current_precision + current_recall)
            F1.append(round(f1_val, 4))
        # ---------------------

    return precision, recall, NDCG, MRR, F1  # 返回 F1@K


def print_results(info_str, valid_result, test_result,
                  valid_auc=None, test_auc=None,
                  valid_f1=None, test_f1=None):  # 新增 valid_f1, test_f1
    """output the evaluation results."""
    if info_str is not None:
        print(f"[{info_str}]")

    # --- 修改：增加 全局F1 和 AUC 的打印 ---
    if valid_result is not None:
        auc_str = f" AUC: {valid_auc:.4f}" if valid_auc is not None else ""
        f1_str = f" F1: {valid_f1:.4f}" if valid_f1 is not None else ""  # 论文的全局F1
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {} F1@K: {}{}{}".format(
            '-'.join([str(x) for x in valid_result[0]]),
            '-'.join([str(x) for x in valid_result[1]]),
            '-'.join([str(x) for x in valid_result[2]]),
            '-'.join([str(x) for x in valid_result[3]]),
            '-'.join([str(x) for x in valid_result[4]]),  # 打印 F1@K
            f1_str,  # 打印 全局F1
            auc_str))  # 打印 AUC

    if test_result is not None:
        auc_str = f" AUC: {test_auc:.4f}" if test_auc is not None else ""
        f1_str = f" F1: {test_f1:.4f}" if test_f1 is not None else ""  # 论文的全局F1
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {} F1@K: {}{}{}".format(
            '-'.join([str(x) for x in test_result[0]]),
            '-'.join([str(x) for x in test_result[1]]),
            '-'.join([str(x) for x in test_result[2]]),
            '-'.join([str(x) for x in test_result[3]]),
            '-'.join([str(x) for x in test_result[4]]),  # 打印 F1@K
            f1_str,  # 打印 全局F1
            auc_str))  # 打印 AUC
    # ---------------------------------