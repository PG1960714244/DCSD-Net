import os

# 强制使用非 GUI 后端，降低 Windows + Anaconda 绘图崩溃概率
os.environ["MPLBACKEND"] = "Agg"

import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# 1. 路径设置
# =========================

# 保存路径
save_dir = Path(r"confusion-matrix")
save_dir.mkdir(parents=True, exist_ok=True)

# 你的预测结果 pred.json，COCO detection result 格式
pred_json_path = r"eval/pred.json"

# 你的验证集或测试集真实标注 val.json / test.json，COCO annotation 格式
val_json_path = r"test.json"


# =========================
# 2. 类别名称
# 注意：这里不写 background，代码会自动添加 background
# 必须与你 val.json 中 categories 的排序一致
# =========================

# NEU-DET 推荐使用缩写
# class_names = ["Cr", "Pa", "In", "Ps", "Rs", "Sc"]

# 如果你要画 GC10-DET，可改成：
class_names = ["Cg", "Cr", "Ss", "Ws", "Wl", "In", "Os", "Rp", "Ph", "Wf"]


# =========================
# 3. 混淆矩阵计算参数
# =========================

conf_thres = 0.25
iou_thres = 0.5

# 如果想更贴近 mAP50 的匹配逻辑，可以设成 0.50
# iou_thres = 0.50


# =========================
# 4. 基础函数
# =========================

def xywh_to_xyxy(box):
    """
    COCO bbox: [x, y, w, h]
    转换为: [x1, y1, x2, y2]
    """
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def box_iou(boxes1, boxes2, eps=1e-7):
    """
    boxes1: [N, 4], xyxy
    boxes2: [M, 4], xyxy
    return: [N, M]
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    boxes1 = boxes1.astype(np.float32)
    boxes2 = boxes2.astype(np.float32)

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h

    area1 = np.clip(b1_x2 - b1_x1, 0, None) * np.clip(b1_y2 - b1_y1, 0, None)
    area2 = np.clip(b2_x2 - b2_x1, 0, None) * np.clip(b2_y2 - b2_y1, 0, None)

    union = area1[:, None] + area2[None, :] - inter_area
    return inter_area / (union + eps)


def load_coco_gt(val_json_path):
    """
    读取 COCO 格式真实标注。
    返回：
    gt_by_image: image_id -> list of {"bbox": xyxy, "category_id": id}
    image_ids: val/test 中所有图像 id
    cat_id_to_idx: COCO category_id -> 矩阵类别索引
    category_names_from_json: val.json 中的类别名称
    """
    with open(val_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = sorted(data["categories"], key=lambda x: x["id"])
    cat_ids = [c["id"] for c in categories]
    category_names_from_json = [c["name"] for c in categories]
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

    image_ids = [img["id"] for img in data["images"]]

    gt_by_image = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue

        image_id = ann["image_id"]
        category_id = ann["category_id"]

        if category_id not in cat_id_to_idx:
            continue

        gt_by_image[image_id].append(
            {
                "bbox": xywh_to_xyxy(ann["bbox"]),
                "category_id": category_id,
            }
        )

    return gt_by_image, image_ids, cat_id_to_idx, category_names_from_json


def load_coco_predictions(pred_json_path, cat_id_to_idx, conf_thres=0.25):
    """
    读取 COCO detection result 格式预测结果。
    pred.json 格式通常为：
    [
      {"image_id": 1, "category_id": 3, "bbox": [x, y, w, h], "score": 0.91},
      ...
    ]
    """
    with open(pred_json_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    pred_by_image = defaultdict(list)

    for pred in preds:
        score = float(pred.get("score", 1.0))
        if score <= conf_thres:
            continue

        category_id = pred["category_id"]
        if category_id not in cat_id_to_idx:
            continue

        image_id = pred["image_id"]

        pred_by_image[image_id].append(
            {
                "bbox": xywh_to_xyxy(pred["bbox"]),
                "category_id": category_id,
                "score": score,
            }
        )

    # 按置信度排序，不是必须，但便于排查
    for image_id in pred_by_image:
        pred_by_image[image_id].sort(key=lambda x: x["score"], reverse=True)

    return pred_by_image


def build_detection_confusion_matrix(
    val_json_path,
    pred_json_path,
    class_names,
    conf_thres=0.25,
    iou_thres=0.45,
):
    """
    构建目标检测混淆矩阵。

    矩阵定义：
    行 = Predicted
    列 = True

    最后一行 background：
        真实目标没有被匹配到，表示漏检 FN。

    最后一列 background：
        预测框没有匹配到任何真实目标，表示误检 FP。
    """
    gt_by_image, image_ids, cat_id_to_idx, category_names_from_json = load_coco_gt(val_json_path)
    pred_by_image = load_coco_predictions(pred_json_path, cat_id_to_idx, conf_thres=conf_thres)

    nc = len(cat_id_to_idx)

    if len(class_names) != nc:
        raise ValueError(
            f"class_names 数量与 val.json 中 categories 数量不一致。"
            f"class_names={len(class_names)}, categories={nc}\n"
            f"val.json categories={category_names_from_json}"
        )

    matrix = np.zeros((nc + 1, nc + 1), dtype=np.float32)
    bg_idx = nc

    for image_id in image_ids:
        gts = gt_by_image.get(image_id, [])
        preds = pred_by_image.get(image_id, [])

        gt_boxes = np.array([g["bbox"] for g in gts], dtype=np.float32) if len(gts) else np.zeros((0, 4), dtype=np.float32)
        pred_boxes = np.array([p["bbox"] for p in preds], dtype=np.float32) if len(preds) else np.zeros((0, 4), dtype=np.float32)

        gt_classes = np.array([cat_id_to_idx[g["category_id"]] for g in gts], dtype=np.int64) if len(gts) else np.zeros((0,), dtype=np.int64)
        pred_classes = np.array([cat_id_to_idx[p["category_id"]] for p in preds], dtype=np.int64) if len(preds) else np.zeros((0,), dtype=np.int64)

        # 情况 1：该图没有真实目标，但有预测框，全部记为背景误检
        if len(gts) == 0:
            for pc in pred_classes:
                matrix[pc, bg_idx] += 1
            continue

        # 情况 2：该图有真实目标，但没有预测框，全部记为漏检
        if len(preds) == 0:
            for gc in gt_classes:
                matrix[bg_idx, gc] += 1
            continue

        # 情况 3：正常 IoU 匹配
        ious = box_iou(gt_boxes, pred_boxes)

        gt_idx, pred_idx = np.where(ious > iou_thres)

        if len(gt_idx):
            matches = np.stack([gt_idx, pred_idx, ious[gt_idx, pred_idx]], axis=1)

            # 模仿 YOLO 的匹配逻辑：
            # 先按 IoU 从高到低排序；
            # 每个预测框只能匹配一个 GT；
            # 每个 GT 也只能匹配一个预测框。
            matches = matches[matches[:, 2].argsort()[::-1]]

            _, unique_pred_indices = np.unique(matches[:, 1], return_index=True)
            matches = matches[unique_pred_indices]

            matches = matches[matches[:, 2].argsort()[::-1]]
            _, unique_gt_indices = np.unique(matches[:, 0], return_index=True)
            matches = matches[unique_gt_indices]

            matched_gt = set(matches[:, 0].astype(int).tolist())
            matched_pred = set(matches[:, 1].astype(int).tolist())

            # 已匹配目标：统计类别正确或类别混淆
            for gi, pi, _ in matches:
                gi = int(gi)
                pi = int(pi)
                true_cls = gt_classes[gi]
                pred_cls = pred_classes[pi]
                matrix[pred_cls, true_cls] += 1

            # 未匹配 GT：漏检，进入 background 行
            for gi, true_cls in enumerate(gt_classes):
                if gi not in matched_gt:
                    matrix[bg_idx, true_cls] += 1

            # 未匹配预测框：误检，进入 background 列
            for pi, pred_cls in enumerate(pred_classes):
                if pi not in matched_pred:
                    matrix[pred_cls, bg_idx] += 1

        else:
            # 没有任何 IoU 超过阈值：
            # 所有 GT 漏检，所有预测框误检
            for true_cls in gt_classes:
                matrix[bg_idx, true_cls] += 1

            for pred_cls in pred_classes:
                matrix[pred_cls, bg_idx] += 1

    display_names = class_names + ["background"]

    return matrix, display_names, category_names_from_json


# =========================
# 5. 检查矩阵
# =========================

def check_matrix(matrix, names, model_name):
    matrix = np.asarray(matrix, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError(f"{model_name}: matrix must be 2D.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{model_name}: matrix must be square, got {matrix.shape}.")

    if matrix.shape[0] != len(names):
        raise ValueError(
            f"{model_name}: matrix size does not match class number. "
            f"Matrix: {matrix.shape[0]}, classes: {len(names)}"
        )

    print(f"\n{model_name}")
    print("Matrix shape:", matrix.shape)
    print("Raw matrix:")
    print(matrix.astype(int))

    print("\nColumn sums, because x-axis is True:")
    for cls_name, s in zip(names, matrix.sum(axis=0)):
        print(f"{cls_name:18s}: {s:.0f}")

    return matrix


# =========================
# 6. YOLO 风格绘制函数
# 绘图形式保持你当前代码风格
# =========================

def plot_yolo_style_confusion_matrix(
    matrix,
    names,
    save_path,
    normalize=True,
    title=None,
    bottom_title=None,
    dpi=250,
):
    """
    YOLO-style confusion matrix.

    matrix:
        Shape [nc+1, nc+1].
        Rows are predicted labels.
        Columns are true labels.

    names:
        Class names including background.

    save_path:
        Output file path.

    normalize:
        If True, normalize each column, consistent with YOLO.
    """
    matrix = np.asarray(matrix, dtype=np.float32)

    # YOLO 的归一化方式：按列归一化
    if normalize:
        array = matrix / (matrix.sum(0).reshape(1, -1) + 1e-9)
    else:
        array = matrix.copy()

    # YOLO 中小于 0.005 的值不显示
    array[array < 0.005] = np.nan

    nc = len(names) - 1

    if title is None:
        title = "Confusion Matrix" + (" Normalized" if normalize else "")

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    sns.set_theme(font_scale=1.0 if nc < 50 else 0.8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.heatmap(
            array,
            ax=ax,
            annot=nc < 30,
            annot_kws={"size": 12},
            cmap="YlGn",
            fmt=".2f" if normalize else ".0f",
            square=True,
            vmin=0.0,
            vmax=1.0 if normalize else None,
            xticklabels=names,
            yticklabels=names,
        ).set_facecolor((1, 1, 1))

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("True", fontsize=12, labelpad=16)
    ax.set_ylabel("Predicted", fontsize=12, labelpad=16)
    ax.set_title(title, fontsize=12, pad=16)

    # 为下方标题预留空间
    fig.subplots_adjust(bottom=0.16, top=0.92, left=0.12, right=0.92)

    # 添加图像下方子标题
    if bottom_title is not None:
        fig.text(
            0.5,
            0.035,
            bottom_title,
            ha="center",
            va="center",
            fontsize=20,
            fontname="Times New Roman",
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")


# =========================
# 7. 主程序
# =========================

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    print("Save directory:", save_dir)
    print("Save directory exists:", save_dir.exists())

    print("\nPrediction json:", pred_json_path)
    print("Validation json:", val_json_path)
    print(f"conf_thres={conf_thres}, iou_thres={iou_thres}")

    cm, classes_with_bg, category_names_from_json = build_detection_confusion_matrix(
        val_json_path=val_json_path,
        pred_json_path=pred_json_path,
        class_names=class_names,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
    )

    print("\nCategories from val.json:")
    print(category_names_from_json)

    cm = check_matrix(cm, classes_with_bg, "Detection confusion matrix")

    save_path = save_dir / "confusion_matrix_from_pred_json.png"

    plot_yolo_style_confusion_matrix(
        matrix=cm,
        names=classes_with_bg,
        save_path=save_path,
        normalize=True,
        title="Confusion Matrix Normalized",
        bottom_title="(a) Confusion matrix",
    )

    print("\nFile check:")
    print(save_path)
    print("Exists:", save_path.exists())
    if save_path.exists():
        print("Size:", save_path.stat().st_size, "bytes")

