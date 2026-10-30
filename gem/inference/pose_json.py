"""将 Body 单图预测转换为含参考姿态的完整骨架 JSON。"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def validate_prediction(prediction):
    """只接受一个人物的 Body 单图结果，并返回该人物记录。"""
    if not isinstance(prediction, dict) or prediction.get("format") != "gem_pose_predictions_v1":
        raise ValueError("输入不是支持的 GEM Pose 预测格式")
    if prediction.get("task") != "body" or prediction.get("mode") != "image":
        raise ValueError("仅支持 Body 单图预测，不支持 Hand 或视频结果")
    frames = prediction.get("frames", [])
    if len(frames) != 1 or len(frames[0].get("instances", [])) != 1:
        raise ValueError("输入必须包含一帧且恰好一个人物")
    if prediction.get("coordinates", {}).get("geometry") != "camera_meters":
        raise ValueError("输入几何必须采用以米为单位的原始相机坐标")
    return frames[0]["instances"][0]


def _array(value):
    """允许读取已保存的 CPU 或 GPU 张量，不主动导入识别网络。"""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _check_hierarchy(names, parents):
    if not names or names[0] != "Hips" or len(names) != len(parents):
        raise ValueError("骨架必须以 Hips 为根，名称和父节点数量必须一致")
    if len(set(names)) != len(names) or parents[0] != -1:
        raise ValueError("关节名称必须唯一，Hips 必须是唯一根节点")
    for index in range(1, len(names)):
        if not 0 <= parents[index] < len(names):
            raise ValueError("父节点索引越界或存在多个根节点")
        visited = set()
        cursor = index
        while cursor != -1:
            if cursor in visited:
                raise ValueError("骨架父子关系存在循环")
            visited.add(cursor)
            cursor = parents[cursor]


def _transforms(value, count):
    a = _array(value)
    if a.shape != (count, 4, 4) or not np.isfinite(a).all():
        raise ValueError("关节变换必须是数量匹配且数值有限的 4×4 矩阵")
    r = a[:, :3, :3]
    if not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=2e-4) or not np.allclose(
        np.linalg.det(r), 1, atol=2e-4
    ):
        raise ValueError("关节旋转必须是正交且行列式为正一的旋转矩阵")
    if not np.allclose(a[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("关节齐次变换最后一行无效")
    return a


def build_pose_document(prediction, names, parents, reference, posed):
    """将公共骨架变换转换为统一基坐标的可序列化字典。"""
    row = validate_prediction(prediction)
    names = [str(name) for name in names]
    parents = [int(parent) for parent in parents]
    _check_hierarchy(names, parents)
    reference = _transforms(reference, len(names))
    posed = _transforms(posed, len(names))
    saved = _array(row["joints_camera"])
    translation = _array(row["camera_translation"])
    if saved.shape != (len(names), 3) or not np.isfinite(saved).all():
        raise ValueError("保存的关节位置数量不匹配或存在非有限数值")
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("相机平移必须包含三个有限数值")
    error = float(np.linalg.norm(posed[:, :3, 3] - saved, axis=1).max())
    if error > 1e-4:
        raise ValueError(f"骨架重建与保存关节不一致，最大误差 {error * 1000:.6f} 毫米")
    # 相机 Y 向下，交换协议 Y 向上；旋转必须同时变换输入与输出基。
    basis = np.diag([1.0, -1.0, 1.0])
    rotations, positions = [], []
    for transforms in [reference, posed]:
        r = basis @ transforms[:, :3, :3] @ basis
        q = Rotation.from_matrix(r).as_quat(canonical=True)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        rotations.append(q)
        positions.append((transforms[:, :3, 3] - transforms[0, :3, 3]) @ basis)
    release = prediction.get("release") or {}
    return {
        "schema_version": "gemx_unity_pose_v1",
        "source": {
            "image": prediction.get("source"),
            "image_size": prediction["frames"][0]["image_size"],
            "model": release.get("pose_release_id"),
            "soma_version": "0.3.3",
            "reference_pose_version": "0.2.0",
            "geometry_profile": "legacy_pose021",
        },
        "coordinates": {
            "unit": "meters",
            "x": "right",
            "y": "up",
            "z": "forward",
            "handedness": "left",
            "quaternion_order": "xyzw",
            "rotation_space": "model",
            "position_origin": "Hips",
        },
        "camera_translation": (translation @ basis).tolist(),
        "validation": {"max_reconstruction_error_meters": error},
        "joints": [
            {
                "name": name,
                "parent_index": parents[i],
                "reference_world_rotation": rotations[0][i].tolist(),
                "posed_world_rotation": rotations[1][i].tolist(),
                "reference_root_relative_position": positions[0][i].tolist(),
                "posed_root_relative_position": positions[1][i].tolist(),
            }
            for i, name in enumerate(names)
        ],
    }


class BodyPoseExporter:
    """复用 SOMA 几何运行时；独立导出无需加载 DINOv3 或模型权重。"""

    def __init__(self, soma_assets, device="cpu", geometry=None):
        self.soma_assets = Path(soma_assets)
        self.device = device
        self.geometry = geometry

    def export(self, prediction):
        """重建源公共骨架的参考和预测变换，并校验保存的三维关节。"""
        import torch

        from gem.common.geometry.rotations import matrix_to_axis_angle
        from gem.pose.geometry.body import BodyGeometry

        row = validate_prediction(prediction)
        params = row.get("parameters", {})
        shapes = {"body_pose": (76, 3, 3), "global_orient": (1, 3, 3), "identity_coeffs": (128,)}
        for key, shape in shapes.items():
            if key not in params or _array(params[key]).shape != shape:
                raise ValueError(f"Body 参数 {key} 缺失或形状无效")
        scale = _array(params.get("scale_params", []))
        if scale.shape not in {(56,), (57,)}:
            raise ValueError("Body 缩放参数必须包含 56 或 57 个数值")
        for key, value in params.items():
            if not np.isfinite(_array(value)).all():
                raise ValueError(f"Body 参数 {key} 存在非有限数值")
        if scale.shape == (57,) and scale[0] != 1.0:
            raise ValueError("Body 全局缩放必须固定为一")
        if self.geometry is None:
            self.geometry = BodyGeometry(self.soma_assets, device=self.device)
        runtime = self.geometry.body_model.runtime
        if tuple(map(str, runtime.public_joint_names))[1] != "Hips":
            raise ValueError("SOMA 公共骨架不符合虚拟根和 Hips 顺序")
        names = list(map(str, runtime.public_joint_names))[1:]
        if len(names) != 77:
            raise ValueError("SOMA 公共骨架必须包含 77 个预测关节")
        parents = runtime.public_joint_parent_ids[1:].cpu().numpy() - 1
        parents[0] = -1
        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32, device=self.device)

        local56 = tensor(scale[-56:]).reshape(1, 56)
        local60 = torch.cat((local56, torch.ones(1, 4, device=self.device)), dim=-1)
        identity = tensor(params["identity_coeffs"]).reshape(1, 128)
        matrices = torch.cat((tensor(params["global_orient"]), tensor(params["body_pose"])), 0)
        angles = matrix_to_axis_angle(matrices).unsqueeze(0)
        translation = tensor(row["camera_translation"]).reshape(1, 3)
        with torch.inference_mode():
            runtime.prepare_identity(
                identity,
                local60,
                repose_to_bind_pose=True,
                global_scale=torch.ones(1, device=self.device),
            )
            posed = runtime.pose(
                angles, transl=translation, pose2rot=True, apply_correctives=False, fk_only=True
            )
            posed = posed.transforms[0, 1:].detach().cpu().numpy().copy()
            reference = runtime.pose(
                torch.zeros_like(angles),
                transl=torch.zeros_like(translation),
                pose2rot=True,
                apply_correctives=False,
                fk_only=True,
            )
            reference = reference.transforms[0, 1:].detach().cpu().numpy().copy()
        return build_pose_document(prediction, names, parents, reference, posed)
