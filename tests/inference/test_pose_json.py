"""验证姿态交换协议的坐标、骨架及数值约束。"""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from gem.inference.pose_json import BodyPoseExporter, build_pose_document, validate_prediction


def sample_prediction():
    """构造可手算的两关节相机坐标结果。"""
    return {
        "format": "gem_pose_predictions_v1",
        "task": "body",
        "mode": "image",
        "source": "人物.jpg",
        "release": {"pose_release_id": "测试模型"},
        "coordinates": {"geometry": "camera_meters"},
        "frames": [
            {
                "image_size": [100, 80],
                "instances": [
                    {
                        "camera_translation": [1, 2, 3],
                        "joints_camera": [[1, 2, 3], [1, 3, 3]],
                        "parameters": {},
                    }
                ],
            }
        ],
    }


def test_相机坐标转换保留髋部平移及旋转():
    reference = np.tile(np.eye(4), (2, 1, 1))
    reference[1, :3, 3] = [0, 1, 0]
    posed = reference.copy()
    posed[:, :3, 3] += [1, 2, 3]
    posed[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    doc = build_pose_document(sample_prediction(), ["Hips", "Child"], [-1, 0], reference, posed)
    assert doc["camera_translation"] == [1, -2, 3]
    assert doc["joints"][1]["parent_index"] == 0
    assert doc["joints"][1]["posed_camera_position"] == [1, -3, 3]
    assert doc["joints"][0]["posed_camera_position"] == [1, -2, 3]
    np.testing.assert_allclose(
        doc["joints"][1]["posed_world_rotation"], [0, 0, -(2**-0.5), 2**-0.5], atol=1e-7
    )
    np.testing.assert_allclose(doc["joints"][0]["reference_camera_position"], [1, -2, 3])
    np.testing.assert_allclose(doc["joints"][1]["reference_camera_position"], [1, -1, 3])
    np.testing.assert_allclose(
        doc["joints"][0]["reference_world_rotation"], [1, 0, 0, 0], atol=1e-7
    )
    assert doc["schema_version"] == "gemx_unity_pose_v2"
    assert doc["coordinates"]["position_origin"] == "camera"
    assert doc["coordinates"]["rotation_space"] == "camera"
    assert "posed_root_relative_position" not in doc["joints"][0]


def test_真实骨架参考与预测均正立且共用相机原点():
    """使用官方运行时捕获参考根旋转清零及位置被归零的回归。"""
    import os
    from pathlib import Path

    torch = pytest.importorskip("torch")
    pytest.importorskip("soma")
    assets = Path(
        os.environ.get("SOMA_ASSETS", Path(__file__).resolve().parents[2] / "assets/soma")
    )
    if not (assets / "SOMA_neutral.npz").is_file():
        pytest.skip("需要 SOMA 资产，可通过 SOMA_ASSETS 指定目录")

    exporter = BodyPoseExporter(assets, device="cpu")
    exporter.initialize()
    # 绕 X 轴半周，将原生正立人体置入 Y 向下的相机坐标。
    parameters = {
        "global_orient": torch.diag(torch.tensor([1.0, -1.0, -1.0])).reshape(1, 3, 3),
        "body_pose": torch.eye(3).repeat(76, 1, 1),
        "identity_coeffs": torch.zeros(128),
        "scale_params": torch.ones(56),
    }
    translation = torch.tensor([1.0, -0.25, 3.0])
    with torch.inference_mode():
        geometry = exporter.geometry(
            {key: value.unsqueeze(0) for key, value in parameters.items()},
            transl=translation.unsqueeze(0),
        )
    prediction = sample_prediction()
    row = prediction["frames"][0]["instances"][0]
    row.update(
        parameters=parameters, camera_translation=translation, joints_camera=geometry["joints"][0]
    )
    doc = exporter.export(prediction)
    by_name = {joint["name"]: joint for joint in doc["joints"]}
    for field in ["reference_camera_position", "posed_camera_position"]:
        np.testing.assert_allclose(by_name["Hips"][field], [1.0, 0.25, 3.0], atol=1e-5)
        assert by_name["Head"][field][1] > by_name["Hips"][field][1]
        assert by_name["LeftFoot"][field][1] < by_name["Hips"][field][1]
    for joint in doc["joints"]:
        np.testing.assert_allclose(
            joint["reference_camera_position"], joint["posed_camera_position"], atol=1e-5
        )
        # 同一中性朝向下参考与预测旋转也应相等，防止仅修正位置。
        reference = np.asarray(joint["reference_world_rotation"])
        posed = np.asarray(joint["posed_world_rotation"])
        assert abs(reference @ posed) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    "field,value", [("task", "hand"), ("mode", "framewise_video"), ("format", "unknown")]
)
def test_rejects_other_tasks_and_formats(field, value):
    p = sample_prediction()
    p[field] = value
    with pytest.raises(ValueError):
        validate_prediction(p)


def test_rejects_multiple_people_and_empty_results():
    p = sample_prediction()
    p["frames"][0]["instances"] *= 2
    with pytest.raises(ValueError, match="一个人物"):
        validate_prediction(p)
    p["frames"][0]["instances"] = []
    with pytest.raises(ValueError, match="一个人物"):
        validate_prediction(p)


def test_rejects_nonfinite_rotations_and_invalid_hierarchy():
    transforms = np.tile(np.eye(4), (2, 1, 1))
    bad = transforms.copy()
    bad[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="有限"):
        build_pose_document(sample_prediction(), ["Hips", "Child"], [-1, 0], transforms, bad)
    for parents in [[1, 0], [-1, 8], [-1, 1]]:
        with pytest.raises(ValueError):
            build_pose_document(
                sample_prediction(), ["Hips", "Child"], parents, transforms, transforms
            )
    bad = transforms.copy()
    bad[1, 0, 0] = -1
    with pytest.raises(ValueError, match="旋转"):
        build_pose_document(sample_prediction(), ["Hips", "Child"], [-1, 0], transforms, bad)


def test_rejects_reconstruction_that_does_not_match_saved_joints():
    p = copy.deepcopy(sample_prediction())
    transforms = np.tile(np.eye(4), (2, 1, 1))
    with pytest.raises(ValueError, match="重建"):
        build_pose_document(p, ["Hips", "Child"], [-1, 0], transforms, transforms)


def test_rejects_out_of_range_parent_reached_via_another_joint():
    transforms = np.tile(np.eye(4), (3, 1, 1))
    with pytest.raises(ValueError, match="索引越界"):
        build_pose_document(
            sample_prediction(), ["Hips", "Child", "Grandchild"], [-1, 2, 9], transforms, transforms
        )


def test_rejects_fractional_parent_indices():
    transforms = np.tile(np.eye(4), (2, 1, 1))
    with pytest.raises(ValueError, match="整数"):
        build_pose_document(
            sample_prediction(), ["Hips", "Child"], [-1, 0.5], transforms, transforms
        )


def test_explicit_initialization_loads_lazy_runtime_before_processing(tmp_path):
    class BrokenModel:
        @property
        def runtime(self):
            raise ValueError("SOMA 资产损坏")

    geometry = SimpleNamespace(body_model=BrokenModel())
    exporter = BodyPoseExporter(tmp_path, geometry=geometry)
    with pytest.raises(ValueError, match="资产损坏"):
        exporter.initialize()
