"""验证姿态交换协议的坐标、骨架及数值约束。"""

import copy

import numpy as np
import pytest

from gem.inference.pose_json import build_pose_document, validate_prediction


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


def test_camera_basis_converts_rotation_and_root_relative_positions():
    reference = np.tile(np.eye(4), (2, 1, 1))
    reference[1, :3, 3] = [0, 1, 0]
    posed = reference.copy()
    posed[:, :3, 3] += [1, 2, 3]
    posed[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    doc = build_pose_document(sample_prediction(), ["Hips", "Child"], [-1, 0], reference, posed)
    assert doc["camera_translation"] == [1, -2, 3]
    assert doc["joints"][1]["parent_index"] == 0
    assert doc["joints"][1]["posed_root_relative_position"] == [0, -1, 0]
    assert doc["joints"][0]["posed_root_relative_position"] == [0, 0, 0]
    np.testing.assert_allclose(
        doc["joints"][1]["posed_world_rotation"], [0, 0, -(2**-0.5), 2**-0.5], atol=1e-7
    )
    assert doc["joints"][0]["reference_world_rotation"] == [0, 0, 0, 1]


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
