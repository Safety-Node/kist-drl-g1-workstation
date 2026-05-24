"""
Tests for TaskSrvProvider scaffold (TASK-39).

Placeholder — most methods are NotImplementedError today. Once the
scenario loader + dispatch + success poll are wired, add:

TODO(REQ-NEW-TASKSRV) [TASK-39]: scenario YAML loader fixture (tmp_path).
TODO(REQ-NEW-TASKSRV) [TASK-39]: on_audio triggers a scenario, MoveConnector
                                  stub records dispatched prompt.
TODO(REQ-NEW-TASKSRV) [TASK-39]: tick() with stub UnitreeG1Provider that
                                  exposes UWB pose / joint_state TopicCaches;
                                  assert success advances index.
TODO(REQ-NEW-TASKSRV) [TASK-39]: timeout path — freeze time, assert FAILED.
TODO(REQ-NEW-TASKSRV) [TASK-39]: state change callback fires on every
                                  IDLE → ACTIVE / ACTIVE → SUCCESS / FAILED.
"""

import pytest

from providers.task_srv_provider import (
    Scenario,
    SubTask,
    SuccessCriterion,
    SuccessKind,
    TaskSrvConfig,
    TaskSrvProvider,
    TaskState,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test gets a fresh provider instance."""
    TaskSrvProvider.reset()
    yield
    TaskSrvProvider.reset()


def test_default_config_values():
    cfg = TaskSrvConfig()
    assert cfg.tick_rate_hz == 10.0
    assert cfg.case_insensitive_keywords is True
    assert cfg.enable_speak_feedback is True
    assert cfg.scenarios_dir.endswith("/")


def test_provider_initializes_with_defaults():
    p = TaskSrvProvider()
    assert p._config.tick_rate_hz == 10.0
    assert p._scenarios == []
    assert p._trigger_index == {}
    assert p._state == TaskState.IDLE
    assert p._active_scenario is None
    assert p._active_sub_task_idx == -1
    assert p._running is False
    assert p._unitree_g1 is None
    assert p._move_connector is None
    assert p._speak_connector is None


def test_state_property_starts_idle():
    p = TaskSrvProvider()
    assert p.state == TaskState.IDLE
    assert p.active_sub_task is None


def test_bind_sets_dependencies():
    p = TaskSrvProvider()

    class _StubG1:
        pass

    class _StubMove:
        pass

    class _StubSpeak:
        pass

    g1, mv, sp = _StubG1(), _StubMove(), _StubSpeak()
    p.bind(unitree_g1=g1, move_connector=mv, speak_connector=sp)
    assert p._unitree_g1 is g1
    assert p._move_connector is mv
    assert p._speak_connector is sp


def test_bind_speak_optional():
    p = TaskSrvProvider()
    p.bind(unitree_g1=object(), move_connector=object())
    assert p._speak_connector is None


def test_register_state_callback():
    p = TaskSrvProvider()
    received = []

    def _cb(state):
        received.append(state)

    p.register_state_callback(_cb)
    assert p._on_state_change is _cb


def test_lifecycle_methods_are_stubs():
    p = TaskSrvProvider()
    with pytest.raises(NotImplementedError):
        p.start()
    with pytest.raises(NotImplementedError):
        p.stop()
    with pytest.raises(NotImplementedError):
        p.on_audio("양상추 꺼내줘")
    with pytest.raises(NotImplementedError):
        p.tick()


# --- Dataclass shape (catches schema drift before YAML loader lands) -------


def test_success_criterion_uwb_pose_shape():
    c = SuccessCriterion(
        kind=SuccessKind.UWB_POSE,
        target={"x": 1.5, "y": 0.3, "yaw": 0.0},
        tolerance={"xy": 0.15, "yaw": 0.2},
        timeout_s=30.0,
    )
    assert c.kind == SuccessKind.UWB_POSE
    assert c.target["x"] == 1.5
    assert c.children == []


def test_success_criterion_joint_state_shape():
    c = SuccessCriterion(
        kind=SuccessKind.JOINT_STATE,
        target_joint="left_gripper_finger_1",
        target_pos=0.85,
        joint_tolerance=0.05,
        timeout_s=10.0,
    )
    assert c.target_joint == "left_gripper_finger_1"
    assert c.target_pos == pytest.approx(0.85)


def test_success_criterion_composite_shape():
    child_a = SuccessCriterion(kind=SuccessKind.UWB_POSE)
    child_b = SuccessCriterion(kind=SuccessKind.JOINT_STATE)
    c = SuccessCriterion(kind=SuccessKind.COMPOSITE, children=[child_a, child_b])
    assert len(c.children) == 2


def test_scenario_shape():
    s = Scenario(
        name="refrigerator_pickup",
        triggers=["양상추 꺼내줘", "양상추 가져와"],
        sub_tasks=[
            SubTask(
                prompt="Walk to the refrigerator.",
                success=SuccessCriterion(kind=SuccessKind.UWB_POSE),
            ),
        ],
    )
    assert s.name == "refrigerator_pickup"
    assert len(s.triggers) == 2
    assert len(s.sub_tasks) == 1
    assert s.sub_tasks[0].prompt.startswith("Walk")
