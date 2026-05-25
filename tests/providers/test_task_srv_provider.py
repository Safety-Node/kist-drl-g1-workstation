"""
Tests for TaskSrvProvider (TASK-39).

Covers:
    - dataclass shape (SuccessCriterion subclasses + Scenario)
    - lifecycle (start with injected scenarios, stop)
    - on_audio keyword routing (match / case-insensitive / ignored when active)
    - tick() advance + timeout
    - dispatch via stub move connector
    - bind() wiring
"""

import math
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import pytest

from providers.task_srv_provider import (
    CompositeSuccess,
    JointStateSuccess,
    Scenario,
    SubTask,
    SuccessCriterion,
    TaskSrvConfig,
    TaskSrvProvider,
    TaskState,
    UwbPoseSuccess,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test gets a fresh provider instance."""
    TaskSrvProvider.reset()
    yield
    TaskSrvProvider.reset()


@dataclass
class _Cache:
    """Minimal UnitreeG1 TopicCache stand-in."""

    value: Any = None
    last_seen_ts: float = 0.0


class _StubG1:
    def __init__(self, uwb_pose=None, joint_state=None):
        self.uwb_pose = _Cache(value=uwb_pose)
        self.joint_state = _Cache(value=joint_state)


class _StubConnector:
    """
    Stand-in for OM1 ActionConnector — exposes ``async connect(payload)``
    matching the real ``ActionConnector[ActionConfig, MoveInput|SpeakInput]``
    contract that ``TaskSrvProvider`` now calls canonically.
    """

    def __init__(self):
        self.dispatched: List[str] = []

    async def connect(self, payload: Any) -> None:
        self.dispatched.append(payload.action)


def _simple_scenario(name="demo", trigger="hello", n_subtasks=2) -> Scenario:
    return Scenario(
        name=name,
        triggers=[trigger],
        sub_tasks=[
            SubTask(
                prompt=f"step {i}",
                success=JointStateSuccess(
                    target_joint="j0",
                    target_pos=1.0,
                    tolerance=0.01,
                    timeout_s=0.5,
                ),
            )
            for i in range(n_subtasks)
        ],
    )


# ---------------------------------------------------------------------------
# Config / construction
# ---------------------------------------------------------------------------


def test_default_config_values():
    cfg = TaskSrvConfig()
    assert cfg.tick_rate_hz == 10.0
    assert cfg.case_insensitive_keywords is True
    assert cfg.enable_speak_feedback is True


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
    assert p.active_scenario_name is None


# ---------------------------------------------------------------------------
# bind()
# ---------------------------------------------------------------------------


def test_bind_sets_dependencies():
    p = TaskSrvProvider()
    g1, mv, sp = _StubG1(), _StubConnector(), _StubConnector()
    p.bind(unitree_g1=g1, move_connector=mv, speak_connector=sp)
    assert p._unitree_g1 is g1
    assert p._move_connector is mv
    assert p._speak_connector is sp


def test_bind_speak_optional():
    p = TaskSrvProvider()
    p.bind(unitree_g1=_StubG1(), move_connector=_StubConnector())
    assert p._speak_connector is None


# ---------------------------------------------------------------------------
# Success criterion shape + evaluate()
# ---------------------------------------------------------------------------


def test_uwb_pose_success_within_tolerance():
    c = UwbPoseSuccess(target=(1.0, 0.0, 0.0), tolerance_xy=0.2, tolerance_yaw=math.inf)
    assert c.evaluate(_Cache(value={"x": 1.05, "y": -0.05}), _Cache()) is True


def test_uwb_pose_success_outside_tolerance():
    c = UwbPoseSuccess(target=(1.0, 0.0, 0.0), tolerance_xy=0.1, tolerance_yaw=math.inf)
    assert c.evaluate(_Cache(value={"x": 1.5, "y": 0.0}), _Cache()) is False


def test_uwb_pose_success_no_pose_value_yet():
    c = UwbPoseSuccess(target=(1.0, 0.0, 0.0))
    assert c.evaluate(_Cache(value=None), _Cache()) is False


def test_joint_state_success_dict_of_name_position_lists():
    c = JointStateSuccess(target_joint="j0", target_pos=1.0, tolerance=0.05)
    state = {"name": ["j0", "j1"], "position": [1.02, 0.0]}
    assert c.evaluate(_Cache(), _Cache(value=state)) is True


def test_joint_state_success_dict_mapping_form():
    c = JointStateSuccess(target_joint="j0", target_pos=1.0, tolerance=0.05)
    assert c.evaluate(_Cache(), _Cache(value={"j0": 1.04})) is True
    assert c.evaluate(_Cache(), _Cache(value={"j0": 1.2})) is False


def test_joint_state_success_missing_joint():
    c = JointStateSuccess(target_joint="missing", target_pos=1.0, tolerance=0.05)
    assert c.evaluate(_Cache(), _Cache(value={"name": ["j0"], "position": [1.0]})) is False


def test_composite_success_all_must_pass():
    children = [
        UwbPoseSuccess(target=(1.0, 0.0, 0.0), tolerance_xy=0.2, tolerance_yaw=math.inf),
        JointStateSuccess(target_joint="j0", target_pos=1.0, tolerance=0.05),
    ]
    c = CompositeSuccess(children=children, timeout_s=5.0)
    uwb_ok = _Cache(value={"x": 1.0, "y": 0.0})
    joint_ok = _Cache(value={"j0": 1.0})
    joint_bad = _Cache(value={"j0": 0.5})
    assert c.evaluate(uwb_ok, joint_ok) is True
    assert c.evaluate(uwb_ok, joint_bad) is False


def test_success_criterion_is_abc():
    """Cannot instantiate the bare abstract base."""
    with pytest.raises(TypeError):
        SuccessCriterion()  # type: ignore[abstract]


def test_scenario_shape():
    s = Scenario(
        name="refrigerator_pickup",
        triggers=["양상추 꺼내줘", "양상추 가져와"],
        sub_tasks=[
            SubTask(
                prompt="Walk to the refrigerator.",
                success=UwbPoseSuccess(
                    target=(1.5, 0.3, 0.0),
                    tolerance_xy=0.15,
                    tolerance_yaw=math.inf,
                    timeout_s=30.0,
                ),
            ),
        ],
    )
    assert s.name == "refrigerator_pickup"
    assert len(s.triggers) == 2
    assert len(s.sub_tasks) == 1
    assert s.sub_tasks[0].prompt.startswith("Walk")


# ---------------------------------------------------------------------------
# start() / stop() lifecycle
# ---------------------------------------------------------------------------


def test_start_loads_injected_scenarios():
    p = TaskSrvProvider()
    sc = _simple_scenario(trigger="냉장고에서 오이")
    p.start(scenarios=[sc])
    assert p._running is True
    assert p._scenarios == [sc]
    # case-insensitive index
    assert "냉장고에서 오이" in p._trigger_index


def test_start_raises_on_duplicate_trigger():
    p = TaskSrvProvider()
    a = _simple_scenario(name="a", trigger="overlap")
    b = _simple_scenario(name="b", trigger="overlap")
    with pytest.raises(ValueError):
        p.start(scenarios=[a, b])


def test_stop_resets_to_idle():
    p = TaskSrvProvider()
    p.bind(unitree_g1=_StubG1(), move_connector=_StubConnector())
    p.start(scenarios=[_simple_scenario()])
    p.on_audio("hello")  # → ACTIVE
    assert p.state == TaskState.ACTIVE
    p.stop()
    assert p.state == TaskState.IDLE
    assert p.active_sub_task is None
    assert p._running is False


# ---------------------------------------------------------------------------
# on_audio() keyword routing
# ---------------------------------------------------------------------------


def test_on_audio_triggers_scenario_and_dispatches_first_subtask():
    mv = _StubConnector()
    p = TaskSrvProvider()
    p.bind(unitree_g1=_StubG1(), move_connector=mv)
    p.start(scenarios=[_simple_scenario(trigger="냉장고")])
    p.on_audio("KAPEX: 냉장고 가까이 가줘")
    assert p.state == TaskState.ACTIVE
    assert mv.dispatched == ["step 0"]
    assert p.active_scenario_name == "demo"
    assert p.active_sub_task.prompt == "step 0"


def test_on_audio_ignored_when_already_active():
    mv = _StubConnector()
    p = TaskSrvProvider()
    p.bind(unitree_g1=_StubG1(), move_connector=mv)
    p.start(scenarios=[_simple_scenario(trigger="hello")])
    p.on_audio("hello there")
    p.on_audio("hello again")  # ignored
    assert mv.dispatched == ["step 0"]


def test_on_audio_no_match_keeps_idle():
    mv = _StubConnector()
    p = TaskSrvProvider()
    p.bind(unitree_g1=_StubG1(), move_connector=mv)
    p.start(scenarios=[_simple_scenario(trigger="hello")])
    p.on_audio("goodbye")
    assert p.state == TaskState.IDLE
    assert mv.dispatched == []


def test_on_audio_ignored_when_not_running():
    p = TaskSrvProvider()
    p.bind(unitree_g1=_StubG1(), move_connector=_StubConnector())
    # No .start() — _running is False.
    p.on_audio("hello")
    assert p.state == TaskState.IDLE


# ---------------------------------------------------------------------------
# tick() — advance / success / timeout
# ---------------------------------------------------------------------------


def test_tick_advances_on_success_then_finishes():
    mv = _StubConnector()
    g1 = _StubG1(joint_state={"j0": 1.0})  # matches JointStateSuccess(target_pos=1.0)
    p = TaskSrvProvider()
    p.bind(unitree_g1=g1, move_connector=mv)
    p.start(scenarios=[_simple_scenario(trigger="go", n_subtasks=2)])
    p.on_audio("go")
    # First sub-task already satisfied → advance to sub-task 1
    p.tick()
    assert mv.dispatched == ["step 0", "step 1"]
    # Second sub-task also satisfied → SUCCESS state
    p.tick()
    assert p.state == TaskState.SUCCESS
    # Next tick clears to IDLE
    p.tick()
    assert p.state == TaskState.IDLE
    assert p.active_sub_task is None


def test_tick_does_nothing_when_idle():
    p = TaskSrvProvider()
    p.tick()  # no provider state — no crash
    assert p.state == TaskState.IDLE


def test_tick_times_out_when_criterion_never_passes():
    mv = _StubConnector()
    g1 = _StubG1(joint_state={"j0": 0.0})  # never reaches target_pos=1.0
    p = TaskSrvProvider()
    p.bind(unitree_g1=g1, move_connector=mv)
    p.start(scenarios=[_simple_scenario(trigger="go", n_subtasks=1)])
    p.on_audio("go")
    # Force the dispatch timestamp into the past beyond timeout_s (0.5).
    p._sub_task_dispatch_ts = time.monotonic() - 5.0
    p.tick()
    assert p.state == TaskState.FAILED
    p.tick()
    assert p.state == TaskState.IDLE
