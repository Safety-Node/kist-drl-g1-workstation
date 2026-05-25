"""
Scenario: 냉장고에서 오이 가져오기 (KIST MBO 시연 1단계).

KAPEX: "G1, 냉장고에서 오이 하나 가져다주시겠어요?"
G1   : ① 냉장고로 이동 → ② 문 개방 → ③ 오이 파지 → ④ 조리대 tray에 안착.

Each sub-task is "blind" to its predecessors — the VLA Provider receives
just the prompt and the live observation. Success is judged from
``UnitreeG1Provider`` state only (no VLA self-report).

Targets / tolerances are placeholders; calibrate against the actual
demo space before showtime. Coordinates assume the UWB origin at the
KAPEX-facing edge of the workbench, x forward, y left.
"""

import math

from providers.task_srv_provider import (
    CompositeSuccess,
    JointStateSuccess,
    Scenario,
    SubTask,
    UwbPoseSuccess,
)


refrigerator_pickup = Scenario(
    name="refrigerator_pickup",
    triggers=[
        "냉장고에서 오이",
        "오이 가져와",
        "오이 하나 가져다",
        "오이 부탁",
    ],
    sub_tasks=[
        # ① 냉장고 앞 정렬
        SubTask(
            prompt="Walk to the refrigerator and face the door.",
            narration="냉장고 앞으로 갑니다.",
            success=UwbPoseSuccess(
                target=(2.10, -0.40, math.pi / 2),
                tolerance_xy=0.20,
                tolerance_yaw=math.inf,   # UWB yaw not yet trusted (kp_angular=0)
                timeout_s=25.0,
            ),
        ),
        # ② 냉장고 문 개방 (왼손 그리퍼가 열림 위치까지)
        SubTask(
            prompt="Open the refrigerator door with the left hand.",
            narration="냉장고 문을 엽니다.",
            success=JointStateSuccess(
                target_joint="left_gripper_finger_1",
                target_pos=0.85,
                tolerance=0.05,
                timeout_s=10.0,
            ),
        ),
        # ③ 오이 파지 (오른손 그리퍼가 잡힘 위치)
        SubTask(
            prompt="Pick up the cucumber from the top shelf.",
            narration="오이를 집겠습니다.",
            success=JointStateSuccess(
                target_joint="right_gripper_finger_1",
                target_pos=0.20,
                tolerance=0.03,
                timeout_s=15.0,
            ),
        ),
        # ④ 조리대 복귀 + tray 위 안착 (위치 + 오른손 release 동시 확인)
        SubTask(
            prompt="Return to the counter and place the cucumber on the tray.",
            narration="조리대에 오이를 놓습니다.",
            success=CompositeSuccess(
                children=[
                    UwbPoseSuccess(
                        target=(0.50, 0.30, -math.pi / 2),
                        tolerance_xy=0.15,
                        tolerance_yaw=math.inf,
                    ),
                    JointStateSuccess(
                        target_joint="right_gripper_finger_1",
                        target_pos=0.80,        # released
                        tolerance=0.05,
                    ),
                ],
                timeout_s=20.0,
            ),
        ),
    ],
)
