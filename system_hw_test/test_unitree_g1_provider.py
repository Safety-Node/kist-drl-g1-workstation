"""
Integration tests for UnitreeG1Provider — CONV-009 verification.

Prerequisites
-------------
1. G1 로봇 전원 ON + NX onboard stack (run_onboard.sh) 실행 중
2. PC에서 env.sh 소스: ``source env.sh``
3. 이 스크립트 실행: ``python3 system_hw_test/test_unitree_g1_provider.py``

CONV-009 Verification
---------------------
1. NX dummy publish → TopicCache.value update + last_seen_ts 진행
2. 연결 끊김 → stale(now, ttl_s) True
3. Reliable 토픽만 끊으면 comm_bridge_alive() False (BestEffort는 cache 흐름)
4. publish 시 watchdog fail → 로그 한 줄 + 무시 (raise 없음)
"""

import sys
import time
import os
import logging

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger("hw_test")

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"


def _result(label: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    print(f"  {tag} {label}" + (f" — {detail}" if detail else ""))
    return ok


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CONV-009-4 (static): publish watchdog + lifecycle guard — no ROS needed
# ---------------------------------------------------------------------------
def test_publish_guard_no_ros() -> bool:
    """Verify publish methods log+drop before start() (no rclpy needed)."""
    _section("CONV-009-4a: publish guard before start() [static]")
    from providers.unitree_g1_provider import UnitreeG1Provider
    UnitreeG1Provider.reset()  # clear singleton for clean test
    p = UnitreeG1Provider()

    errors = []

    for name, fn in [
        ("publish_loco_cmd",       lambda: p.publish_loco_cmd({"name": "StandUp"})),
        ("publish_loco_cmd (bad)", lambda: p.publish_loco_cmd({"name": "BadCmd"})),
        ("publish_twist",          lambda: p.publish_twist(1.0, 0.0, 0.0)),
        ("publish_audio_out",      lambda: p.publish_audio_out(b"\x00" * 100)),
        ("publish_joint_chunk_arm",lambda: p.publish_joint_chunk_arm(
            {"chunk_id": 1, "steps": []}
        )),
        ("publish_joint_chunk_low",lambda: p.publish_joint_chunk_low(
            {"chunk_id": 1, "steps": []}
        )),
        ("publish_audio_out (>65500B)", lambda: p.publish_audio_out(b"\x00" * 70000)),
    ]:
        try:
            fn()
            ok = True
        except Exception as e:
            ok = False
            errors.append(f"{name}: {e}")
        _result(f"{name}: no raise", ok)

    return len(errors) == 0


# ---------------------------------------------------------------------------
# CONV-009-1/2/3/4b — live ROS tests (NX must be running)
# ---------------------------------------------------------------------------
def test_topic_cache_update(provider) -> bool:
    """CONV-009-1: TopicCache.value update + last_seen_ts 진행."""
    _section("CONV-009-1: TopicCache update (NX must be publishing)")
    WAIT_S = 5.0
    print(f"  {INFO} Waiting up to {WAIT_S}s for /bridge/sensors/imu …")

    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        cache = provider._imu_base
        if cache.last_seen_ts > 0.0:
            break
        time.sleep(0.1)

    cache = provider._imu_base
    ok_value = cache.value is not None
    ok_ts = cache.last_seen_ts > 0.0
    ok_stale = not cache.stale(time.monotonic(), ttl_s=2.0)

    _result("imu_base.value is not None", ok_value,
            f"type={type(cache.value).__name__}" if ok_value else "no message received")
    _result("imu_base.last_seen_ts > 0", ok_ts,
            f"ts={cache.last_seen_ts:.3f}")
    _result("imu_base.stale(ttl=2s) == False", ok_stale)

    # Check a Reliable topic too
    print(f"\n  {INFO} Checking Reliable topics (estop/buf_state/speaker_state) …")
    for attr, label in [
        ("_estop",         "/bridge/safety/estop"),
        ("_buf_state",     "/bridge/motor/buf_state"),
        ("_speaker_state", "/bridge/audio/speaker_state"),
    ]:
        c = getattr(provider, attr)
        has_msg = c.last_seen_ts > 0.0
        _result(f"{label} received", has_msg,
                "check NX onboard stack publishes this topic" if not has_msg else "")

    return ok_value and ok_ts and ok_stale


def test_stale_detection(provider) -> bool:
    """CONV-009-2: stale(now, ttl_s) True after TTL expires."""
    _section("CONV-009-2: stale() detection")
    from providers.unitree_g1_provider import TopicCache

    # Manufacture a cache entry from 2 seconds ago
    old_ts = time.monotonic() - 2.0
    cache = TopicCache(value="fake", last_seen_ts=old_ts)
    ok_stale = cache.stale(time.monotonic(), ttl_s=1.0)
    ok_fresh = not cache.stale(time.monotonic(), ttl_s=10.0)
    ok_empty = TopicCache().stale(time.monotonic(), ttl_s=0.001)

    _result("stale when age > ttl", ok_stale, f"age≈2s, ttl=1s")
    _result("not stale when age < ttl", ok_fresh, f"age≈2s, ttl=10s")
    _result("empty cache always stale", ok_empty)

    return ok_stale and ok_fresh and ok_empty


def test_comm_bridge_alive(provider) -> bool:
    """CONV-009-3: comm_bridge_alive() depends on Reliable topics only."""
    _section("CONV-009-3: comm_bridge_alive() — Reliable-only liveness")

    alive = provider.comm_bridge_alive()
    print(f"  {INFO} comm_bridge_alive() = {alive}")

    # Check if any reliable topic was seen
    now = time.monotonic()
    ttl_s = provider._heartbeat_timeout_ms / 1000.0
    estop_alive = not provider._estop.stale(now, ttl_s)
    buf_alive   = not provider._buf_state.stale(now, ttl_s)
    spk_alive   = not provider._speaker_state.stale(now, ttl_s)

    _result("estop freshness", estop_alive,
            "run_onboard.sh에서 safety/estop 발행되는지 확인" if not estop_alive else "")
    _result("buf_state freshness", buf_alive,
            "run_onboard.sh에서 motor/buf_state 발행되는지 확인" if not buf_alive else "")
    _result("speaker_state freshness", spk_alive,
            "run_onboard.sh에서 audio/speaker_state 발행되는지 확인" if not spk_alive else "")

    expected_alive = estop_alive or buf_alive or spk_alive
    ok = alive == expected_alive
    _result(f"comm_bridge_alive() == {expected_alive}", ok)

    if not expected_alive:
        print(f"\n  {SKIP} No Reliable topics received — "
              "NX 측 comm_bridge가 Reliable 토픽을 발행하지 않으면 comm_bridge_alive()=False는 정상.")

    return ok


def test_publish_watchdog_drop(provider) -> bool:
    """CONV-009-4b: publish when watchdog fails → log + no raise."""
    _section("CONV-009-4b: publish with watchdog fail → log+drop")

    # Force a stale state by temporarily zeroing caches
    from providers.unitree_g1_provider import TopicCache
    orig_estop = provider._estop
    orig_buf = provider._buf_state
    orig_spk = provider._speaker_state

    provider._estop = TopicCache()         # never received
    provider._buf_state = TopicCache()
    provider._speaker_state = TopicCache()

    errors = []
    try:
        for name, fn in [
            ("publish_loco_cmd",         lambda: provider.publish_loco_cmd({"name": "StandUp"})),
            ("publish_twist",            lambda: provider.publish_twist(1.0, 0.0, 0.0)),
            ("publish_audio_out",        lambda: provider.publish_audio_out(b"\x00" * 100)),
            ("publish_joint_chunk_arm",  lambda: provider.publish_joint_chunk_arm(
                {"chunk_id": 1, "steps": []}
            )),
            ("publish_joint_chunk_low",  lambda: provider.publish_joint_chunk_low(
                {"chunk_id": 1, "steps": []}
            )),
        ]:
            try:
                fn()
                ok = True
            except Exception as e:
                ok = False
                errors.append(f"{name}: {e}")
            _result(f"{name}: no raise when watchdog=False", ok)
    finally:
        # Restore
        provider._estop = orig_estop
        provider._buf_state = orig_buf
        provider._speaker_state = orig_spk

    return len(errors) == 0


def test_push_callbacks(provider) -> bool:
    """Verify audio and estop push callbacks fire when TopicCache is updated."""
    _section("CONV-009 bonus: push callback dispatch")

    audio_received = []
    estop_received = []

    def audio_cb(pcm: bytes, ts: float) -> None:
        audio_received.append((pcm, ts))

    def estop_cb(active: bool, ts: float) -> None:
        estop_received.append((active, ts))

    provider.register_audio_callback(audio_cb)
    provider.register_estop_callback(estop_cb)

    # Double-register should be idempotent
    provider.register_audio_callback(audio_cb)

    import threading
    with provider._cb_lock:
        n_audio = len(provider._audio_callbacks)
        n_estop = len(provider._estop_callbacks)

    ok_dedup = _result("audio callback dedup (registered once)", n_audio == 1,
                       f"got {n_audio}")

    # Simulate a callback fire (directly call _on_estop)
    from g1_onboard_msgs.msg import EstopFlag
    fake_estop = EstopFlag()
    fake_estop.active = True
    provider._on_estop(fake_estop)

    time.sleep(0.05)
    ok_estop = _result("estop callback fired", len(estop_received) == 1,
                       f"received {len(estop_received)} calls")
    if estop_received:
        ok_active = _result("estop active=True", estop_received[0][0] is True)
    else:
        ok_active = False

    # Cleanup
    provider.unregister_audio_callback(audio_cb)
    provider.unregister_estop_callback(estop_cb)
    provider.unregister_estop_callback(estop_cb)  # double-unregister should be no-op

    with provider._cb_lock:
        ok_unreg = _result("unregister no-op for unknown cb", len(provider._audio_callbacks) == 0)

    return ok_dedup and ok_estop and ok_active and ok_unreg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "="*60)
    print("  UnitreeG1Provider Integration Test — CONV-009")
    print("="*60)
    print(f"  {INFO} ROS_DOMAIN_ID  = {os.environ.get('ROS_DOMAIN_ID', '(not set)')}")
    print(f"  {INFO} RMW_IMPL       = {os.environ.get('RMW_IMPLEMENTATION', '(not set)')}")
    print(f"  {INFO} CYCLONEDDS_URI = {os.environ.get('CYCLONEDDS_URI', '(not set)')}")

    results = {}

    # Static tests (no ROS)
    results["publish_guard_no_ros"] = test_publish_guard_no_ros()

    # Live ROS tests
    print(f"\n{INFO} Initializing UnitreeG1Provider …")
    from providers.unitree_g1_provider import UnitreeG1Provider
    UnitreeG1Provider.reset()
    provider = UnitreeG1Provider(
        heartbeat_timeout_ms=2000,
        sensor_ttl_ms=500,
        state_ttl_ms=2000,
    )

    try:
        provider.start()
        print(f"  {INFO} Provider started. Waiting 2s for DDS discovery …")
        time.sleep(2.0)

        results["topic_cache_update"] = test_topic_cache_update(provider)
        results["stale_detection"]    = test_stale_detection(provider)
        results["comm_bridge_alive"]  = test_comm_bridge_alive(provider)
        results["publish_watchdog"]   = test_publish_watchdog_drop(provider)
        results["push_callbacks"]     = test_push_callbacks(provider)

    finally:
        provider.stop()

    # Summary
    _section("SUMMARY")
    all_pass = True
    for name, ok in results.items():
        tag = PASS if ok else FAIL
        print(f"  {tag} {name}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print(f"  {PASS} All tests passed")
        sys.exit(0)
    else:
        print(f"  {FAIL} Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
