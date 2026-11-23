"""Tests for round-scoped de-identification (irreversible codes)."""

from __future__ import annotations

import io
import os
import sqlite3
import threading

import pytest

import app as app_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    application = app_module.app
    application.config.update(
        TESTING=True,
        DB_PATH=db_path,
        SECRET_KEY="test-secret",
        DEID_KEY_PATH=None,
    )
    app_module.bootstrap(db_path)
    with application.test_client() as test_client:
        yield test_client
    app_module._schema_ready_paths.discard(db_path)


def make_csv(rows):
    out = io.StringIO()
    out.write("user_id,group\n")
    for user_id, group in rows:
        out.write(f"{user_id},{group}\n")
    return out.getvalue().encode("utf-8")


def upload(client, period, rows, follow=True):
    return client.post(
        f"/period/{period}",
        data={"file": (io.BytesIO(make_csv(rows)), f"p{period}.csv")},
        content_type="multipart/form-data",
        follow_redirects=follow,
    )


def configure(client, counts, probs):
    data = {"periods": str(len(counts))}
    for i, (count, prob) in enumerate(zip(counts, probs), start=1):
        data[f"count_{i}"] = str(count)
        data[f"prob_{i}"] = str(prob)
    return client.post("/plan", data=data, follow_redirects=True)


def rows_for(period, count, b_count, prefix="u"):
    return [
        (f"{prefix}{period}-{i}", "B" if i < b_count else "A") for i in range(count)
    ]


def activate(client, serial=None):
    if serial is None:
        serial = current_serial(client)
    return client.post(
        "/round/deid", data={"round_serial": str(serial)}, follow_redirects=True
    )


def fetch(application, sql, params=()):
    conn = sqlite3.connect(application.config["DB_PATH"])
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def assignment_ids(application):
    return {row["user_id"] for row in fetch(application, "SELECT user_id FROM assignments")}


def is_code(value: str) -> bool:
    return value.startswith(app_module.DEID_CODE_PREFIX)


def current_serial(client) -> int:
    """Read the serial the browser would submit from the overview page."""
    text = client.get("/").get_data(as_text=True)
    marker = 'name="round_serial" value="'
    if marker in text:
        return int(text.split(marker, 1)[1].split('"', 1)[0])
    # Already active: the form is gone.  Fall back to the DB.
    row = fetch(app_module.app, "SELECT round_serial FROM round_meta WHERE id = 1")
    return int(row[0]["round_serial"])


# ---------------------------------------------------------------------------
# Activation basics
# ---------------------------------------------------------------------------


def test_deid_box_visible_and_inactive_by_default(client):
    configure(client, [100, 900], ["0.5", "0.5"])
    text = client.get("/").get_data(as_text=True)
    assert "本轮去标识" in text
    assert "未生效" in text
    assert "启用本轮去标识" in text


def test_activate_with_partial_list_still_accepts_other_periods(client):
    configure(client, [300, 300, 400], ["0.5", "0.4", "0.6"])
    upload(client, 2, rows_for(2, 300, 120))
    before_results = fetch(app_module.app, "SELECT * FROM results")

    resp = activate(client)
    text = resp.get_data(as_text=True)
    assert "去标识已生效" in text
    assert "已生效" in text

    ids = assignment_ids(app_module.app)
    assert len(ids) == 300
    assert all(is_code(value) for value in ids)
    assert not any(value.startswith("u2-") for value in ids)

    # Period 1 is still missing: its submission must still be accepted.
    resp = upload(client, 1, rows_for(1, 300, 150))
    assert "已接收" in resp.get_data(as_text=True)
    ids = assignment_ids(app_module.app)
    assert len(ids) == 600
    assert all(is_code(value) for value in ids)

    # Results computed before the switch are unchanged in substance: both
    # periods verified once the prefix is complete.
    after_results = fetch(app_module.app, "SELECT * FROM results ORDER BY period")
    assert [row["period"] for row in after_results] == [1, 2]
    assert after_results[1]["observed_b"] == 270
    assert before_results == []  # gap before switch: nothing verified yet


def test_activation_preserves_existing_verification_results(client):
    configure(client, [400, 600], ["0.5", "0.5"])
    upload(client, 1, rows_for(1, 400, 200))
    before = fetch(app_module.app, "SELECT * FROM results")[0]
    activate(client)
    after = fetch(app_module.app, "SELECT * FROM results")[0]
    for key in (
        "period",
        "verified_users",
        "observed_b",
        "expected_b",
        "variance",
        "tail_low",
        "tail_high",
        "p_value",
        "threshold",
        "flagged",
    ):
        assert after[key] == before[key]


def test_activation_with_no_uploads_yet(client):
    configure(client, [500, 500], ["0.5", "0.5"])
    resp = activate(client)
    assert "去标识已生效" in resp.get_data(as_text=True)
    assert "0 条" in resp.get_data(as_text=True)
    upload(client, 1, rows_for(1, 500, 250))
    ids = assignment_ids(app_module.app)
    assert len(ids) == 500
    assert all(is_code(value) for value in ids)


def test_reactivating_only_reports_current_state(client):
    configure(client, [100, 900], ["0.5", "0.5"])
    upload(client, 1, rows_for(1, 100, 50))
    activate(client)
    ids_before = assignment_ids(app_module.app)
    state_before = fetch(app_module.app, "SELECT * FROM deid_state")[0]

    resp = activate(client)
    text = resp.get_data(as_text=True)
    assert "已生效" in text
    assert "未做任何改动" in text
    assert assignment_ids(app_module.app) == ids_before
    state_after = fetch(app_module.app, "SELECT * FROM deid_state")[0]
    assert state_after["activated_at"] == state_before["activated_at"]
    assert state_after["activated_users"] == state_before["activated_users"]


# ---------------------------------------------------------------------------
# Codes: deterministic within a round, different across rounds
# ---------------------------------------------------------------------------


def test_same_user_gets_same_code_within_round_including_later_uploads(client):
    configure(client, [2, 998], ["0.5", "0.5"])
    upload(client, 1, [("u1", "A"), ("u2", "B")])
    activate(client)
    period1 = {
        row["user_id"] for row in fetch(app_module.app, "SELECT user_id FROM assignments")
    }
    # Same original ids in another period would be rejected, but the code
    # generated for the same id must be stable.  Verify through the public
    # helper with the persisted key and serial.
    key = app_module.load_deid_key(create=False)
    serial = int(fetch(app_module.app, "SELECT round_serial FROM round_meta")[0][0])
    expected = {app_module.deid_code(key, serial, "u1"), app_module.deid_code(key, serial, "u2")}
    assert period1 == expected


def test_reset_starts_new_round_with_different_codes(client):
    configure(client, [2, 998], ["0.5", "0.5"])
    upload(client, 1, [("u1", "A"), ("u2", "B")])
    activate(client)
    old_ids = assignment_ids(app_module.app)
    assert all(is_code(value) for value in old_ids)

    client.post("/round/reset", data={"confirm": "RESET"}, follow_redirects=True)
    text = client.get("/").get_data(as_text=True)
    assert "未生效" in text
    assert fetch(app_module.app, "SELECT COUNT(*) c FROM assignments")[0]["c"] == 0

    # New round starts with original ids until it is enabled separately.
    upload(client, 1, [("u1", "A"), ("u2", "B")])
    assert assignment_ids(app_module.app) == {"u1", "u2"}

    activate(client)
    new_ids = assignment_ids(app_module.app)
    assert all(is_code(value) for value in new_ids)
    assert new_ids != old_ids


def test_stale_switch_request_cannot_activate_new_round(client):
    configure(client, [2, 998], ["0.5", "0.5"])
    upload(client, 1, [("u1", "A"), ("u2", "B")])
    old_serial = current_serial(client)
    client.post("/round/reset", data={"confirm": "RESET"}, follow_redirects=True)
    upload(client, 1, [("u1", "A"), ("u2", "B")])

    resp = client.post(
        "/round/deid", data={"round_serial": str(old_serial)}, follow_redirects=True
    )
    assert "旧的去标识请求不再生效" in resp.get_data(as_text=True)
    assert assignment_ids(app_module.app) == {"u1", "u2"}
    assert fetch(app_module.app, "SELECT active FROM deid_state")[0]["active"] == 0


def test_cumulative_verification_matches_control_round(client, tmp_path):
    """A round where the switch happens mid-way must yield the same numbers
    as a control round that never switches on."""
    plan = [250, 250, 250, 250]
    probs = ["0.5", "0.4", "0.6", "0.5"]
    data = {
        1: rows_for(1, 250, 120),
        2: rows_for(2, 250, 140),
        3: rows_for(3, 250, 160),
        4: rows_for(4, 250, 110),
    }

    configure(client, plan, probs)
    upload(client, 1, data[1])
    upload(client, 3, data[3])
    activate(client)  # switch while period 2 is still missing
    upload(client, 2, data[2])
    upload(client, 4, data[4])
    switched = fetch(app_module.app, "SELECT * FROM results ORDER BY period")

    # Control round on the same fixture: identical data, no switch.
    client.post("/round/reset", data={"confirm": "RESET"}, follow_redirects=True)
    # Reset keeps the plan (including probabilities), so just re-upload.
    upload(client, 1, data[1])
    upload(client, 2, data[2])
    upload(client, 3, data[3])
    upload(client, 4, data[4])
    control = fetch(app_module.app, "SELECT * FROM results ORDER BY period")

    assert [r["period"] for r in switched] == [1, 2, 3, 4]
    assert [r["period"] for r in control] == [1, 2, 3, 4]
    for left, right in zip(switched, control):
        for column in (
            "verified_users",
            "observed_b",
            "expected_b",
            "variance",
            "tail_low",
            "tail_high",
            "p_value",
            "threshold",
            "flagged",
        ):
            assert left[column] == right[column], column


def test_stored_fingerprints_cannot_identify_users(client):
    configure(client, [100, 900], ["0.5", "0.5"])
    rows = rows_for(1, 100, 50)
    upload(client, 1, rows)
    activate(client)
    stored = fetch(app_module.app, "SELECT checksum FROM uploads")[0]["checksum"]
    # Neither the raw id nor the historical unkeyed digest may survive.
    import hashlib

    canonical = "\n".join(
        f"{uid}\t{grp}" for uid, grp in sorted(rows, key=lambda r: (r[0], r[1]))
    )
    plain = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert stored != plain
    assert "u1-0" not in stored


# ---------------------------------------------------------------------------
# Re-uploads and duplicates after the switch
# ---------------------------------------------------------------------------


def test_identical_reupload_of_original_csv_acknowledged_after_switch(client):
    configure(client, [100, 900], ["0.5", "0.5"])
    rows = rows_for(1, 100, 50)
    upload(client, 1, rows)
    activate(client)
    resp = upload(client, 1, rows)
    text = resp.get_data(as_text=True)
    assert "相同明细已接收" in text
    assert fetch(app_module.app, "SELECT COUNT(*) c FROM uploads")[0]["c"] == 1
    assert fetch(app_module.app, "SELECT COUNT(*) c FROM assignments")[0]["c"] == 100


def test_changed_reupload_rejected_after_switch_and_keeps_results(client):
    configure(client, [100, 900], ["0.5", "0.5"])
    upload(client, 1, rows_for(1, 100, 50))
    activate(client)
    changed = rows_for(1, 100, 50)
    changed[0] = (changed[0][0], "B")
    resp = upload(client, 1, changed)
    assert "拒收" in resp.get_data(as_text=True)
    assert fetch(app_module.app, "SELECT COUNT(*) c FROM uploads")[0]["c"] == 1
    assert fetch(app_module.app, "SELECT observed_b FROM results")[0]["observed_b"] == 50


def test_cross_period_duplicate_rejected_after_switch(client):
    configure(client, [3, 997], ["0.5", "0.5"])
    upload(client, 1, [("u1", "A"), ("u2", "A"), ("u3", "A")])
    activate(client)
    resp = upload(client, 2, [("u2", "B")] + [(f"v{i}", "A") for i in range(996)])
    text = resp.get_data(as_text=True)
    assert "只计一次" in text
    # The message reports the original id, never the stored code.
    assert "u2" in text
    assert "deid-" not in text
    assert fetch(app_module.app, "SELECT COUNT(*) c FROM assignments")[0]["c"] == 3


def test_cross_period_duplicate_across_activation_boundary_rejected(client):
    configure(client, [2, 998], ["0.5", "0.5"])
    upload(client, 1, [("u1", "A"), ("u2", "B")])
    activate(client)
    resp = upload(client, 2, [("u1", "A"), ("u2", "B")] + [(f"v{i}", "A") for i in range(996)])
    assert "只计一次" in resp.get_data(as_text=True)


def test_activation_requires_a_configured_round(client):
    resp = client.post(
        "/round/deid", data={"round_serial": "0"}, follow_redirects=True
    )
    assert "尚未固定本轮计划" in resp.get_data(as_text=True)
    assert fetch(app_module.app, "SELECT active FROM deid_state")[0]["active"] == 0


def test_activation_without_serial_is_rejected(client):
    configure(client, [100, 900], ["0.5", "0.5"])
    upload(client, 1, rows_for(1, 100, 50))
    resp = client.post("/round/deid", data={}, follow_redirects=True)
    assert "轮次标识不正确" in resp.get_data(as_text=True)
    assert fetch(app_module.app, "SELECT active FROM deid_state")[0]["active"] == 0
    assert not any(is_code(value) for value in assignment_ids(app_module.app))


# ---------------------------------------------------------------------------
# Failure handling and restarts
# ---------------------------------------------------------------------------


def test_failed_switch_keeps_original_ids_and_normal_uploads(client, monkeypatch):
    configure(client, [3, 997], ["0.5", "0.5"])
    upload(client, 1, [("u1", "A"), ("u2", "A"), ("u3", "A")])

    def boom(*_args, **_kwargs):
        raise app_module.DeidError("simulated failure")

    monkeypatch.setattr(app_module, "deid_code", boom)
    resp = activate(client)
    text = resp.get_data(as_text=True)
    assert "去标识切换失败" in text
    assert "仍按原编号报送" in text
    assert assignment_ids(app_module.app) == {"u1", "u2", "u3"}
    assert fetch(app_module.app, "SELECT active FROM deid_state")[0]["active"] == 0

    # Normal reporting continues with original ids.
    upload(client, 2, [(f"v{i}", "A") for i in range(997)])
    assert "v0" in assignment_ids(app_module.app)
    assert fetch(app_module.app, "SELECT COUNT(*) c FROM assignments")[0]["c"] == 1000


def test_state_survives_restart(client, tmp_path):
    configure(client, [3, 997], ["0.5", "0.5"])
    upload(client, 1, [("u1", "A"), ("u2", "A"), ("u3", "A")])
    activate(client)
    ids = assignment_ids(app_module.app)
    db_path = app_module.app.config["DB_PATH"]

    # Simulate a fresh worker process: drop the schema cache and use a new
    # client/context against the same files.
    app_module._schema_ready_paths.discard(db_path)
    with app_module.app.test_client() as new_client:
        text = new_client.get("/").get_data(as_text=True)
        assert "已生效" in text
        resp = upload(new_client, 2, [(f"v{i}", "A") for i in range(997)])
        assert "已接收" in resp.get_data(as_text=True)
    new_ids = assignment_ids(app_module.app)
    assert len(new_ids) == 1000
    assert all(is_code(value) for value in new_ids)
    assert ids <= new_ids


def test_missing_key_blocks_uploads_but_not_inspection(client):
    configure(client, [3, 997], ["0.5", "0.5"])
    upload(client, 1, [("u1", "A"), ("u2", "A"), ("u3", "A")])
    activate(client)
    os.remove(app_module.deid_key_path())

    resp = upload(client, 2, [(f"v{i}", "A") for i in range(997)])
    text = resp.get_data(as_text=True)
    assert "密钥" in text
    assert fetch(app_module.app, "SELECT COUNT(*) c FROM uploads")[0]["c"] == 1
    # The overview still renders and says the key is a problem.
    assert "密钥不可用" in client.get("/").get_data(as_text=True)

    # Re-submitting the switch must not mint a replacement key silently.
    resp = client.post(
        "/round/deid", data={"round_serial": str(current_serial(client))},
        follow_redirects=True,
    )
    assert "密钥" in resp.get_data(as_text=True)
    assert not os.path.exists(app_module.deid_key_path())
    assert fetch(app_module.app, "SELECT COUNT(*) c FROM assignments")[0]["c"] == 3
    assert fetch(app_module.app, "SELECT active FROM deid_state")[0]["active"] == 1


# ---------------------------------------------------------------------------
# Legacy database upgrade
# ---------------------------------------------------------------------------


LEGACY_SCHEMA = """
CREATE TABLE round_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    periods INTEGER NOT NULL,
    configured_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE plan (
    period INTEGER PRIMARY KEY,
    planned_users INTEGER NOT NULL,
    prob_b TEXT NOT NULL
);
CREATE TABLE uploads (
    period INTEGER PRIMARY KEY,
    checksum TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    b_count INTEGER NOT NULL,
    accepted_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE assignments (
    user_id TEXT NOT NULL PRIMARY KEY,
    period INTEGER NOT NULL,
    grp TEXT NOT NULL
);
CREATE TABLE results (
    period INTEGER PRIMARY KEY,
    verified_users INTEGER NOT NULL,
    observed_b INTEGER NOT NULL,
    expected_b TEXT NOT NULL,
    variance TEXT NOT NULL,
    tail_low TEXT NOT NULL,
    tail_high TEXT NOT NULL,
    p_value TEXT NOT NULL,
    threshold TEXT NOT NULL,
    flagged INTEGER NOT NULL,
    verified_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
INSERT INTO round_meta (id, periods) VALUES (1, 2);
INSERT INTO plan VALUES (1, 3, '0.5');
INSERT INTO plan VALUES (2, 997, '0.5');
INSERT INTO assignments VALUES ('old1', 1, 'A');
INSERT INTO assignments VALUES ('old2', 1, 'B');
INSERT INTO assignments VALUES ('old3', 1, 'A');
"""


def test_legacy_database_upgrades_and_activates(client, tmp_path):
    legacy_path = str(tmp_path / "legacy.sqlite3")
    conn = sqlite3.connect(legacy_path)
    conn.executescript(LEGACY_SCHEMA)
    conn.commit()
    conn.close()

    app_module.bootstrap(legacy_path)
    app_module.app.config.update(DB_PATH=legacy_path, DEID_KEY_PATH=None)
    app_module._schema_ready_paths.add(legacy_path)
    try:
        columns = {
            row["name"]
            for row in fetch(app_module.app, "PRAGMA table_info(round_meta)")
        }
        assert "round_serial" in columns
        assert fetch(app_module.app, "SELECT active FROM deid_state")[0]["active"] == 0
        assert assignment_ids(app_module.app) == {"old1", "old2", "old3"}

        with app_module.app.test_client() as legacy_client:
            text = legacy_client.get("/").get_data(as_text=True)
            assert "未生效" in text
            resp = activate(legacy_client)
            assert "去标识已生效" in resp.get_data(as_text=True)
        ids = assignment_ids(app_module.app)
        assert len(ids) == 3
        assert all(is_code(value) for value in ids)
    finally:
        app_module._schema_ready_paths.discard(legacy_path)


# ---------------------------------------------------------------------------
# Concurrency: switch racing with submissions
# ---------------------------------------------------------------------------


def test_switch_racing_with_uploads_leaves_only_codes(tmp_path):
    db_path = str(tmp_path / "race.sqlite3")
    application = app_module.app
    application.config.update(
        TESTING=True, DB_PATH=db_path, SECRET_KEY="test-secret", DEID_KEY_PATH=None
    )
    app_module.bootstrap(db_path)
    try:
        with application.test_client() as setup_client:
            configure(setup_client, [100, 100, 800], ["0.5", "0.5", "0.5"])
            upload(setup_client, 1, rows_for(1, 100, 50))
            serial = current_serial(setup_client)

        payloads = {
            2: make_csv(rows_for(2, 100, 50)),
            3: make_csv(rows_for(3, 800, 400)),
        }
        barrier = threading.Barrier(3)
        errors = []

        def uploader(period, payload):
            with application.test_client() as worker:
                barrier.wait()
                resp = worker.post(
                    f"/period/{period}",
                    data={"file": (io.BytesIO(payload), f"p{period}.csv")},
                    content_type="multipart/form-data",
                    follow_redirects=True,
                )
                if resp.status_code >= 500:
                    errors.append(resp.status_code)

        def switcher():
            with application.test_client() as worker:
                barrier.wait()
                worker.post(
                    "/round/deid",
                    data={"round_serial": str(serial)},
                    follow_redirects=True,
                )

        threads = [
            threading.Thread(target=uploader, args=(2, payloads[2])),
            threading.Thread(target=uploader, args=(3, payloads[3])),
            threading.Thread(target=switcher),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert fetch(application, "SELECT active FROM deid_state")[0]["active"] == 1
        ids = assignment_ids(application)
        assert all(is_code(value) for value in ids)
        # All three accepted periods are present; some upload may legitimately
        # have been retried after the switch, but none may be lost.
        assert fetch(application, "SELECT COUNT(*) c FROM uploads")[0]["c"] == 3
        assert len(ids) == 1000
    finally:
        app_module._schema_ready_paths.discard(db_path)


def test_concurrent_switch_requests_are_idempotent(tmp_path):
    db_path = str(tmp_path / "switch-race.sqlite3")
    application = app_module.app
    application.config.update(
        TESTING=True, DB_PATH=db_path, SECRET_KEY="test-secret", DEID_KEY_PATH=None
    )
    app_module.bootstrap(db_path)
    try:
        with application.test_client() as setup_client:
            configure(setup_client, [50, 950], ["0.5", "0.5"])
            upload(setup_client, 1, rows_for(1, 50, 25))
            serial = current_serial(setup_client)

        barrier = threading.Barrier(3)
        bodies = []

        def worker():
            with application.test_client() as worker_client:
                barrier.wait()
                resp = worker_client.post(
                    "/round/deid",
                    data={"round_serial": str(serial)},
                    follow_redirects=True,
                )
                bodies.append(resp.get_data(as_text=True))

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Exactly one request performed the switch; the others report the
        # current state. None may fail or mint a second key.
        activated = [b for b in bodies if "去标识已生效：" in b]
        reported = [b for b in bodies if "未做任何改动" in b]
        assert len(activated) == 1, bodies
        assert len(reported) == 2, bodies
        assert fetch(application, "SELECT COUNT(*) c FROM assignments")[0]["c"] == 50
        assert all(is_code(value) for value in assignment_ids(application))
    finally:
        app_module._schema_ready_paths.discard(db_path)


def test_period_page_explains_that_csv_still_uses_original_ids(client):
    configure(client, [100, 900], ["0.5", "0.5"])
    assert "本轮尚未启用去标识" in client.get("/period/1").get_data(as_text=True)
    activate(client)
    text = client.get("/period/1").get_data(as_text=True)
    assert "入库时自动换成本轮固定代号" in text


def test_reupload_of_period_uploaded_after_switch_is_duplicate(client):
    configure(client, [100, 900], ["0.5", "0.5"])
    activate(client)
    rows = rows_for(1, 100, 50)
    first = upload(client, 1, rows)
    assert "第 1 期已接收" in first.get_data(as_text=True)
    second = upload(client, 1, rows)
    assert "相同明细已接收" in second.get_data(as_text=True)
    assert fetch(app_module.app, "SELECT COUNT(*) c FROM uploads")[0]["c"] == 1
    assert fetch(app_module.app, "SELECT COUNT(*) c FROM assignments")[0]["c"] == 100
