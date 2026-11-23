"""Flask + SQLite service for phased A/B split verification.

The workflow is deliberately small:

* a round is configured once with 1..8 periods, each with a planned
  headcount and an independent group-B probability,
* one CSV per period (user id + actual group) is uploaded, in any order,
* complete consecutive prefixes are verified against the cumulative
  binomial distribution implied by the plan,
* every verification records the observed cumulative B count, an exact
  two-sided p-value and whether it crosses the Bonferroni threshold,
* at any point (even before every list is in) the round can be switched to
  de-identified storage: original ids in ``assignments`` are replaced by
  irreversible, round-scoped HMAC codes, and later submissions are stored as
  codes on arrival.  Verification numbers do not change.

Run with ``gunicorn`` (see ``Dockerfile`` / ``compose.yaml``) or with
``flask --app app run`` for development.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import os
import random
import secrets
import sqlite3
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from functools import lru_cache
from typing import Optional, Sequence

from flask import (
    Flask,
    abort,
    current_app,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

import stats

# ---------------------------------------------------------------------------
# Round constants
# ---------------------------------------------------------------------------

MAX_PERIODS = 8
TOTAL_USERS = 1000
MAX_CSV_BYTES = 2 * 1024 * 1024
MAX_USER_ID_LENGTH = 128
ALPHA = Decimal("0.05")

DEFAULT_DB = os.environ.get(
    "DIV_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "split.sqlite3"),
)

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-secret-change-me"),
    MAX_CONTENT_LENGTH=MAX_CSV_BYTES + 64 * 1024,
    DB_PATH=DEFAULT_DB,
    # Optional explicit override.  By default the key lives next to the
    # database (``<db dir>/deid.key``) so the compose named volume keeps it
    # across restarts.  It is deliberately *outside* SQLite: a copy of the
    # database alone cannot be used to reverse codes back to user ids.
    DEID_KEY_PATH=os.environ.get("DEID_KEY_FILE"),
)
app.jinja_env.globals.update(
    MAX_PERIODS=MAX_PERIODS,
    TOTAL_USERS=TOTAL_USERS,
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS round_meta (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    periods      INTEGER NOT NULL CHECK (periods BETWEEN 1 AND 8),
    round_serial INTEGER NOT NULL DEFAULT 1,
    configured_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS plan (
    period        INTEGER PRIMARY KEY CHECK (period BETWEEN 1 AND 8),
    planned_users INTEGER NOT NULL CHECK (planned_users > 0),
    prob_b        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS uploads (
    period        INTEGER PRIMARY KEY CHECK (period BETWEEN 1 AND 8),
    checksum      TEXT    NOT NULL,
    row_count     INTEGER NOT NULL,
    b_count       INTEGER NOT NULL,
    accepted_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS assignments (
    user_id       TEXT    NOT NULL PRIMARY KEY,
    period        INTEGER NOT NULL CHECK (period BETWEEN 1 AND 8),
    grp           TEXT    NOT NULL CHECK (grp IN ('A','B'))
);

CREATE INDEX IF NOT EXISTS assignments_period_idx ON assignments(period);

CREATE TABLE IF NOT EXISTS results (
    period         INTEGER PRIMARY KEY CHECK (period BETWEEN 1 AND 8),
    verified_users INTEGER NOT NULL,
    observed_b     INTEGER NOT NULL,
    expected_b     TEXT    NOT NULL,
    variance       TEXT    NOT NULL,
    tail_low       TEXT    NOT NULL,
    tail_high      TEXT    NOT NULL,
    p_value        TEXT    NOT NULL,
    threshold      TEXT    NOT NULL,
    flagged        INTEGER NOT NULL CHECK (flagged IN (0, 1)),
    verified_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Whether original user ids have been replaced by irreversible round codes.
-- The activated round serial is stored so a switch started for an old round
-- can never mark a later (reset) round as de-identified.
CREATE TABLE IF NOT EXISTS deid_state (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    active         INTEGER NOT NULL CHECK (active IN (0, 1)),
    round_serial   INTEGER NOT NULL DEFAULT 0,
    activated_at   TEXT,
    activated_users INTEGER NOT NULL DEFAULT 0,
    key_check      TEXT
);
"""


def connect(path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(
        path or current_app.config["DB_PATH"],
        timeout=30,
        isolation_level=None,  # explicit BEGIN IMMEDIATE transactions
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        # A read-only filesystem or another writer; the connection still works.
        pass
    return conn


_schema_ready_paths: set[str] = set()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    migrate_schema(conn)


def migrate_schema(conn: sqlite3.Connection) -> None:
    """Upgrade databases created by earlier versions in place.

    ``CREATE TABLE IF NOT EXISTS`` leaves an existing ``round_meta`` alone, so
    columns added later must be filled in explicitly.  This is idempotent and
    cheap, which matters because every worker calls it once.
    """
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(round_meta)")
    }
    if "round_serial" not in columns:
        try:
            conn.execute(
                "ALTER TABLE round_meta ADD COLUMN round_serial INTEGER NOT NULL DEFAULT 1"
            )
        except sqlite3.OperationalError:
            # A concurrent worker added it first.
            pass
    deid_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(deid_state)")
    }
    if "key_check" not in deid_columns:
        try:
            conn.execute("ALTER TABLE deid_state ADD COLUMN key_check TEXT")
        except sqlite3.OperationalError:
            pass
    conn.execute(
        "INSERT OR IGNORE INTO deid_state (id, active, round_serial, activated_users) "
        "VALUES (1, 0, 0, 0)"
    )


def get_db() -> sqlite3.Connection:
    """Per-request connection, with lazy idempotent schema creation."""
    if "db" not in g:
        db = connect()
        path = current_app.config["DB_PATH"]
        if path not in _schema_ready_paths:
            try:
                init_schema(db)
            except sqlite3.OperationalError:
                # Another worker may be creating the schema concurrently.
                time.sleep(0.1)
                init_schema(db)
            _schema_ready_paths.add(path)
        g.db = db
    return g.db


@app.teardown_appcontext
def close_db(exc: Optional[BaseException]) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def bootstrap(path: Optional[str] = None) -> None:
    """Create the database file and schema (used by CLI/entrypoint/tests)."""
    target = path or app.config["DB_PATH"]
    os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
    conn = connect(target)
    try:
        init_schema(conn)
    finally:
        conn.close()
    _schema_ready_paths.add(target)


@app.cli.command("init-db")
def init_db_command() -> None:
    bootstrap()
    print(f"initialised {app.config['DB_PATH']}")


# ---------------------------------------------------------------------------
# Small data-access helpers
# ---------------------------------------------------------------------------


def round_periods(db: sqlite3.Connection) -> Optional[int]:
    row = db.execute("SELECT periods FROM round_meta WHERE id = 1").fetchone()
    return int(row["periods"]) if row else None


def round_serial(db: sqlite3.Connection) -> int:
    """Monotonic identifier of the current round.

    Reset bumps it, so a de-identification switch that was prepared for the
    previous round can be recognised as stale and refused.
    """
    row = db.execute("SELECT round_serial FROM round_meta WHERE id = 1").fetchone()
    return int(row["round_serial"]) if row else 0


def deid_status(db: sqlite3.Connection) -> dict:
    """Current de-identification state, scoped to the live round."""
    row = db.execute(
        "SELECT active, round_serial, activated_at, activated_users, key_check "
        "FROM deid_state WHERE id = 1"
    ).fetchone()
    serial = round_serial(db)
    active = bool(row and row["active"] and int(row["round_serial"]) == serial)
    status = {
        "active": active,
        "round_serial": serial,
        "activated_at": row["activated_at"] if active else None,
        "activated_users": int(row["activated_users"]) if active else 0,
        "key_check": row["key_check"] if active else None,
        "key_ok": True,
    }
    if active:
        try:
            key = load_deid_key(create=False)
        except (OSError, DeidError):
            key = None
        status["key_ok"] = bool(key) and deid_key_check(key) == row["key_check"]
    return status


# ---------------------------------------------------------------------------
# De-identification helpers
# ---------------------------------------------------------------------------

DEID_CODE_PREFIX = "deid-"
DEID_CODE_HEX = 32  # 128 bits of HMAC-SHA256 output
DEID_KEY_BYTES = 32
_KEY_CHECK_CONTEXT = b"split-verify-deid-key-check\x00"


class DeidError(RuntimeError):
    """A failure that must abort a de-identification switch."""


def deid_key_path() -> str:
    configured = current_app.config.get("DEID_KEY_PATH")
    if configured:
        return configured
    db_dir = os.path.dirname(os.path.abspath(current_app.config["DB_PATH"]))
    return os.path.join(db_dir, "deid.key")


def load_deid_key(create: bool = False) -> Optional[bytes]:
    """Read the HMAC key, optionally creating it atomically with mode 0600.

    The key lives beside the database rather than inside it.  A leaked copy of
    the SQLite file therefore cannot be used to turn codes back into user ids,
    while the compose named volume keeps both across restarts.
    """
    path = deid_key_path()
    try:
        with open(path, "rb") as handle:
            key = handle.read().strip()
        if not key:
            raise DeidError("去标识密钥文件为空")
        return key
    except FileNotFoundError:
        if not create:
            return None

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    key = secrets.token_hex(DEID_KEY_BYTES).encode("ascii")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # Another worker won the race; use the key it wrote.
        return load_deid_key(create=False)
    try:
        os.write(fd, key + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    return key


def deid_key_check(key: bytes) -> str:
    return hashlib.sha256(_KEY_CHECK_CONTEXT + key).hexdigest()


def deid_code(key: bytes, serial: int, user_id: str) -> str:
    """Deterministic, round-scoped pseudonym for ``user_id``.

    * same round + same user id  -> same code (dedupe keeps working),
    * different round            -> different code (``serial`` is in the MAC),
    * without ``key``            -> cannot be re-derived from the code.
    """
    message = f"{serial}\x00{user_id}".encode("utf-8")
    digest = hmac.new(key, message, hashlib.sha256).hexdigest()[:DEID_CODE_HEX]
    return f"{DEID_CODE_PREFIX}{digest}"


def upload_checksum(
    rows: Sequence[tuple[str, str]], key: Optional[bytes], serial: int
) -> str:
    """Content fingerprint used for re-upload detection.

    Without de-identification this is the historical plain SHA-256 over the
    original ``user_id + group`` lines, so existing databases stay compatible.
    Once de-identification is active the same fingerprint is keyed: otherwise
    a small user-id space could be brute-forced from the checksum alone, which
    would defeat the point of storing codes.
    """
    canonical = "\n".join(
        f"{user_id}\t{group}"
        for user_id, group in sorted(rows, key=lambda r: (r[0], r[1]))
    ).encode("utf-8")
    if key is None:
        return hashlib.sha256(canonical).hexdigest()
    message = f"{serial}\x00".encode("utf-8") + canonical
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def stored_user_ids(
    db: sqlite3.Connection, parsed: ParsedUpload
) -> tuple[list[str], Optional[str], Optional[bytes], int]:
    """Map parsed original ids to what should be stored, or explain why not.

    Returns ``(stored_ids, error, key, serial)``.  ``key`` is ``None`` while
    de-identification is off; when it is on, callers must use the key (and the
    round ``serial``) for the stored content fingerprint as well.
    """
    status = deid_status(db)
    if not status["active"]:
        return [row.user_id for row in parsed.rows], None, None, status["round_serial"]
    try:
        key = load_deid_key(create=False)
    except (OSError, DeidError) as exc:
        return (
            [],
            f"去标识已生效但密钥不可用（{exc}），拒绝以原编号入库；请恢复密钥或重置本轮。",
            None,
            status["round_serial"],
        )
    if key is None:
        return (
            [],
            "去标识已生效但密钥文件缺失，拒绝以原编号入库；请恢复密钥或重置本轮。",
            None,
            status["round_serial"],
        )
    if deid_key_check(key) != status["key_check"]:
        return (
            [],
            "去标识已生效但密钥与本轮不匹配，拒绝以原编号入库；请恢复密钥或重置本轮。",
            None,
            status["round_serial"],
        )
    codes = [
        deid_code(key, status["round_serial"], row.user_id) for row in parsed.rows
    ]
    return codes, None, key, status["round_serial"]


@dataclass
class DeidOutcome:
    status: str  # "activated" | "active" | "error"
    message: str
    converted: int = 0


def enable_deidentification(expected_serial: Optional[int] = None) -> DeidOutcome:
    """Replace stored user ids with irreversible round codes in one transaction.

    Concurrent uploads serialise on ``BEGIN IMMEDIATE``: an upload either
    commits before the switch (and gets converted) or after it (and is stored
    as codes straight away).  Any failure rolls everything back, so the round
    keeps accepting submissions with original ids exactly as before.

    The key is only created *after* confirming the round is not already
    active.  Otherwise a missing key would be silently replaced by a fresh one
    that cannot reproduce the codes already in the database.
    """
    db = get_db()
    try:
        existing_key = load_deid_key(create=False)
    except (OSError, DeidError) as exc:
        return DeidOutcome(
            "error", f"去标识切换失败：无法读取密钥（{exc}）。本轮仍按原编号报送。"
        )
    existing_check = deid_key_check(existing_key) if existing_key else None

    last_error: Optional[BaseException] = None
    for attempt in range(6):
        try:
            db.execute("BEGIN IMMEDIATE")
            if round_periods(db) is None:
                _rollback(db)
                return DeidOutcome("error", "尚未固定本轮计划，无法启用去标识。")

            serial = round_serial(db)
            if expected_serial is not None and int(expected_serial) != serial:
                _rollback(db)
                return DeidOutcome(
                    "error",
                    "本轮已更换（可能已重置），旧的去标识请求不再生效，请刷新页面后重试。",
                )

            state = db.execute(
                "SELECT active, round_serial, activated_users, key_check "
                "FROM deid_state WHERE id = 1"
            ).fetchone()
            if (
                state is not None
                and int(state["active"])
                and int(state["round_serial"]) == serial
            ):
                _rollback(db)
                count = int(state["activated_users"])
                # Another worker may have created the key after our first read;
                # re-read before calling it a mismatch.
                check = existing_check
                if check is None:
                    try:
                        retry_key = load_deid_key(create=False)
                    except (OSError, DeidError):
                        retry_key = None
                    if retry_key is not None:
                        check = deid_key_check(retry_key)
                if check != state["key_check"]:
                    return DeidOutcome(
                        "error",
                        "本轮去标识已生效，但当前密钥缺失或与生效时不一致；未做任何改动。"
                        "请恢复原密钥（deid.key）或重置本轮后重新启用。",
                        count,
                    )
                return DeidOutcome(
                    "active",
                    f"本轮去标识已生效（{count} 条记录使用代号），未做任何改动。",
                    count,
                )

            if existing_key is None:
                try:
                    key = load_deid_key(create=True)
                except (OSError, DeidError) as exc:
                    _rollback(db)
                    return DeidOutcome(
                        "error",
                        f"去标识切换失败：无法准备密钥（{exc}）。本轮仍按原编号报送。",
                    )
            else:
                key = existing_key
            assert key is not None
            key_check = deid_key_check(key)

            rows = db.execute(
                "SELECT user_id, period, grp FROM assignments"
            ).fetchall()
            code_for_user: dict[str, str] = {}
            user_for_code: dict[str, str] = {}
            replacements: list[tuple[str, int, str]] = []
            per_period: dict[int, list[tuple[str, str]]] = {}
            for row in rows:
                original = str(row["user_id"])
                code = code_for_user.get(original)
                if code is None:
                    code = deid_code(key, serial, original)
                    previous = user_for_code.get(code)
                    if previous is not None and previous != original:
                        raise DeidError("代号冲突")
                    code_for_user[original] = code
                    user_for_code[code] = original
                period = int(row["period"])
                grp = str(row["grp"])
                replacements.append((code, period, grp))
                per_period.setdefault(period, []).append((original, grp))

            # Replace in one shot rather than updating primary keys in place;
            # rollback restores the original ids if anything fails.
            db.execute("DELETE FROM assignments")
            db.executemany(
                "INSERT INTO assignments (user_id, period, grp) VALUES (?, ?, ?)",
                replacements,
            )
            # The upload fingerprints were plain hashes of the original ids;
            # key them now so the checksums cannot be brute-forced either.
            # Same upload content + same key still yields "duplicate" on a
            # re-upload of the original CSV.
            for period, pairs in per_period.items():
                checksum = upload_checksum(pairs, key, serial)
                db.execute(
                    "UPDATE uploads SET checksum = ? WHERE period = ?",
                    (checksum, period),
                )
            db.execute(
                "INSERT INTO deid_state "
                "(id, active, round_serial, activated_at, activated_users, key_check) "
                "VALUES (1, 1, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'), ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "active = 1, round_serial = excluded.round_serial, "
                "activated_at = excluded.activated_at, "
                "activated_users = excluded.activated_users, "
                "key_check = excluded.key_check",
                (serial, len(replacements), key_check),
            )
            db.execute("COMMIT")
            return DeidOutcome(
                "activated",
                f"本轮去标识已生效：{len(replacements)} 条已收明细改用不可逆代号，"
                "后续报送也只保存代号；核验结果不受影响。",
                len(replacements),
            )
        except sqlite3.IntegrityError as exc:
            _rollback(db)
            last_error = exc
            if attempt < 2:
                continue
            return DeidOutcome(
                "error", "去标识切换失败（代号冲突），本轮仍按原编号报送。"
            )
        except sqlite3.OperationalError as exc:
            _rollback(db)
            last_error = exc
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                time.sleep(0.05 * (attempt + 1) + random.random() * 0.02)
                continue
            return DeidOutcome(
                "error", f"去标识切换失败（{exc}），本轮仍按原编号报送。"
            )
        except DeidError as exc:
            _rollback(db)
            return DeidOutcome(
                "error", f"去标识切换失败（{exc}），本轮仍按原编号报送。"
            )
        except Exception:
            _rollback(db)
            raise

    return DeidOutcome(
        "error", f"数据库繁忙，去标识切换未执行，本轮仍按原编号报送（{last_error}）。"
    )


def load_plan(db: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(db.execute("SELECT period, planned_users, prob_b FROM plan ORDER BY period"))


def plan_key(plan: Sequence[sqlite3.Row]) -> tuple[tuple[int, str], ...]:
    return tuple((int(row["planned_users"]), str(row["prob_b"])) for row in plan)


@lru_cache(maxsize=16)
def cached_distributions(key: tuple[tuple[int, str], ...]):
    return stats.all_prefix_distributions(
        [(n, Decimal(p)) for n, p in key]
    )


def plan_distributions(plan: Sequence[sqlite3.Row]):
    return cached_distributions(plan_key(plan))


def verification_threshold(periods: int) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = stats.PRECISION
        return ALPHA / Decimal(periods)


def plan_is_locked(db: sqlite3.Connection) -> bool:
    row = db.execute(
        "SELECT (SELECT COUNT(*) FROM uploads) + (SELECT COUNT(*) FROM results) AS n"
    ).fetchone()
    return bool(row and row["n"])


def round_state(db: sqlite3.Connection) -> dict:
    """Everything the dashboard needs, with results joined to the plan."""
    periods = round_periods(db)
    if periods is None:
        return {
            "configured": False,
            "periods": 0,
            "plan": [],
            "uploads": {},
            "results": {},
            "pending": [],
            "threshold": None,
            "earliest_flag": None,
            "locked": False,
            "deid": deid_status(db),
            "max_periods": MAX_PERIODS,
            "total_users": TOTAL_USERS,
        }

    plan = load_plan(db)
    uploads = {
        int(row["period"]): row
        for row in db.execute("SELECT period, row_count, b_count, accepted_at, checksum FROM uploads")
    }
    results = {
        int(row["period"]): row
        for row in db.execute(
            "SELECT period, verified_users, observed_b, expected_b, variance, "
            "tail_low, tail_high, p_value, threshold, flagged, verified_at "
            "FROM results ORDER BY period"
        )
    }
    verified_periods = sorted(results)
    pending = [
        period
        for period in sorted(uploads)
        if period not in results
    ]
    earliest_flag = next(
        (results[p]["period"] for p in verified_periods if results[p]["flagged"]), None
    )
    return {
        "configured": True,
        "periods": periods,
        "plan": plan,
        "uploads": uploads,
        "results": results,
        "pending": pending,
        "threshold": verification_threshold(periods),
        "earliest_flag": earliest_flag,
        "locked": plan_is_locked(db),
        "deid": deid_status(db),
        "max_periods": MAX_PERIODS,
        "total_users": TOTAL_USERS,
    }


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

USER_HEADERS = {"user_id", "userid", "user", "用户编号", "用户id", "用户标识", "成员id"}
GROUP_HEADERS = {"group", "grp", "ab_group", "ab组", "组别", "分组", "实际组别"}


class CSVError(ValueError):
    """A user-facing CSV validation error."""


@dataclass(frozen=True)
class UploadRow:
    user_id: str
    group: str


@dataclass
class ParsedUpload:
    rows: tuple[UploadRow, ...]
    b_count: int


def _normalise_header(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace("-", "_")


def parse_csv(raw: bytes) -> ParsedUpload:
    if not raw:
        raise CSVError("CSV 文件为空")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CSVError("CSV 必须是 UTF-8 编码") from exc
    if "\x00" in text:
        raise CSVError("CSV 含有二进制空字符")

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise CSVError("CSV 文件为空") from exc

    normalised = [_normalise_header(cell) for cell in header]
    user_index = next(
        (i for i, name in enumerate(normalised) if name in USER_HEADERS), None
    )
    group_index = next(
        (i for i, name in enumerate(normalised) if name in GROUP_HEADERS), None
    )
    if user_index is None or group_index is None:
        raise CSVError("表头必须同时包含用户编号列（user_id）和组别列（group）")
    if user_index == group_index:
        raise CSVError("用户编号列与组别列不能相同")

    rows: list[UploadRow] = []
    seen: set[str] = set()
    for line_no, record in enumerate(reader, start=2):
        if not any(cell.strip() for cell in record):
            continue
        if len(record) <= max(user_index, group_index):
            raise CSVError(f"第 {line_no} 行字段不足")
        user_id = record[user_index].strip()
        group = record[group_index].strip().upper()
        if not user_id:
            raise CSVError(f"第 {line_no} 行用户编号为空")
        if len(user_id) > MAX_USER_ID_LENGTH:
            raise CSVError(f"第 {line_no} 行用户编号超过 {MAX_USER_ID_LENGTH} 字符")
        if group not in ("A", "B"):
            raise CSVError(f"第 {line_no} 行组别必须为 A 或 B（实际为 {record[group_index]!r}）")
        if user_id in seen:
            raise CSVError(f"用户编号 {user_id!r} 在文件内重复")
        seen.add(user_id)
        rows.append(UploadRow(user_id=user_id, group=group))
        if len(rows) > TOTAL_USERS:
            raise CSVError(f"单次上传不能超过 {TOTAL_USERS} 行")

    if not rows:
        raise CSVError("CSV 没有数据行")

    return ParsedUpload(
        rows=tuple(rows),
        b_count=sum(1 for row in rows if row.group == "B"),
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def validate_upload_against_plan(
    parsed: ParsedUpload, planned_users: int
) -> None:
    if len(parsed.rows) != planned_users:
        raise CSVError(
            f"人数不符：该期计划 {planned_users} 人，CSV 实际 {len(parsed.rows)} 人"
        )


def recompute_results(db: sqlite3.Connection) -> list[int]:
    """Rebuild the verified contiguous prefix and return its periods.

    Must be called inside the caller's write transaction.  Results are
    deleted and re-inserted so a period never has more than one row.
    """
    plan = load_plan(db)
    periods = round_periods(db) or len(plan)
    if not plan:
        db.execute("DELETE FROM results")
        return []

    distributions = plan_distributions(plan)
    threshold = verification_threshold(periods)

    db.execute("DELETE FROM results")
    cum_users = 0
    cum_b = 0
    verified: list[int] = []
    for index, row in enumerate(plan):
        upload = db.execute(
            "SELECT row_count, b_count FROM uploads WHERE period = ?",
            (row["period"],),
        ).fetchone()
        if upload is None:
            break

        cum_users += int(upload["row_count"])
        cum_b += int(upload["b_count"])
        distribution = distributions[index]
        p_value = stats.two_sided_p_value(distribution, cum_b)
        tail_low, tail_high = stats.tail_probabilities(distribution, cum_b)
        expected_b = sum(
            Decimal(str(plan[i]["planned_users"])) * Decimal(str(plan[i]["prob_b"]))
            for i in range(index + 1)
        )
        variance = sum(
            Decimal(str(plan[i]["planned_users"]))
            * Decimal(str(plan[i]["prob_b"]))
            * (Decimal(1) - Decimal(str(plan[i]["prob_b"])))
            for i in range(index + 1)
        )
        flagged = 1 if p_value <= threshold else 0
        db.execute(
            "INSERT INTO results (period, verified_users, observed_b, expected_b, "
            "variance, tail_low, tail_high, p_value, threshold, flagged) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(row["period"]),
                cum_users,
                cum_b,
                str(expected_b),
                str(variance),
                str(tail_low),
                str(tail_high),
                str(p_value),
                str(threshold),
                flagged,
            ),
        )
        verified.append(int(row["period"]))
    return verified


@dataclass
class UploadOutcome:
    status: str  # "accepted" | "duplicate" | "conflict" | "error"
    message: str


def _rollback(db: sqlite3.Connection) -> None:
    try:
        db.execute("ROLLBACK")
    except sqlite3.OperationalError:
        pass


def store_upload(period: int, parsed: ParsedUpload) -> UploadOutcome:
    """Validate and store an upload atomically.

    Writers serialise on ``BEGIN IMMEDIATE``.  A repeated identical upload is
    acknowledged without inserting a second row/result; different content
    for a period that is already stored is rejected.  Any failure rolls back,
    leaving previously stored details and verification results untouched.
    """
    last_error: Optional[BaseException] = None
    for attempt in range(6):
        db = get_db()
        try:
            db.execute("BEGIN IMMEDIATE")
            plan_row = db.execute(
                "SELECT planned_users, prob_b FROM plan WHERE period = ?", (period,)
            ).fetchone()
            if plan_row is None:
                _rollback(db)
                return UploadOutcome("error", f"第 {period} 期不在计划中")

            expected_rows = int(plan_row["planned_users"])
            if len(parsed.rows) != expected_rows:
                _rollback(db)
                return UploadOutcome(
                    "error",
                    f"人数不符：该期计划 {expected_rows} 人，CSV 实际 {len(parsed.rows)} 人",
                )

            # Map to what will actually be stored.  While de-identification is
            # off this is the original ids and an unkeyed fingerprint; once it
            # is on the ids are round-scoped codes and the fingerprint is
            # keyed, so neither can be inverted from the database alone.
            stored_ids, deid_error, deid_key, serial = stored_user_ids(db, parsed)
            if deid_error is not None:
                _rollback(db)
                return UploadOutcome("error", deid_error)
            checksum = upload_checksum(
                [(row.user_id, row.group) for row in parsed.rows], deid_key, serial
            )

            existing = db.execute(
                "SELECT checksum, row_count, b_count FROM uploads WHERE period = ?",
                (period,),
            ).fetchone()
            if existing is not None:
                _rollback(db)
                if existing["checksum"] == checksum:
                    return UploadOutcome(
                        "duplicate",
                        f"第 {period} 期相同明细已接收，本次未重复计入。",
                    )
                return UploadOutcome(
                    "conflict",
                    f"第 {period} 期已接收过不同内容的明细，拒收本次上传。",
                )

            # Every user counts once for the whole round.  The PRIMARY KEY on
            # assignments is the final guard; this pre-check gives a precise
            # message for sequential uploads.
            other_users = {
                row["user_id"]
                for row in db.execute(
                    "SELECT user_id FROM assignments WHERE period <> ?", (period,)
                )
            }
            conflicts = [uid for uid in stored_ids if uid in other_users]
            if conflicts:
                _rollback(db)
                # Never echo a stored code as if it were a user id: report the
                # original ids the data team actually sent.
                original = [
                    row.user_id
                    for row, uid in zip(parsed.rows, stored_ids)
                    if uid in other_users
                ]
                sample = ", ".join(sorted(original)[:5])
                more = " 等" if len(original) > 5 else ""
                return UploadOutcome(
                    "error",
                    f"拒收：{len(original)} 个用户编号已出现在其他期（{sample}{more}），每人全程只计一次。",
                )

            db.execute(
                "INSERT INTO uploads (period, checksum, row_count, b_count) "
                "VALUES (?, ?, ?, ?)",
                (period, checksum, len(parsed.rows), parsed.b_count),
            )
            db.executemany(
                "INSERT INTO assignments (user_id, period, grp) VALUES (?, ?, ?)",
                zip(stored_ids, [period] * len(stored_ids), [row.group for row in parsed.rows]),
            )
            verified = recompute_results(db)
            db.execute("COMMIT")
            suffix = (
                f"连续核验到第 {verified[-1]} 期。"
                if verified
                else "尚无连续到齐的期次，暂未核验。"
            )
            return UploadOutcome(
                "accepted",
                f"第 {period} 期已接收（{len(parsed.rows)} 人，B 组 {parsed.b_count} 人）。{suffix}",
            )
        except sqlite3.IntegrityError as exc:
            _rollback(db)
            last_error = exc
            # Most likely a concurrent insert of one of the same users, or the
            # same period; re-check once if this was a lock retry.
            if attempt < 2:
                continue
            return UploadOutcome(
                "error",
                "拒收：用户编号与已有期次冲突（每人全程只计一次），或该期已被并发上传占用。",
            )
        except sqlite3.OperationalError as exc:
            _rollback(db)
            last_error = exc
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                time.sleep(0.05 * (attempt + 1) + random.random() * 0.02)
                continue
            raise
        except Exception:
            # Any non-SQL failure must not leave a half-written period behind.
            _rollback(db)
            raise

    return UploadOutcome(
        "error",
        f"数据库繁忙，请稍后重试（{last_error}）。",
    )


# ---------------------------------------------------------------------------
# Template helpers / routes
# ---------------------------------------------------------------------------


@app.template_filter("prob")
def prob_filter(value) -> str:
    return stats.format_probability(Decimal(str(value)))


@app.template_filter("expected")
def expected_filter(value) -> str:
    return stats.format_expected(Decimal(str(value)))


@app.template_filter("sci")
def sci_filter(value) -> str:
    return stats.format_probability(Decimal(str(value)), digits=4)


@app.get("/healthz")
def healthz():
    db = get_db()
    db.execute("SELECT 1")
    return jsonify({"ok": True})


@app.get("/")
def index():
    db = get_db()
    state = round_state(db)
    latest = None
    latest_stats = None
    if state["configured"] and state["results"]:
        verified = sorted(state["results"])
        latest_period = verified[-1]
        result = state["results"][latest_period]
        plan = state["plan"]
        distributions = plan_distributions(plan)
        distribution = distributions[latest_period - 1]
        latest_stats = stats.distribution_stats(
            distribution,
            Decimal(result["expected_b"]),
            Decimal(result["variance"]),
        )
        latest = result
    return render_template(
        "index.html",
        state=state,
        latest=latest,
        latest_stats=latest_stats,
    )


@app.route("/plan", methods=["GET", "POST"])
def plan():
    db = get_db()
    periods = round_periods(db)
    locked = plan_is_locked(db)
    existing = load_plan(db)

    if request.method == "POST":
        if locked:
            flash("本轮已有上传或核验结果，计划已锁定；如需换比例请先重置本轮数据。", "error")
            return redirect(url_for("plan"))
        try:
            try:
                periods_value = int(request.form.get("periods", ""))
            except ValueError as exc:
                raise ValueError("计划期数必须是整数") from exc
            if not 1 <= periods_value <= MAX_PERIODS:
                raise ValueError(f"计划期数必须为 1 到 {MAX_PERIODS}")
            counts: list[int] = []
            probabilities: list[Decimal] = []
            for period_no in range(1, periods_value + 1):
                count_text = request.form.get(f"count_{period_no}", "").strip()
                prob_text = request.form.get(f"prob_{period_no}", "").strip()
                if not count_text or not prob_text:
                    raise ValueError(f"第 {period_no} 期的人数或 B 组概率未填写")
                try:
                    count = int(count_text)
                except ValueError as exc:
                    raise ValueError(f"第 {period_no} 期人数必须是整数") from exc
                if count < 1:
                    raise ValueError(f"第 {period_no} 期人数必须为正整数")
                if count > TOTAL_USERS:
                    raise ValueError(f"第 {period_no} 期人数不能超过 {TOTAL_USERS}")
                try:
                    probability = Decimal(prob_text)
                except InvalidOperation as exc:
                    raise ValueError(f"第 {period_no} 期概率格式不正确") from exc
                if not probability.is_finite() or not (Decimal(0) < probability < Decimal(1)):
                    raise ValueError(f"第 {period_no} 期概率必须严格介于 0 与 1 之间")
                if abs(probability.as_tuple().exponent) > 12:
                    raise ValueError(f"第 {period_no} 期概率最多 12 位小数")
                counts.append(count)
                probabilities.append(probability)
            if sum(counts) != TOTAL_USERS:
                raise ValueError(
                    f"各期应到人数合计必须为 {TOTAL_USERS}，当前为 {sum(counts)}"
                )
            canonical = [stats.canonical_probability(p) for p in probabilities]
        except (ValueError, InvalidOperation) as exc:
            flash(str(exc), "error")
            return (
                render_template(
                    "plan.html",
                    periods=periods or 4,
                    existing=existing,
                    locked=locked,
                    form=request.form,
                    max_periods=MAX_PERIODS,
                    total_users=TOTAL_USERS,
                ),
                400,
            )

        try:
            db.execute("BEGIN IMMEDIATE")
            if plan_is_locked(db):
                _rollback(db)
                flash("计划已被其他操作锁定，请刷新后重试。", "error")
                return redirect(url_for("plan"))
            db.execute("DELETE FROM plan")
            db.executemany(
                "INSERT INTO plan (period, planned_users, prob_b) VALUES (?, ?, ?)",
                [
                    (index + 1, count, probability)
                    for index, (count, probability) in enumerate(zip(counts, canonical))
                ],
            )
            db.execute(
                "INSERT INTO round_meta (id, periods) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET periods = excluded.periods, "
                "configured_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')",
                (periods_value,),
            )
            db.execute("COMMIT")
        except Exception:
            _rollback(db)
            raise
        flash(f"已固定 {periods_value} 期计划，合计 {TOTAL_USERS} 人。", "success")
        return redirect(url_for("index"))

    return render_template(
        "plan.html",
        periods=periods or 4,
        existing=existing,
        locked=locked,
        form=None,
        max_periods=MAX_PERIODS,
        total_users=TOTAL_USERS,
    )


@app.post("/round/reset")
def reset_round():
    db = get_db()
    if request.form.get("confirm", "").strip() != "RESET":
        flash("重置确认口令不正确（请输入 RESET）。", "error")
        return redirect(url_for("plan"))
    try:
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM results")
        db.execute("DELETE FROM assignments")
        db.execute("DELETE FROM uploads")
        # A new round gets a fresh serial and starts without de-identification.
        # A switch request prepared for the old round can no longer apply,
        # because its serial no longer matches.
        db.execute(
            "UPDATE round_meta SET round_serial = round_serial + 1 WHERE id = 1"
        )
        db.execute(
            "UPDATE deid_state SET active = 0, round_serial = 0, activated_at = NULL, "
            "activated_users = 0, key_check = NULL WHERE id = 1"
        )
        db.execute("COMMIT")
    except Exception:
        _rollback(db)
        raise
    flash("已清空本轮上传明细与核验结果，计划保留且可重新调整；新一轮需重新启用去标识。", "success")
    return redirect(url_for("plan"))


@app.post("/round/deid")
def enable_deid():
    """Switch the current round to code-only storage.

    Safe to call repeatedly: if the current round is already de-identified
    the response is a plain "already active" confirmation and nothing is
    written.  A stale request from before a reset is refused.  The page always
    sends the serial it rendered, so a missing/garbled one is treated as
    invalid rather than as "any round".
    """
    expected_serial_text = request.form.get("round_serial", "").strip()
    try:
        expected_serial = int(expected_serial_text)
    except ValueError:
        flash("去标识请求无效（轮次标识不正确），请刷新页面后重试。", "error")
        return redirect(url_for("index"))
    outcome = enable_deidentification(expected_serial)
    if outcome.status in ("activated", "active"):
        flash(outcome.message, "success")
    else:
        flash(outcome.message, "error")
    return redirect(url_for("index"))


@app.route("/period/<int:period_no>", methods=["GET", "POST"])
def period_detail(period_no: int):
    db = get_db()
    periods = round_periods(db)
    if periods is None:
        flash("请先固定本轮计划。", "error")
        return redirect(url_for("plan"))
    if not 1 <= period_no <= periods:
        abort(404)

    plan_row = db.execute(
        "SELECT period, planned_users, prob_b FROM plan WHERE period = ?",
        (period_no,),
    ).fetchone()
    if plan_row is None:
        abort(404)

    if request.method == "POST":
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            flash("请选择要上传的 CSV 文件。", "error")
            return redirect(url_for("period_detail", period_no=period_no))
        raw = upload.read(MAX_CSV_BYTES + 1)
        if len(raw) > MAX_CSV_BYTES:
            flash(f"CSV 超过 {MAX_CSV_BYTES // 1024} KiB 限制。", "error")
            return redirect(url_for("period_detail", period_no=period_no))
        try:
            parsed = parse_csv(raw)
            validate_upload_against_plan(parsed, int(plan_row["planned_users"]))
        except CSVError as exc:
            flash(f"上传失败：{exc}", "error")
            return redirect(url_for("period_detail", period_no=period_no))

        outcome = store_upload(period_no, parsed)
        flash(outcome.message, "success" if outcome.status in ("accepted", "duplicate") else "error")
        if outcome.status in ("error", "conflict"):
            return redirect(url_for("period_detail", period_no=period_no), code=303)
        return redirect(url_for("period_detail", period_no=period_no))

    state = round_state(db)
    upload_row = state["uploads"].get(period_no)
    result_row = state["results"].get(period_no)
    plan = state["plan"]
    distributions = plan_distributions(plan)
    distribution = distributions[period_no - 1]
    expected_b = sum(
        Decimal(str(plan[i]["planned_users"])) * Decimal(str(plan[i]["prob_b"]))
        for i in range(period_no)
    )
    variance = sum(
        Decimal(str(plan[i]["planned_users"]))
        * Decimal(str(plan[i]["prob_b"]))
        * (Decimal(1) - Decimal(str(plan[i]["prob_b"])))
        for i in range(period_no)
    )
    summary = stats.distribution_stats(distribution, expected_b, variance)

    quantiles = []
    for q in ("0.001", "0.005", "0.025", "0.05", "0.5", "0.95", "0.975", "0.995", "0.999"):
        quantiles.append(
            (q, stats.quantile(distribution, Decimal(q)))
        )

    # A compact histogram for display.  Chunk masses are summed with Decimal
    # and only the (bounded, well-conditioned) ratio to the peak becomes a
    # float, so extreme probabilities cannot underflow the chart.
    width = max(1, (len(distribution) + 19) // 20)
    chunks = []
    for start in range(0, len(distribution), width):
        chunk = distribution[start : start + width]
        chunks.append((start, start + len(chunk) - 1, sum(chunk)))
    peak_mass = max((mass for _, _, mass in chunks), default=Decimal(0))
    histogram = []
    for start, end, mass in chunks:
        ratio = mass / peak_mass if peak_mass > 0 else Decimal(0)
        histogram.append(
            {
                "start": start,
                "end": end,
                "mass": stats.format_probability(mass, digits=3),
                "height": float(ratio),
            }
        )

    return render_template(
        "period.html",
        state=state,
        period_no=period_no,
        plan_row=plan_row,
        upload=upload_row,
        result=result_row,
        summary=summary,
        quantiles=quantiles,
        histogram=histogram,
        distribution=distribution,
    )


@app.get("/api/period/<int:period_no>")
def period_api(period_no: int):
    """Machine-readable view for the data team."""
    db = get_db()
    state = round_state(db)
    if not state["configured"] or not 1 <= period_no <= state["periods"]:
        abort(404)
    plan = state["plan"]
    distributions = plan_distributions(plan)
    distribution = distributions[period_no - 1]
    result = state["results"].get(period_no)
    payload = {
        "period": period_no,
        "planned_users": int(plan[period_no - 1]["planned_users"]),
        "prob_b": plan[period_no - 1]["prob_b"],
        "threshold": str(state["threshold"]),
        "distribution": [str(value) for value in distribution],
    }
    if result is not None:
        payload.update(
            {
                "verified": True,
                "verified_users": int(result["verified_users"]),
                "observed_b": int(result["observed_b"]),
                "expected_b": result["expected_b"],
                "variance": result["variance"],
                "tail_low": result["tail_low"],
                "tail_high": result["tail_high"],
                "p_value": result["p_value"],
                "flagged": bool(result["flagged"]),
            }
        )
    else:
        payload["verified"] = False
    return jsonify(payload)


@app.errorhandler(413)
def too_large(_error):
    flash(f"上传内容超过 {MAX_CSV_BYTES // 1024} KiB 限制。", "error")
    # Never echo an arbitrary Referer header into a redirect; go to a fixed
    # page instead.
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html"), 404


if __name__ == "__main__":
    bootstrap()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=True)
