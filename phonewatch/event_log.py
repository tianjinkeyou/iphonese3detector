"""Durable, additive event history with an AI-readable daily text export."""

import json
import os
import sqlite3
import tempfile
import uuid
from datetime import date, datetime
from pathlib import Path


class EventJournal:
    """Store observations, then regenerate the day's UTF-8 text from SQLite.

    SQLite is the source of truth. The text is a shareable export, so deleting
    or editing it cannot change recorded events or the estimated pickup count.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.event_directory = self.directory / "events"
        self.session_id = uuid.uuid4().hex
        self.db = sqlite3.connect(self.directory / "usage.sqlite3")
        with self.db:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS activity_events (
                    id INTEGER PRIMARY KEY,
                    day TEXT NOT NULL,
                    at TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    details TEXT NOT NULL
                )
            """)
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS activity_events_day_id
                ON activity_events(day, id)
            """)

    @staticmethod
    def _day(day: str) -> str:
        if not isinstance(day, str):
            raise ValueError("Day must use YYYY-MM-DD")
        try:
            parsed = date.fromisoformat(day)
        except ValueError as error:
            raise ValueError("Day must use YYYY-MM-DD") from error
        if parsed.isoformat() != day:
            raise ValueError("Day must use YYYY-MM-DD")
        return day

    def record(self, at: datetime, kind: str, details: dict | None = None) -> int:
        """Commit one event and refresh its daily text; return the durable ID.

        A naive timestamp means the host's local time. A supplied timezone is
        retained, including its offset; the event belongs to that local date.
        Invalid values are rejected before writing either SQLite or the text.
        """
        if not isinstance(at, datetime):
            raise ValueError("Event time must be a datetime")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("Event kind must be a nonempty string")
        if details is None:
            details = {}
        if not isinstance(details, dict):
            raise ValueError("Event details must be a JSON object")
        # Serialize before the transaction. This also rejects NaN, infinity,
        # unsupported objects, cycles, and inconsistent dictionary key types.
        encoded_details = json.dumps(
            details, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        )
        if at.tzinfo is None or at.utcoffset() is None:
            at = at.astimezone()
        day = at.date().isoformat()
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO activity_events(day,at,session_id,kind,details) VALUES (?,?,?,?,?)",
                (day, at.isoformat(), self.session_id, kind, encoded_details),
            )
            event_id = cursor.lastrowid
        self.export_day(day)
        return event_id

    def pickup_total(self, day: str) -> int:
        day = self._day(day)
        return self.db.execute(
            "SELECT COUNT(*) FROM activity_events WHERE day=? AND kind='pickup'",
            (day,),
        ).fetchone()[0]

    def _timing_summary(self, day: str) -> list[str]:
        """Include existing persisted totals without reconstructing old data."""
        tables = {
            row[0] for row in self.db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        totals = {}
        if "daily_state" in tables:
            states = dict(self.db.execute(
                "SELECT state,seconds FROM daily_state WHERE day=?", (day,),
            ))
            totals["absent"] = states.get("absent", 0.0)
            totals["unknown"] = states.get("unknown", 0.0)
        if "watch_daily" in tables:
            row = self.db.execute(
                "SELECT seconds FROM watch_daily WHERE day=?", (day,),
            ).fetchone()
            totals["suspected_use"] = row[0] if row else 0.0
        if not totals:
            return []
        values = " ".join(f"{name}={seconds:.3f}" for name, seconds in totals.items())
        return [f"# 分开计时（秒，导出时已入库）：{values}"]

    def export_day(self, day: str) -> Path:
        """Rebuild a day in event-ID order and replace the text atomically."""
        day = self._day(day)
        rows = self.db.execute(
            "SELECT id,at,session_id,kind,details FROM activity_events WHERE day=? ORDER BY id",
            (day,),
        ).fetchall()
        pickups = sum(kind == "pickup" for _, _, _, kind, _ in rows)
        lines = [
            "# PhoneWatch 典型事件记录（UTF-8；以 # 开头为说明，其余每行是一个 JSON 对象）",
            f"# 日期：{day}；当日拿起次数（估计）：{pickups}",
            "# 此记录用于辅助分析：摄像头只记录外观和状态变化，不能证明手机身份、换机或实际使用。",
            "# pickup 默认由 E/F 停放姿势与空位参考确认从停放变为空位；presence 模式或缺少 E/F 时兼容旧的区域在位判断。",
            "# 启动时已为空位不算拿起；unknown 不等于 absent，只有 unknown 不会计次。",
            "# 先前确认停放后，严格 E 空位证据可快速计一次拿起并标注 rapid_empty=true，不必等待通常的 1 秒离开确认。",
            "# 整次换机若未被摄像头拍到或全程遮挡，无法可靠计次；rapid_empty 也不能证明手机身份或换机。",
            "# 有效摄像头画面出现长 unknown 后，只有先前已确认停放且随后 E/F 可靠确认空位，才最多补记一次 pickup。",
            "# 补记标注 gap_recovered=true、timing_uncertain=true 和 unknown_seconds；不推断间隙中的确切拿起时刻或发生了几次拿放。",
            "# 摄像头中断或校准重置后的变化不能补记；原有 absent、unknown 和疑似使用计时分类保持不变。",
            "# away_seconds 是确认拿起到放回的经过时间，可包含短暂 unknown；补记时从确认空位起算，不含之前的 unknown 间隙。",
            "# away_seconds 不是已核实使用时间，也不是 absent 累计时间。",
            "# present=确认在位；absent=确认离开；unknown/uncertain=检测不确定；疑似使用时间与离开、不确定时间分开。",
            "# 拿起次数从本版本开始记录，未推算旧历史；程序关闭或摄像头中断期间不推算拿起或使用。",
            "# 各次启动的 session 不同；同日记录按数据库 id 排列，time 保留记录时的时区偏移。",
            "# 常用事件：state_change=状态变化；pickup=估计拿起；return=放回；unobserved_change=未连续观察到的变化。",
            "# camera_lost/camera_restored=摄像头中断/恢复；suspect_start/suspect_end=疑似使用开始/结束；photo_saved=照片保存。",
            "# session_start/session_end=程序启动/结束；calibration_start/calibration_end=校准开始/结束；summary=累计数据快照。",
            *self._timing_summary(day),
            "# JSON 字段：id=事件编号，time=事件时间，event=事件类型，session=运行会话，details=辅助信息。",
        ]
        for event_id, at, session_id, kind, details in rows:
            lines.append(json.dumps(
                {"id": event_id, "time": at, "event": kind,
                 "session": session_id, "details": json.loads(details)},
                ensure_ascii=False, allow_nan=False, separators=(",", ":"),
            ))
        self.event_directory.mkdir(parents=True, exist_ok=True)
        path = self.event_directory / f"{day}.txt"
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=self.event_directory,
                prefix=f".{day}-", suffix=".tmp", delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write("\n".join(lines) + "\n")
                temporary.flush()
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path

    def close(self):
        self.db.close()
