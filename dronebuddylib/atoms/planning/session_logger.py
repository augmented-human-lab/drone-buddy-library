"""
Session Logger for VLM-based Planning Sessions.

Records VLM API request/response pairs, YOLO detection confidence scores,
and session outcome to a .txt file in the session_detail directory at the
root of the workspace.

File naming format: DD_MM_YYYY_HHMMSS.txt
Output directory:   <workspace_root>/session_detail/
"""

import os
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from dronebuddylib.utils.logger import Logger

logger = Logger()

# Root directory of the workspace (two levels up from this file:
#   this file => atoms/planning/ => atoms/ => dronebuddylib/ => root)
_THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE_DIR)))
SESSION_DETAIL_DIR = os.path.join(_WORKSPACE_ROOT, "session_detail")


@dataclass
class VLMCallRecord:
    """One complete VLM request/response interaction."""
    call_type: str          # e.g. "Plan Generation", "Plan Regeneration (Round 2)", "Object Description"
    system_prompt: str
    conversation_history: List[Dict[str, str]]  # [{role, content}, …] prior to this call
    user_message: str
    image_path: Optional[str]
    response_content: str
    latency_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class YOLODetectionRecord:
    """One YOLO detection event (item flagged as potential correct item)."""
    search_round: int           # 1 or 2
    waypoint: str
    target_object: str
    confidence: float           # highest-confidence frame that was sent to VLM


class SessionLogger:
    """
    Accumulates data throughout a single planning search session and writes
    the full log to a .txt file when save() is called.

    Usage
    -----
    logger = SessionLogger(target_object="cup", session_start_time=time.time())
    logger.record_vlm_call(...)          # called by PlannerAgent
    logger.record_yolo_detection(...)    # called by PlannerExecutor
    logger.save(success=True, reason="", duration=42.1, ...)
    """

    def __init__(self, target_object: str = "", session_start_time: float = 0.0):
        self.target_object = target_object
        self.session_start_time = session_start_time or time.time()
        self.vlm_calls: List[VLMCallRecord] = []
        self.yolo_detections: List[YOLODetectionRecord] = []
        # Absolute timestamp recorded the moment the user accepts going into round 2.
        # None when the session had only one search round.
        self.round_1_end_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def record_vlm_call(
        self,
        call_type: str,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        user_message: str,
        response_content: str,
        latency_seconds: float,
        image_path: Optional[str] = None,
    ):
        """Record one VLM request/response pair."""
        record = VLMCallRecord(
            call_type=call_type,
            system_prompt=system_prompt,
            conversation_history=list(conversation_history),
            user_message=user_message,
            image_path=image_path,
            response_content=response_content,
            latency_seconds=latency_seconds,
        )
        self.vlm_calls.append(record)
        logger.log_debug('SessionLogger', f'Recorded VLM call: {call_type} ({latency_seconds:.3f}s)')

    def record_yolo_detection(
        self,
        search_round: int,
        waypoint: str,
        target_object: str,
        confidence: float,
    ):
        """Record a YOLO detection event (item flagged as potential correct item)."""
        record = YOLODetectionRecord(
            search_round=search_round,
            waypoint=waypoint,
            target_object=target_object,
            confidence=confidence,
        )
        self.yolo_detections.append(record)
        logger.log_debug('SessionLogger',
                         f'Recorded YOLO detection: round {search_round}, conf {confidence:.4f}')

    def record_round_transition(self, transition_time: float):
        """
        Mark the end of round 1 / start of round 2.

        Call this immediately after the user accepts re-planning (before
        regenerate_plan is called), so the timestamp captures the boundary
        between the two search rounds as precisely as possible.
        """
        self.round_1_end_time = transition_time
        logger.log_debug('SessionLogger', 'Round 1 ended — transitioning to round 2')

    # ------------------------------------------------------------------
    # Save API
    # ------------------------------------------------------------------

    def save(
        self,
        success: bool,
        reason: str,
        session_duration: float,
        waypoints_visited: List[str],
        scans_performed: int,
        found_at_waypoint: Optional[str] = None,
        round_durations: Optional[List[float]] = None,
    ):
        """
        Write the full session log to a .txt file.

        The file is always written regardless of success or failure so that
        every session leaves a record.
        """
        try:
            os.makedirs(SESSION_DETAIL_DIR, exist_ok=True)
        except Exception as exc:
            logger.log_error('SessionLogger', f'Could not create session_detail directory: {exc}')
            return

        now = datetime.now()
        filename = now.strftime("%d_%m_%Y_%H%M%S") + ".txt"
        filepath = os.path.join(SESSION_DETAIL_DIR, filename)

        try:
            content = self._build_report(
                success=success,
                reason=reason,
                session_duration=session_duration,
                waypoints_visited=waypoints_visited,
                scans_performed=scans_performed,
                found_at_waypoint=found_at_waypoint,
                round_durations=round_durations,
                report_time=now,
            )
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.log_success('SessionLogger', f'Session log saved: {filepath}')
        except Exception as exc:
            logger.log_error('SessionLogger', f'Failed to write session log: {exc}')

    # ------------------------------------------------------------------
    # Internal formatting
    # ------------------------------------------------------------------

    def _build_report(
        self,
        success: bool,
        reason: str,
        session_duration: float,
        waypoints_visited: List[str],
        scans_performed: int,
        found_at_waypoint: Optional[str],
        report_time: datetime,
        round_durations: Optional[List[float]] = None,
    ) -> str:
        lines: List[str] = []

        sep_heavy = "=" * 70
        sep_light = "-" * 70

        # ── Header ────────────────────────────────────────────────────────
        lines.append(sep_heavy)
        lines.append("  DRONE BUDDY — PLANNING SESSION LOG")
        lines.append(sep_heavy)
        lines.append(f"  Session Date    : {report_time.strftime('%d/%m/%Y %H:%M:%S')}")
        lines.append(f"  Target Object   : {self.target_object or '(unknown)'}")
        outcome_str = "SUCCESS" if success else "FAILURE"
        lines.append(f"  Session Outcome : {outcome_str}")
        if reason:
            lines.append(f"  Reason          : {reason}")
        if success and found_at_waypoint:
            lines.append(f"  Found At        : {found_at_waypoint}")
        lines.append(sep_heavy)
        lines.append("")

        # ── VLM API Call Records ──────────────────────────────────────────
        if self.vlm_calls:
            lines.append(f"{'VLM API CALLS':^70}")
            lines.append(sep_heavy)
            lines.append("")

            for idx, call in enumerate(self.vlm_calls, start=1):
                lines.append(f"--- VLM CALL #{idx}  [{call.call_type.upper()}] ---")
                lines.append(f"Timestamp : {call.timestamp}")
                lines.append(f"Latency   : {call.latency_seconds:.3f} seconds")
                if call.image_path:
                    lines.append(f"Image sent: {call.image_path}")
                lines.append("")

                lines.append("[SYSTEM PROMPT]")
                lines.append(call.system_prompt.strip())
                lines.append("")

                # Prior conversation history (if any)
                if call.conversation_history:
                    lines.append("[PRIOR CONVERSATION HISTORY]")
                    for msg in call.conversation_history:
                        role = msg.get("role", "?").upper()
                        content = msg.get("content", "")
                        # Trim extremely long messages (e.g. base64 images in history)
                        if len(content) > 4000:
                            content = content[:4000] + "\n... [truncated — content exceeds 4000 chars] ..."
                        lines.append(f"[{role}]")
                        lines.append(content)
                        lines.append("")

                lines.append("[REQUEST — USER MESSAGE]")
                lines.append(call.user_message.strip())
                lines.append("")

                lines.append("[RESPONSE — RAW VLM OUTPUT]")
                lines.append(call.response_content.strip())
                lines.append("")
                lines.append(sep_light)
                lines.append("")
        else:
            lines.append("No VLM API calls were recorded for this session.")
            lines.append("")

        # ── YOLO Detection Records ────────────────────────────────────────
        lines.append(f"{'YOLO DETECTION RECORDS':^70}")
        lines.append(sep_heavy)
        lines.append("")

        if self.yolo_detections:
            for det in self.yolo_detections:
                lines.append(f"--- YOLO DETECTION  [SEARCH ROUND {det.search_round}] ---")
                lines.append(f"Waypoint      : {det.waypoint}")
                lines.append(f"Target Object : {det.target_object}")
                lines.append(f"Confidence    : {det.confidence:.4f}  ({det.confidence * 100:.2f}%)")
                lines.append("")
        else:
            lines.append("No YOLO detections were recorded (target object was not flagged in any scan).")
            lines.append("")

        # ── Session Summary ────────────────────────────────────────────────
        lines.append(sep_heavy)
        lines.append(f"{'SESSION SUMMARY':^70}")
        lines.append(sep_heavy)
        lines.append(f"Outcome          : {outcome_str}")
        lines.append(f"Total Duration   : {session_duration:.1f} seconds")
        
        # Per-round timing breakdown
        if round_durations and len(round_durations) >= 1:
            for i, rd in enumerate(round_durations):
                lines.append(f"  Round {i + 1} Duration: {rd:.1f} seconds")
        
        lines.append(f"Waypoints Visited: {', '.join(waypoints_visited) if waypoints_visited else 'None'}")
        lines.append(f"Scans Performed  : {scans_performed}")
        lines.append(f"VLM Calls Made   : {len(self.vlm_calls)}")
        lines.append(f"YOLO Detections  : {len(self.yolo_detections)}")
        if self.yolo_detections:
            conf_values = [f"{d.confidence:.4f}" for d in self.yolo_detections]
            lines.append(f"YOLO Conf Scores : {', '.join(conf_values)}")
        lines.append(sep_heavy)
        lines.append("")

        return "\n".join(lines)
