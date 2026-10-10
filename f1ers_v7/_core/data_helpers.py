"""Race-control bad intervals and ISO timestamp conversion; extracted unchanged."""
from __future__ import annotations

from datetime import datetime


def _epoch(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _bad_intervals(messages):
    active, intervals = {}, []
    for item in sorted(messages,key=lambda a:a["date"]):
        when=_epoch(item["date"])
        key=(item.get("scope"),item.get("sector"))
        flag=item.get("flag")
        message=item["message"]
        if flag in ("YELLOW","DOUBLE YELLOW","RED"):
            active.setdefault(key,when)
        elif flag == "CLEAR":
            if key in active:intervals.append((active.pop(key),when))
        elif flag == "GREEN" and item.get("scope")=="Track":
            intervals.extend((v,when) for v in active.values());active.clear()
        elif item.get("category")=="SafetyCar":
            if "DEPLOYED" in message:active.setdefault(("SafetyCar",None),when)
            elif "ENDING" in message:
                # Conservative 60s buffer after ending message.
                if ("SafetyCar",None) in active:intervals.append((active.pop(("SafetyCar",None)),when+60))
    intervals.extend((v,float("inf")) for v in active.values())
    return intervals
