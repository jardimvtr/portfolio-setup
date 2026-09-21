from __future__ import annotations

import atexit
from datetime import datetime
from pathlib import Path
import sys
import time


# ============================================================
# GLOBAL PORTFOLIO EXECUTION TIMER
# ============================================================

PORTFOLIO_ROOT = Path(
    r"C:\Users\vitor\Documents\Portfolio"
).resolve()

START_TIME = time.perf_counter()
START_DATETIME = datetime.now()


# ============================================================
# IDENTIFY EXECUTED SCRIPT
# ============================================================

def get_script_path():

    if not sys.argv:
        return None

    script = sys.argv[0]

    if not script:
        return None

    try:
        return Path(
            script
        ).resolve()

    except Exception:
        return None


SCRIPT_PATH = get_script_path()


# ============================================================
# CHECK WHETHER SCRIPT BELONGS TO PORTFOLIO
# ============================================================

def is_portfolio_script():

    if SCRIPT_PATH is None:
        return False

    try:
        SCRIPT_PATH.relative_to(
            PORTFOLIO_ROOT
        )

        return True

    except ValueError:
        return False


# ============================================================
# FORMAT TIME
# ============================================================

def format_elapsed(
    seconds,
):

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (
            seconds % 3600
        )
        // 60
    )

    remaining_seconds = (
        seconds % 60
    )

    if hours > 0:
        return (
            f"{hours:02d} h "
            f"{minutes:02d} min "
            f"{remaining_seconds:05.2f} sec"
        )

    return (
        f"{minutes:02d} min "
        f"{remaining_seconds:05.2f} sec"
    )


# ============================================================
# FORMAT TIMESTAMP
# ============================================================

def format_timestamp(
    timestamp,
):

    return timestamp.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# FINAL REPORT
# ============================================================

def report_execution_time():

    if not is_portfolio_script():
        return

    END_DATETIME = datetime.now()

    elapsed = (
        time.perf_counter()
        - START_TIME
    )

    print()

    print("=" * 80)

    print("GLOBAL EXECUTION TIME")

    print("=" * 80)

    print(
        f"Script: "
        f"{SCRIPT_PATH.name}"
    )

    print(
        f"Started: "
        f"{format_timestamp(START_DATETIME)}"
    )

    print(
        f"Finished: "
        f"{format_timestamp(END_DATETIME)}"
    )

    print(
        f"Total execution time: "
        f"{format_elapsed(elapsed)}"
    )

    print("=" * 80)


atexit.register(
    report_execution_time
)