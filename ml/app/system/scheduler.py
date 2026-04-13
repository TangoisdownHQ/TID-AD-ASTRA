# app/system/scheduler.py
import asyncio
import datetime
import threading
import traceback
from app.system.auto_retrain import auto_retrain
from app.system.selfaware import update_awareness_state
from app.system.update_datasets import refresh_if_stale

# =========================================================
# 🕒 CONFIGURATION
# =========================================================
SCHEDULE_INTERVAL_HOURS = 6   # dataset + retrain check every 6h
RETRY_BACKOFF_HOURS = 2       # retry sooner if previous cycle failed


# =========================================================
# 🧠 BACKGROUND LOOP
# =========================================================
async def schedule_retraining():
    """
    Periodically check for dataset drift and auto-retrain.
    Runs forever as an async background task.
    """
    print("🧠 Autonomous scheduler initialized.")

    retry_interval = SCHEDULE_INTERVAL_HOURS

    while True:
        now = datetime.datetime.now().isoformat(timespec="seconds")
        print(f"🧩 [Scheduler] {now} — Checking for retraining needs...")

        try:
            dataset_refreshed = refresh_if_stale(max_age_hours=SCHEDULE_INTERVAL_HOURS)
            result = auto_retrain()

            # Guard against incomplete returns
            status = result.get("status", "unknown")
            metrics = result.get("metrics", {})
            dataset_source = result.get("dataset_source", "unknown")

            update_awareness_state(
                last_scheduler_run=now,
                last_scheduler_status=status,
                last_scheduler_result=result,
                last_scheduler_metrics=metrics,
                last_dataset_source=dataset_source,
                last_scheduler_dataset_refresh=dataset_refreshed,
            )

            print(f"✅ [Scheduler] Cycle complete: {status}")
            retry_interval = SCHEDULE_INTERVAL_HOURS  # reset to normal interval

        except Exception as e:
            tb = traceback.format_exc()
            update_awareness_state(
                last_scheduler_run=now,
                last_scheduler_status="error",
                last_scheduler_error=str(e),
                last_scheduler_trace=tb
            )
            print(f"⚠️ [Scheduler] Error: {e}")
            print(tb)
            retry_interval = RETRY_BACKOFF_HOURS  # shorten interval after failure

        print(f"⏳ Sleeping for {retry_interval} hours...\n")
        await asyncio.sleep(retry_interval * 3600)


def start_scheduler_background():
    """
    Run the scheduler in a separate daemon thread so it doesn’t block FastAPI.
    """
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(schedule_retraining())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    print("🧠 Autonomous scheduler started in background thread.")
