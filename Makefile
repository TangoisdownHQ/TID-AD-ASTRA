# ============================================
# 🌌  TID-AD-ASTRA | Mission Control Makefile
# ============================================

PYTHON = ml/.venv/bin/python
API_PORT = 8000

run-api:
	@echo "🚀 Launching TID-AD-ASTRA API on port $(API_PORT)..."
	cd ml && .venv/bin/python -m uvicorn app.main:app --port $(API_PORT) --reload

cli-explain:
	@echo "🧠 Launching TID-AD-ASTRA Explainability Console..."
	$(PYTHON) -m app.cli.cli_explain

run:
	@echo "🚀 Initializing TID-AD-ASTRA Mission Console..."
	@echo "🧩 Starting backend server..."
	cd ml && nohup .venv/bin/python -m uvicorn app.main:app --port $(API_PORT) > ../api.log 2>&1 &
	@echo "🛰  Launching backend process and checking system heartbeat..."
	@TRIES=0; \
	until curl -s http://127.0.0.1:$(API_PORT)/health > /dev/null 2>&1; do \
		TRIES=$$((TRIES+1)); \
		case $$TRIES in \
			1) printf "🩺 [1] Initializing telemetry...";; \
			2) printf "\n🩺 [2] Checking planetary datasets...";; \
			3) printf "\n🩺 [3] Aligning orbital models...";; \
			4) printf "\n🩺 [4] Awaiting system greenlight...";; \
			*) printf ".";; \
		esac; \
		sleep 1; \
	done; \
	echo "\n✅ Backend online — system heartbeat confirmed."; \
	echo "🧠 Launching Explainability Console...\n"; \
	$(PYTHON) -m app.cli.cli_explain

stop:
	@echo "🧯 Stopping TID-AD-ASTRA API..."
	@pkill -f "uvicorn app.main:app" || echo "No API process found."

logs:
	@echo "📜 Showing latest 20 lines of API log:"
	@tail -n 20 api.log

clean:
	@echo "🧹 Cleaning Python cache files and logs..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@rm -f api.log

