.PHONY: setup run plan test lint fmt drills edge verify next-day clean

setup:        ## install everything (uv)
	uv sync

run:          ## start Farah's page
	uv run streamlit run app.py

plan:         ## same plan in the terminal
	uv run python run_plan.py

test:         ## unit tests + the brief's rules as invariants (tests/)
	uv run pytest

lint:         ## ruff check
	uv run ruff check .

fmt:          ## ruff format + autofix
	uv run ruff format . && uv run ruff check --fix .

drills:       ## condition-change drills -> scenarios/drill_results.md
	uv run python scenarios/drills.py

edge:         ## awkward-input scenarios
	uv run python scenarios/edge_cases.py

verify: lint test edge drills  ## everything a reviewer would run, in one go
	uv run ruff format --check .
	@echo "verify: lint, format, tests, edge cases and drills all passed"

clean:
	rm -rf state.json .session .pytest_cache .ruff_cache **/__pycache__

next-day:     ## simulate tomorrow from today's data + what was applied -> tools/next_day.csv
	uv run python tools/simulate_next_day.py
