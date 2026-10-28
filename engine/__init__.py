"""Decision engine for the liquidation run-rate tool.

Everything deterministic lives here. The LLM (engine.llm) only reads the output of
this package; it never writes a number back into it.
"""
