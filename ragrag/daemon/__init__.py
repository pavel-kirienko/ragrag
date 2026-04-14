"""Ragrag daemon package.

The daemon owns long-lived state (loaded models, open knowledge bases) and
serves CLI clients over a Unix domain socket using JSON-RPC 2.0. Clients
auto-spawn the daemon if it is not already running; sandboxed or unsupported
environments fall back to running the search engine in-process.

See ROADMAP.md (Phase A) for the high-level design.
"""
