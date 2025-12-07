Dynamic Trust Engine — Prototype
This project implements a market-neutral pairs trading experiment where holding periods adapt based on a trust metric rather than fixed intervals.

Overview

The engine:
downloads price data via yfinance
estimates hedge ratios between assets
computes spread changes as PnL
assigns a trust score to each active pair
adjusts review frequency based on trust
applies volatility-adjusted risk controls

compares performance against:
fixed-interval rebalance
benchmark buy-and-hold

Outputs include:
equity curves
trust evolution visualization
PnL statistics (Sharpe ratio, drawdown, CAGR, etc)

Purpose
This repository is a research prototype.
It is intended to explore whether adaptive scheduling improves spread strategy behaviour relative to static rebalancing.

The implementation is minimal, experimental, and subject to revision.

Status
core logic implemented
performance comparison works
visualization included
further research pending

Planned extensions
better spread entry/exit logic
position sizing / vol targeting
cost models

regime conditioning
