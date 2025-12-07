import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


@dataclass
class PairState:
    pair: Tuple[str, str]
    entry_date: pd.Timestamp
    last_review_idx: int
    next_review_idx: int
    equity: float = 1.0
    peak_equity: float = 1.0
    ret_history: deque = field(default_factory=lambda: deque(maxlen=20))
    trust: float = 0.5


class DynamicTrustArb:
    """
    Dynamic Trust & Survival Engine vs Fixed-Interval Holding.
    Uses hedge-ratio spread trading for each pair.
    """

    def __init__(
        self,
        candidate_pairs: List[Tuple[str, str]],
        start: str,
        end: str,
        benchmark: str = "SPY",
        initial_capital: float = 100_000.0,
        max_active_pairs: int = 3,
        w1: float = 0.6,
        w2: float = 0.4,
        pnl_span: float = 0.10,
        corr_lookback: int = 60,
        vol_window: int = 20,
        atr_window: int = 14,
        blacklist_days: int = 45,
        high_trust: float = 0.8,
        low_trust: float = 0.4,
        fixed_interval: int = 10,
    ):
        self.candidate_pairs = candidate_pairs
        self.start = start
        self.end = end
        self.benchmark = benchmark
        self.initial_capital = initial_capital
        self.max_active_pairs = max_active_pairs

        self.w1 = w1
        self.w2 = w2
        self.pnl_span = pnl_span
        self.corr_lookback = corr_lookback
        self.vol_window = vol_window
        self.atr_window = atr_window
        self.blacklist_days = blacklist_days
        self.high_trust = high_trust
        self.low_trust = low_trust
        self.fixed_interval = fixed_interval

        self.prices: Optional[pd.DataFrame] = None
        self.log_prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        self.dates: Optional[pd.DatetimeIndex] = None
        self.benchmark_equity: Optional[pd.Series] = None

        self.hedge_beta: Dict[Tuple[str, str], float] = {}  # (A,B) -> beta

        self.equity_dynamic: Optional[pd.Series] = None
        self.equity_fixed: Optional[pd.Series] = None
        self.trust_log: Optional[pd.DataFrame] = None

        self._prepare_data()
        self._compute_hedge_ratios()

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _prepare_data(self):
        tickers = sorted(
            list({t for pair in self.candidate_pairs for t in pair} | {self.benchmark})
        )

        data = yf.download(
            tickers=tickers,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if data is None or len(data) == 0:
            raise ValueError("No data downloaded. Check tickers, dates, or connection.")

        # Handle multi-index vs single-index, Adj Close vs Close
        if isinstance(data.columns, pd.MultiIndex):
            lvl0 = data.columns.get_level_values(0)
            if "Adj Close" in lvl0:
                price_field = "Adj Close"
            elif "Close" in lvl0:
                price_field = "Close"
            else:
                raise ValueError(
                    f"Downloaded data missing 'Adj Close'/'Close'. Top-level cols: {set(lvl0)}"
                )
            prices = data.xs(price_field, axis=1, level=0).copy()
        else:
            if "Adj Close" in data.columns:
                prices = data["Adj Close"].copy()
            elif "Close" in data.columns:
                prices = data["Close"].copy()
            else:
                raise ValueError(
                    f"Downloaded data missing 'Adj Close'/'Close'. Columns: {list(data.columns)}"
                )

        present = [c for c in prices.columns if c in tickers]
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            print("Warning: missing tickers in price data:", missing)

        prices = prices[present]
        prices = prices.sort_index(axis=1)
        prices = prices.ffill().bfill()

        self.prices = prices
        self.log_prices = np.log(prices.replace(0, np.nan)).ffill().bfill()
        self.returns = prices.pct_change().fillna(0.0)
        self.dates = prices.index

        if self.benchmark in self.returns.columns:
            bench_rets = self.returns[self.benchmark]
            self.benchmark_equity = (1 + bench_rets).cumprod() * self.initial_capital
        else:
            print(
                f"Warning: benchmark {self.benchmark} not found; benchmark curve disabled."
            )
            self.benchmark_equity = None

    # ------------------------------------------------------------------
    # Hedge ratios & spread utilities
    # ------------------------------------------------------------------
    def _compute_hedge_ratios(self):
        """
        Estimate a static hedge ratio beta for each pair using log prices:
        log(A) ~ alpha + beta * log(B)
        """
        for (a, b) in self.candidate_pairs:
            if a not in self.log_prices.columns or b not in self.log_prices.columns:
                self.hedge_beta[(a, b)] = 1.0
                continue

            xa = self.log_prices[a]
            xb = self.log_prices[b]
            df = pd.concat([xa, xb], axis=1).dropna()
            if len(df) < 30:
                beta = 1.0
            else:
                # regress log(A) on log(B)
                x = df.iloc[:, 1].values  # log B
                y = df.iloc[:, 0].values  # log A
                # y = alpha + beta x
                beta, _ = np.polyfit(x, y, 1)
            self.hedge_beta[(a, b)] = float(beta)

    def _spread_value(self, date_idx: int, pair: Tuple[str, str]) -> Optional[float]:
        """
        Spread S_t = log(A_t) - beta * log(B_t).
        """
        a, b = pair
        if (a not in self.log_prices.columns) or (b not in self.log_prices.columns):
            return None
        beta = self.hedge_beta.get((a, b), 1.0)
        try:
            la = float(self.log_prices.iloc[date_idx][a])
            lb = float(self.log_prices.iloc[date_idx][b])
        except Exception:
            return None
        if np.isnan(la) or np.isnan(lb):
            return None
        return la - beta * lb

    # ------------------------------------------------------------------
    # Utility methods (returns, trust, alpha, etc.)
    # ------------------------------------------------------------------
    def _pair_return(self, date_idx: int, pair: Tuple[str, str]) -> float:
        """
        Daily PnL of one spread unit ≈ change in spread (log A - beta log B).
        """
        if date_idx == 0:
            return 0.0
        s_t = self._spread_value(date_idx, pair)
        s_prev = self._spread_value(date_idx - 1, pair)
        if s_t is None or s_prev is None:
            return 0.0
        return float(s_t - s_prev)

    def _pair_corr_and_vol(
        self, date_idx: int, pair: Tuple[str, str]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Correlation between leg returns, and volatility of spread changes.
        """
        a, b = pair
        if date_idx < self.corr_lookback:
            return None, None

        # leg returns for correlation
        a_ret = self.returns[a].iloc[date_idx - self.corr_lookback : date_idx]
        b_ret = self.returns[b].iloc[date_idx - self.corr_lookback : date_idx]
        if a_ret.isna().any() or b_ret.isna().any():
            corr = None
        else:
            corr = a_ret.corr(b_ret)

        # spread changes for volatility
        idx_slice = range(date_idx - self.corr_lookback, date_idx)
        spread_changes = []
        for i in idx_slice:
            if i == 0:
                continue
            r = self._pair_return(i, pair)
            spread_changes.append(r)
        if len(spread_changes) == 0:
            vol = None
        else:
            vol = float(np.std(spread_changes))

        if corr is None or pd.isna(corr) or vol is None or pd.isna(vol):
            return None, None
        return float(corr), float(vol)

    def _pair_zscore(self, date_idx: int, pair: Tuple[str, str]) -> Optional[float]:
        """
        Z-score of spread at time t relative to its recent history.
        """
        if date_idx < self.corr_lookback:
            return None
        spreads = []
        for i in range(date_idx - self.corr_lookback, date_idx + 1):
            s = self._spread_value(i, pair)
            if s is None:
                return None
            spreads.append(s)

        window = np.array(spreads[:-1])  # history up to t-1
        if len(window) < 10:
            return None
        mu = window.mean()
        sigma = window.std()
        if sigma == 0 or np.isnan(sigma):
            return None
        z = (spreads[-1] - mu) / sigma
        return float(z)

    def _alpha_score(self, date_idx: int, pair: Tuple[str, str]) -> Optional[float]:
        """
        Alpha Score = |corr| * |Z-score| * volatility of spread changes.
        """
        corr, vol = self._pair_corr_and_vol(date_idx, pair)
        z = self._pair_zscore(date_idx, pair)
        if corr is None or vol is None or z is None:
            return None
        return float(abs(corr) * abs(z) * vol)

    def _normalized_pnl(self, equity: float) -> float:
        pnl_frac = equity - 1.0
        x = (pnl_frac + self.pnl_span) / (2 * self.pnl_span)
        return float(np.clip(x, 0.0, 1.0))

    def _corr_stability(self, date_idx: int, pair: Tuple[str, str]) -> float:
        """
        Stability of correlation between legs, using long vs short windows.
        """
        if date_idx < self.corr_lookback * 2:
            return 0.5
        a, b = pair
        a_all = self.returns[a].iloc[:date_idx]
        b_all = self.returns[b].iloc[:date_idx]
        if a_all.isna().any() or b_all.isna().any():
            return 0.5
        corr_long = a_all.corr(b_all)
        if pd.isna(corr_long):
            return 0.5
        a_short = a_all.iloc[-self.corr_lookback :]
        b_short = b_all.iloc[-self.corr_lookback :]
        corr_short = a_short.corr(b_short)
        if pd.isna(corr_short):
            return 0.5
        stab = 1.0 - min(abs(corr_short - corr_long), 1.0)
        return float(np.clip(stab, 0.0, 1.0))

    def _trust_score(self, date_idx: int, state: PairState) -> float:
        pnl_norm = self._normalized_pnl(state.equity)
        stab = self._corr_stability(date_idx, state.pair)
        trust = self.w1 * pnl_norm + self.w2 * stab
        return float(np.clip(trust, 0.0, 1.0))

    def _atr_from_state(self, state: PairState) -> float:
        if len(state.ret_history) == 0:
            return 0.0
        arr = np.array(state.ret_history, dtype=float)
        return float(np.mean(np.abs(arr)))

    # ------------------------------------------------------------------
    # Alpha selection
    # ------------------------------------------------------------------
    def _select_alpha_pairs(
        self,
        date_idx: int,
        universe: List[Tuple[str, str]],
        k: int,
    ) -> List[Tuple[str, str]]:
        scores = []
        for pair in universe:
            score = self._alpha_score(date_idx, pair)
            if score is None or np.isnan(score):
                continue
            scores.append((pair, score))
        if not scores:
            return []
        scores.sort(key=lambda x: x[1], reverse=True)
        return [p for p, s in scores[:k]]

    # ------------------------------------------------------------------
    # Backtest run methods
    # ------------------------------------------------------------------
    def run(self):
        self._run_dynamic()
        self._run_fixed()

    def _run_dynamic(self):
        dates = self.dates
        n = len(dates)
        equity_series = []
        active_pairs: Dict[Tuple[str, str], PairState] = {}
        blacklist_until: Dict[Tuple[str, str], int] = {}
        trust_log_rows = []

        capital = self.initial_capital

        start_idx = max(self.corr_lookback, self.vol_window) + 1
        if start_idx >= n:
            raise ValueError("Not enough data for the specified lookback windows.")

        for idx in range(start_idx, n):
            date = dates[idx]

            # release from blacklist
            available_universe = []
            for pair in self.candidate_pairs:
                until_idx = blacklist_until.get(pair, -1)
                if idx <= until_idx:
                    continue
                available_universe.append(pair)

            # update existing pairs
            for pair, state in list(active_pairs.items()):
                pair_ret = self._pair_return(idx, pair)
                state.equity *= (1.0 + pair_ret)
                state.peak_equity = max(state.peak_equity, state.equity)
                state.ret_history.append(pair_ret)

                state.trust = self._trust_score(idx, state)
                trust_log_rows.append(
                    {
                        "date": date,
                        "pair": f"{pair[0]}-{pair[1]}",
                        "trust": state.trust,
                        "active": True,
                    }
                )

            # stop loss
            for pair, state in list(active_pairs.items()):
                atr = self._atr_from_state(state)
                current_drawdown = (
                    (state.peak_equity - state.equity) / state.peak_equity
                    if state.peak_equity > 0
                    else 0.0
                )
                total_loss = max(0.0, 1.0 - state.equity)
                if (current_drawdown > 3.0 * atr) or (total_loss > 0.05):
                    blacklist_until[pair] = idx + self.blacklist_days
                    del active_pairs[pair]

            # trust-based review scheduling
            for pair, state in list(active_pairs.items()):
                if idx >= state.next_review_idx:
                    t = state.trust
                    state.last_review_idx = idx
                    if t > self.high_trust:
                        state.next_review_idx = idx + 15
                    elif t < self.low_trust:
                        state.next_review_idx = idx + 1
                    else:
                        state.next_review_idx = idx + 5

            # add new pairs if slots
            slots = self.max_active_pairs - len(active_pairs)
            if slots > 0 and available_universe:
                candidates = [p for p in available_universe if p not in active_pairs]
                if candidates:
                    k = min(slots, 3)
                    new_pairs = self._select_alpha_pairs(idx, candidates, k)
                    for pair in new_pairs:
                        state = PairState(
                            pair=pair,
                            entry_date=date,
                            last_review_idx=idx,
                            next_review_idx=idx + 5,
                            equity=1.0,
                            peak_equity=1.0,
                            ret_history=deque(maxlen=self.atr_window),
                        )
                        active_pairs[pair] = state

            if active_pairs:
                avg_equity = np.mean([s.equity for s in active_pairs.values()])
            else:
                avg_equity = 1.0
            portfolio_equity = capital * avg_equity
            equity_series.append((date, portfolio_equity))

        self.equity_dynamic = pd.Series(
            [v for _, v in equity_series],
            index=[d for d, _ in equity_series],
            name="DynamicTrust",
        )
        self.trust_log = pd.DataFrame(trust_log_rows)

    def _run_fixed(self):
        dates = self.dates
        n = len(dates)
        equity_series = []
        active_pairs: Dict[Tuple[str, str], PairState] = {}
        blacklist_until: Dict[Tuple[str, str], int] = {}

        capital = self.initial_capital

        start_idx = max(self.corr_lookback, self.vol_window) + 1
        if start_idx >= n:
            raise ValueError("Not enough data for the specified lookback windows.")

        next_global_review_idx = start_idx

        for idx in range(start_idx, n):
            date = dates[idx]

            # release from blacklist
            available_universe = []
            for pair in self.candidate_pairs:
                until_idx = blacklist_until.get(pair, -1)
                if idx <= until_idx:
                    continue
                available_universe.append(pair)

            # update existing pairs
            for pair, state in list(active_pairs.items()):
                pair_ret = self._pair_return(idx, pair)
                state.equity *= (1.0 + pair_ret)
                state.peak_equity = max(state.peak_equity, state.equity)
                state.ret_history.append(pair_ret)

            # stop loss
            for pair, state in list(active_pairs.items()):
                atr = self._atr_from_state(state)
                current_drawdown = (
                    (state.peak_equity - state.equity) / state.peak_equity
                    if state.peak_equity > 0
                    else 0.0
                )
                total_loss = max(0.0, 1.0 - state.equity)
                if (current_drawdown > 3.0 * atr) or (total_loss > 0.05):
                    blacklist_until[pair] = idx + self.blacklist_days
                    del active_pairs[pair]

            # fixed interval rebalance
            if idx >= next_global_review_idx:
                active_pairs.clear()
                if available_universe:
                    k = min(self.max_active_pairs, 3)
                    new_pairs = self._select_alpha_pairs(idx, available_universe, k)
                    for pair in new_pairs:
                        state = PairState(
                            pair=pair,
                            entry_date=date,
                            last_review_idx=idx,
                            next_review_idx=idx + self.fixed_interval,
                            equity=1.0,
                            peak_equity=1.0,
                            ret_history=deque(maxlen=self.atr_window),
                        )
                        active_pairs[pair] = state

                next_global_review_idx = idx + self.fixed_interval

            if active_pairs:
                avg_equity = np.mean([s.equity for s in active_pairs.values()])
            else:
                avg_equity = 1.0
            portfolio_equity = capital * avg_equity
            equity_series.append((date, portfolio_equity))

        self.equity_fixed = pd.Series(
            [v for _, v in equity_series],
            index=[d for d, _ in equity_series],
            name="FixedInterval",
        )

    # ------------------------------------------------------------------
    # Performance stats (unchanged)
    # ------------------------------------------------------------------
    def _compute_stats(self, equity: pd.Series, name: str) -> Dict[str, float]:
        if equity is None or equity.empty:
            return {"name": name}
        eq = equity.dropna()
        if len(eq) < 2:
            return {"name": name}
        start_val = float(eq.iloc[0])
        end_val = float(eq.iloc[-1])
        total_return = end_val / start_val - 1.0
        rets = eq.pct_change().dropna()
        if rets.empty:
            return {"name": name, "total_return": total_return}
        mean_daily = float(rets.mean())
        std_daily = float(rets.std())
        n_days = len(rets)
        ann_factor = np.sqrt(252.0)
        cagr = (end_val / start_val) ** (252.0 / n_days) - 1.0
        vol_annual = std_daily * ann_factor
        sharpe = (mean_daily / std_daily) * ann_factor if std_daily > 0 else np.nan
        running_max = eq.cummax()
        drawdown = (eq - running_max) / running_max
        max_dd = float(drawdown.min())
        max_dd_pct = -max_dd
        return {
            "name": name,
            "total_return": total_return,
            "CAGR": cagr,
            "vol_annual": vol_annual,
            "Sharpe": sharpe,
            "max_drawdown": max_dd_pct,
        }

    def summary(self) -> pd.DataFrame:
        if self.equity_dynamic is None or self.equity_fixed is None:
            raise ValueError("Run .run() before calling .summary().")
        rows = []
        rows.append(self._compute_stats(self.equity_dynamic, "Dynamic Trust"))
        rows.append(self._compute_stats(self.equity_fixed, "Fixed Interval"))
        if self.benchmark_equity is not None:
            rows.append(
                self._compute_stats(self.benchmark_equity, f"Benchmark ({self.benchmark})")
            )
        df = pd.DataFrame(rows).set_index("name")
        return df

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_equity(self):
        if self.equity_dynamic is None or self.equity_fixed is None:
            raise ValueError("Run the backtest first by calling .run().")

        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_dynamic.index, self.equity_dynamic.values, label="Dynamic Trust")
        plt.plot(self.equity_fixed.index, self.equity_fixed.values, label="Fixed Interval")
        if self.benchmark_equity is not None:
            plt.plot(
                self.benchmark_equity.index,
                self.benchmark_equity.values,
                label=f"Benchmark ({self.benchmark})",
            )
        else:
            print("Note: benchmark equity not available, skipping benchmark plot.")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.title("Equity Curve: Dynamic Trust vs Fixed Interval vs Benchmark")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_trust_timeline(self):
        if self.trust_log is None or self.trust_log.empty:
            raise ValueError("No trust log available. Run the dynamic backtest first.")
        df = self.trust_log.copy()
        pairs = sorted(df["pair"].unique())
        pair_to_y = {p: i for i, p in enumerate(pairs)}
        df["y"] = df["pair"].map(pair_to_y)
        fig, ax = plt.subplots(figsize=(12, 6))
        sc = ax.scatter(
            df["date"],
            df["y"],
            c=df["trust"],
            cmap="RdYlGn",
            alpha=0.8,
            edgecolor="k",
            s=40,
        )
        ax.set_yticks(list(pair_to_y.values()))
        ax.set_yticklabels(pairs)
        ax.set_xlabel("Date")
        ax.set_ylabel("Pair")
        ax.set_title("Trust Timeline (Dynamic Trust Engine)")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Trust Score")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    candidate_pairs = [
        ("SPY", "QQQ"),
        ("IWM", "SPY"),
        ("EFA", "EEM"),
        ("XLK", "XLF"),
        ("XLE", "XOP"),
    ]

    engine = DynamicTrustArb(
        candidate_pairs=candidate_pairs,
        start="2020-01-01",
        end="2024-12-31",
        benchmark="SPY",
        initial_capital=100_000.0,
    )

    engine.run()

    print(engine.summary())  # <-- numeric comparison

    engine.plot_equity()
    engine.plot_trust_timeline()

