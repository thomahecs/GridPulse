# Resume claim audit

Based on the supplied data and this experimental run, not on desired resume metrics.

| Claim | Status | Evidence |
| --- | --- | --- |
| 36 months of hourly training | Not verified | Only 16752 hourly training rows are supplied, about 23 months |
| 1–3 hour lags and previous-day load | Verified | lag_1h, lag_2h, lag_3h, lag_24h |
| Calendar features | Verified | Hour/month encoded from timestamps |
| Temperature features | Not verified | No measured or forecast temperature input in the archive |
| Holiday features | Not verified | No documented holiday features or holiday file in original training data |
| Grid search with chronological CV | Verified | Full run uses 12 settings / 3 ordered CV folds; quick mode does not tune |
| Unseen 3-month holdout | Not verified | Supplied test is 744 hours (2015-08-01 00:00:00 to 2015-08-31 23:00:00) |
| 54.9 MW MAE | Not verified | Measured 47.572 in the dataset's load units; MW unit is not verified |
| MAE equals 0.9% of mean load | Not verified | Measured 4.855% |
| Persistence MAE 91.7 | Not verified | Measured 58.951 |
| Seasonal naive MAE 109.2 | Not verified | Measured 76.480 |
| Ridge MAE 78.9 | Not verified | Measured 51.727 |
| 40.1% below persistence | Not verified | Measured 19.30% |
| 49.7% below seasonal naive | Not verified | Measured 37.80% |
| 30.4% below Ridge | Not verified | Measured 8.03% |
| Lag-1 top importance | Not verified | Current top feature: delta_1h_vs_24h |
| Evening peaks largest error source | Not verified | Highest-error hour: 10:00; evening (18–22) MAE 39.85, other hours 49.60 |
| Temperature/holiday ablation 59.8 → 54.9 (8.2%) | Not verified | No temperature or holiday covariate data; current ablation tests calendar/lag profiles only |
| Six chronological backtests | Verified | Measured 6 windows |
| Backtest MAE range 52.1–61.4 | Not verified | Observed 17.32–116.19 |
| Beat persistence in all six folds | Verified | Won 6 of 6 folds |

**Verified: 5 / 21.**

An unverified claim should not be presented as a reproduced result.
