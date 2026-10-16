import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from utils.experiments import make_xgb
from utils.features import build_features


def test_saved_model_roundtrip(tmp_path):
    n = 168
    t = pd.date_range("2023-01-01", periods=n, freq="h")
    d = pd.DataFrame({"time": t, "power_load": np.sin(np.arange(n) * 0.3) * 30 + 300})
    frame, names = build_features(d)
    model = make_xgb({"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1})
    model.fit(frame[names], frame.power_load)
    p1 = model.predict(frame[names])
    path = tmp_path / "roundtrip.json"
    model.save_model(str(path))
    loaded = XGBRegressor()
    loaded.load_model(str(path))
    np.testing.assert_allclose(p1, loaded.predict(frame[names]))
