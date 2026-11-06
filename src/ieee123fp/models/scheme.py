"""An end-to-end protection scheme that puts the study's pieces together, the way a relay
would use them: secure trip decision, then fault type, then fault zone for the layout the
grid is in right now.

    scheme = ProtectionScheme(f=4).fit(load())
    scheme.decide(X, ph, topo="T3_loop")   # -> DataFrame: votes, trip, ftype, zone, zone_conf
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from ..config import ZONES
from ..dataset import Dataset
from .defences import ResilientVote
from .topology_aware import TopologyAwareZoneLocator


class ProtectionScheme:
    """
    trip   : f-resilient vote -- more than f sensors must independently see a fault, so an
             attacker controlling <= f sensors cannot cause a trip alone.
    ftype  : gradient boosting on all features (transfers across layouts, see README).
    zone   : topology-aware locator decoded on the *current* layout's graph.
    """

    def __init__(self, f: int = 4, fpr: float = 0.01, random_state: int = 0):
        self.f, self.fpr, self.random_state = f, fpr, random_state

    def fit(self, ds: Dataset, cal_fraction: float = 0.2) -> ProtectionScheme:
        y = ds.meta.is_fault.values
        fit_idx, cal_idx = train_test_split(np.arange(len(ds)), test_size=cal_fraction,
                                            stratify=ds.meta.ftype, random_state=self.random_state)
        self.vote_ = ResilientVote(fpr=self.fpr).fit(ds.X[fit_idx], y[fit_idx], ds.X[cal_idx], y[cal_idx])
        faults = ds.faults()
        self.type_ = HistGradientBoostingClassifier(max_iter=300, random_state=self.random_state).fit(
            faults.X, faults.meta.ftype.values)
        self.zone_ = TopologyAwareZoneLocator(random_state=self.random_state).fit(
            faults.X, faults.ph, faults.meta.topo.values, faults.meta.bus.values)
        return self

    def decide(self, X: np.ndarray, ph: dict, topo: str) -> pd.DataFrame:
        """Decisions for events observed while the grid is in layout `topo`."""
        votes = self.vote_.votes(X)
        trip = votes >= self.f + 1
        out = pd.DataFrame({"votes": votes, "trip": trip, "ftype": "NONE", "zone": "NONE", "zone_conf": np.nan})
        if trip.any():
            sub_ph = {k: v[trip] for k, v in ph.items()}
            post = self.zone_.predict_proba(X[trip], sub_ph, topo)
            out.loc[trip, "ftype"] = self.type_.predict(X[trip])
            out.loc[trip, "zone"] = np.array(ZONES)[post.argmax(1)]
            out.loc[trip, "zone_conf"] = post.max(1)
        return out
