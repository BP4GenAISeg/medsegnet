# utils/metric_collector.py

import logging
from enum import Enum
from collections import defaultdict
from typing import Any, Callable, Dict, List, Union

import numpy as np
import torch

from utils.assertions import ensure


class Agg(Enum):
    MEAN = "mean"
    SUM = "sum"
    LAST = "last"
    LIST_MEAN = "list_mean"
    RAW = "raw"


def _agg_raw(vals: List[Any]) -> str:
    return ",".join(map(str, vals))


def _agg_mean(vals: List[float]) -> Union[float, None]:
    return float(np.mean(vals)) if vals else None


def _agg_sum(vals: List[float]) -> Union[float, None]:
    return float(np.sum(vals)) if vals else None


def _agg_last(vals: List[Any]) -> Any:
    return vals[-1] if vals else None


def _agg_list_mean(vals: List[Any]) -> Any:
    """
    Compute the mean over a list of scalars or a list of equal-length lists (or higher-d arrays):
      - 1D list of numbers -> float mean
      - ND list            -> elementwise mean over the first axis
    """
    arr = np.array(vals, dtype=float)
    if arr.ndim == 1:
        return float(np.nanmean(arr)) if vals else None
    # for 2D, 3D, etc: mean across axis=0, return nested lists
    return np.nanmean(arr, axis=0).tolist()


# map each Agg to its function
_AGG_FN: Dict[Agg, Callable[[List[Any]], Any]] = {
    Agg.RAW: _agg_raw,
    Agg.MEAN: _agg_mean,
    Agg.SUM: _agg_sum,
    Agg.LAST: _agg_last,
    Agg.LIST_MEAN: _agg_list_mean,
}


class MetricCollector:
    """
    Collect per-batch scalar lists and collapse them at the end.
    """

    def __init__(self):
        self._data: Dict[str, List[Any]] = defaultdict(list)
        self._rules: Dict[str, Agg] = {}
        self._processed = 0
        self._skipped = 0

    def reset(self) -> None:
        self._data.clear()
        self._processed = 0
        self._skipped = 0

    def set_rule(self, name: str, agg: Union[Agg, str]) -> None:
        if isinstance(agg, str):
            agg = Agg(agg)
        self._rules[name] = agg

    def update(self, batch_metrics: Dict[str, Any]) -> None:
        """
        batch_metrics must contain only Python numbers or nested lists thereof.
        """
        for k, v in batch_metrics.items():
            self._processed += 1
            self._data[k].append(v)

    def skip(self) -> None:
        self._skipped += 1

    def aggregate(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, vals in self._data.items():
            rule = self._rules.get(name, Agg.MEAN)
            fn = _AGG_FN[rule]
            out[name] = fn(vals) if vals else None

        out["num_processed"] = self._processed
        out["num_skipped"] = self._skipped
        return out


class TrainingCollector(MetricCollector):
    def __init__(self):
        super().__init__()
        # epoch uses LAST
        self.set_rule("ep", Agg.LAST)
        # for per-head or per-class lists, use LIST_MEAN
        for k in ("loss_list", "dice_list", "cls_dice_list", "cons_loss_list"):
            self.set_rule(k, Agg.LIST_MEAN)

    def summary(self) -> Dict[str, Any]:
        agg = super().aggregate()

        ep = agg.get("ep")
        loss_list = agg.get("loss_list") or []
        dice_list = agg.get("dice_list") or []
        cls_dice_list = agg.get("cls_dice_list") or []
        cons_loss_list = agg.get("cons_loss_list") or []

        n_scales = len(loss_list)
        weights = [1.0 / n_scales] * n_scales if n_scales else []

        # weighted averages across scales
        w_loss = (
            float(sum(w * l for w, l in zip(weights, loss_list))) if loss_list else None
        )
        w_dice = (
            float(sum(w * d for w, d in zip(weights, dice_list))) if dice_list else None
        )
        avg_cons_loss = float(np.mean(cons_loss_list)) if cons_loss_list else None

        return {
            "ep": ep,
            "w_loss": w_loss,
            "loss_list": loss_list or None,
            "w_dice": w_dice,
            "dice_list": dice_list or None,
            "cls_dice_list": cls_dice_list or None,
            "avg_cons_loss": avg_cons_loss,
            "cons_loss_list": cons_loss_list or None,
        }


class ValidationCollector(MetricCollector):
    def __init__(self):
        super().__init__()
        # epoch uses LAST
        self.set_rule("ep", Agg.LAST)
        # for per-head or per-class lists, use LIST_MEAN
        for k in ("loss_list", "dice_list", "cls_dice_list"):
            self.set_rule(k, Agg.LIST_MEAN)

    def summary(self) -> Dict[str, Any]:
        agg = super().aggregate()

        ep = agg.get("ep")
        loss_list = agg.get("loss_list") or []
        dice_list = agg.get("dice_list") or []
        cls_dice_list = agg.get("cls_dice_list") or []

        n_scales = len(loss_list)
        weights = [1.0 / n_scales] * n_scales if n_scales else []

        # weighted averages across scales
        w_loss = (
            float(sum(w * l for w, l in zip(weights, loss_list))) if loss_list else None
        )
        w_dice = (
            float(sum(w * d for w, d in zip(weights, dice_list))) if dice_list else None
        )

        return {
            "ep": ep,
            "w_loss": w_loss,
            "loss_list": loss_list or None,
            "w_dice": w_dice,
            "dice_list": dice_list or None,
            "cls_dice_list": cls_dice_list or None,
        }


# TODO: Potential idea below, BaseCollector and MultiscaleCollector
class BaseCollector(MetricCollector):
    def __init__(self):
        super().__init__()
        self.set_rule("ep", Agg.LAST)
        self.set_rule("loss_list", Agg.LIST_MEAN)
        self.set_rule("dice_list", Agg.LIST_MEAN)
        self.set_rule("cls_dice_list", Agg.LIST_MEAN)

    def summary(self) -> Dict[str, Any]:
        agg = super().aggregate()
        ep = agg.get("ep")
        loss_list = agg.get("loss_list") or []
        dice_list = agg.get("dice_list") or []
        cls_dice_list = agg.get("cls_dice_list") or []

        w_loss = float(np.mean(loss_list)) if loss_list else None
        w_dice = float(np.mean(dice_list)) if dice_list else None

        return {
            "ep": ep,
            "w_loss": w_loss,
            "loss_list": loss_list or None,
            "w_dice": w_dice,
            "dice_list": dice_list or None,
            "cls_dice_list": cls_dice_list or None,
        }


class MultiscaleCollector(BaseCollector):
    def __init__(self):
        super().__init__()
        self.set_rule("cons_loss_list", Agg.LIST_MEAN)

    def summary(self) -> Dict[str, Any]:
        base_summary = super().summary()
        agg = super().aggregate()
        cons_loss_list = agg.get("cons_loss_list") or []
        avg_cons_loss = float(np.mean(cons_loss_list)) if cons_loss_list else None

        base_summary.update(
            {
                "avg_cons_loss": avg_cons_loss,
                "cons_loss_list": cons_loss_list or None,
            }
        )
        return base_summary
