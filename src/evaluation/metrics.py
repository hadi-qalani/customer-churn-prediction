from src.evaluation.registry import METRICS


def load_metrics(metric_names: list[str]):
    return [METRICS[name] for name in metric_names]
