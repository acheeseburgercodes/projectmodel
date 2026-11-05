"""Profile hubs, test cluster separation, and name the three operating segments."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

OUTPUTS = Path("outputs")
PROFILE_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    table = pd.read_csv("data/modelling_table.csv")
    profile = (
        table.groupby("hub_id", as_index=False)
        .agg(
            shipments=("shipment_id", "size"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
        .sort_values("hub_id")
        .reset_index(drop=True)
    )
    profile.to_csv(OUTPUTS / "hub_profile.csv", index=False, float_format="%.4f")

    scaled = StandardScaler().fit_transform(profile[PROFILE_COLUMNS])
    separation_rows = []
    solutions = {}
    for k in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled)
        labels = kmeans.labels_
        sizes = sorted(pd.Series(labels).value_counts().tolist(), reverse=True)
        distances = pairwise_distances(scaled)
        upper = np.triu(np.ones((len(profile), len(profile)), dtype=bool), k=1)
        same_cluster = labels[:, None] == labels[None, :]
        within = distances[upper & same_cluster].mean()
        across = distances[upper & ~same_cluster].mean()
        separation_rows.append({
            "k": k, "cluster_sizes": "-".join(str(size) for size in sizes),
            "across_over_within": across / within,
        })
        solutions[k] = labels
    pd.DataFrame(separation_rows).to_csv(OUTPUTS / "part_d_separation.csv", index=False, float_format="%.4f")

    clustered = profile.copy()
    clustered["cluster"] = solutions[3]
    delay_order = clustered.groupby("cluster")["delay_rate"].mean().sort_values().index
    names = {cluster: name for cluster, name in zip(delay_order, ["Standard", "Watch", "Escalate"])}
    clustered["segment"] = clustered["cluster"].map(names)
    clustered.drop(columns="cluster").to_csv(OUTPUTS / "hub_segments.csv", index=False, float_format="%.4f")

    hubs = pd.read_csv("data/hubs.csv")
    comparison = hubs.merge(clustered[["hub_id", "segment"]], on="hub_id", validate="one_to_one")
    expected = comparison["hub_tier"].map({1: "Standard", 2: "Watch", 3: "Escalate"})
    mismatches = comparison.loc[comparison["segment"] != expected, ["hub_id", "hub_city", "hub_tier", "segment"]]
    mismatches.to_csv(OUTPUTS / "tier_vs_segment_mismatch.csv", index=False)


if __name__ == "__main__":
    main()
