# OutlierDetector

!!! note "Exploration, not technical QC"
    `OutlierDetector` (including `cluster_kmeans`) is for interactively
    exploring/visualising tile metrics - e.g. browsing a slide's tissue
    diversity via k-means clusters. It is **not** used by the `clean` CLI
    command, which instead uses [`histoslice.qc.quality_control`](qc.md) for
    technical QC. K-means clusters tile metrics by similarity; on a real
    slide, biologically valid tissue types (tumor, stroma, adipose, ...) can
    easily end up in their own cluster, so treating a whole cluster as
    "outliers" is not a reliable technical QC signal.

::: histoslice.utils._process.OutlierDetector
