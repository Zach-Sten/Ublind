# Ublind 🎵 - Let your data speak for itself! -

**What does your data sound like?**

Ublind turns single-cell embeddings into music. Pick an embedding — UMAP, PCA, t-SNE, whatever — and ublind maps every dimension to sound. Each dimension sweeps independently through time on its own instrument, with counterpoint so voices move in opposite directions. Clusters become chords, trajectories become melodies, outliers become surprises.

Built for AnnData. Sometimes you can *hear* structure in data that you can't see.

> ⚠️ **Beta** — early development, expect rough edges. Feedback welcome.


## Install

Requires Python ≥ 3.10 and ffmpeg for animated visualizations.

```bash
pip install ublind

# ffmpeg (needed for sweep animations)
conda install -c conda-forge ffmpeg
```

Or install from source:

```bash
git clone https://github.com/Zach-Sten/Ublind.git
cd Ublind
pip install -e .
```

For realistic instrument sounds via SoundFont rendering (optional):

```bash
conda install -c conda-forge fluidsynth
pip install pyfluidsynth
```

## Quick start

```python
import scanpy as sc
import ublind as ub

adata = sc.read_h5ad("my_data.h5ad")

# Sweep mode — every dimension becomes a voice
ub.pp.preprocess(adata, embedding="X_umap", time=10, scale="pentatonic",
    dim_inst_map={0: "piano", 1: "cello"})
ub.tl.render(adata, "sweep.wav")
ub.pl.sweep(adata, color_by="cell_type", legend_loc="on data")

# Cluster mode — hear each cluster one at a time
ub.pp.preprocess_clusters(adata, embedding="X_umap", cluster_key="cell_type",
    time=10, order="largest", scale="pentatonic")
ub.tl.render(adata, "clusters.wav")
ub.pl.sweep_clusters(adata, legend_loc="on data")

# Interactive — click a cluster to hear its chord
ub.pl.interactive(adata, color_by="cell_type")
```

### Pick your instruments

```python
ub.pp.preprocess(
    adata,
    embedding="X_pca",
    time=15,
    dim_inst_map={0: "piano", 1: "cello", 2: "flute"},
    scale="minor",
    subsample=3000,
)
```

### Available scales

`pentatonic` · `major` · `minor` · `blues` · `dorian` · `mixolydian` · `whole_tone` · `chromatic`

### Available instruments

`piano` · `cello` · `violin` · `flute` · `guitar` · `harp` · `trumpet` · `sax` · `clarinet` · `vibraphone` — or any General MIDI program number.

### Cluster ordering

```python
ub.pp.preprocess_clusters(adata, cluster_key="leiden", order="largest")   # big → small
ub.pp.preprocess_clusters(adata, cluster_key="leiden", order="smallest")  # small → big
ub.pp.preprocess_clusters(adata, cluster_key="leiden", order="alphabetical")
ub.pp.preprocess_clusters(adata, cluster_key="leiden", order="custom",
    custom_order=["T cells", "B cells", "Monocytes"])
```

## Author

Zachary Stensland — UCSF
