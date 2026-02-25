# ublind 🎵

**What does your data sound like?**

ublind turns single-cell embeddings into music. Pick an embedding — UMAP, PCA, t-SNE, whatever, and ublind maps it to sound. The first dimension becomes **time**, the rest become **pitch**, each on its own instrument. Clusters become chords, trajectories become melodies, outliers become surprises.

Built for AnnData. Sometimes you can *hear* structure in data that you can't see. 

> ⚠️ **Beta** — early development, expect rough edges. Feedback welcome.


## Install

Requires Python ≥ 3.10 and ffmpeg for animated visualizations.

```bash
# ffmpeg (needed for sweep animations)
conda install -c conda-forge ffmpeg

# install ublind
cd ublind
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

# embedding → music
ub.pp.preprocess(adata, embedding="X_umap", time=10, scale="pentatonic")

# render audio
ub.tl.render(adata, "output.wav")

# animated sweep with sound
ub.pl.sweep(adata)
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

## Author

Zachary Stensland — UCSF
