# Bone Fight

A Python package for generalized alignment of omics datasets.

Bone Fight can be used to create a probabilistic map that relates one omics dataset
to another; this map can then be used to transfer properties from the source dataset
to the target dataset.

![Ostomachion](Ostomachion2.jpg)

Bone Fight is based on [Tangram](https://github.com/broadinstitute/Tangram) (Biancalani* T., Scalia* G. et al. 2020 - Deep learning and alignment of spatially-resolved whole transcriptomes of single cells in the mouse brain with Tangram. biorXiv doi:10.1101/2020.08.29.272831; photo courtesy of [Oh so souvenir](https://ohsosouvenir.com/products/gadgets/ostomachion-puzzle-game-detail).

Bone Fight extends and generalizes standard Tangram in several ways (see below). It's named after a game similar to Tangram, first described by 
Aristotle who named it [Ostomachion](https://en.wikipedia.org/wiki/Ostomachion) - "bone fight". It's believed that the pieces were made of bone, and that
two players would compete to see who would create a target figure the fastest.

## Volume priors

Tangram was designed to map single-cell data to spatial gene expression data. In order
to allocate cells to voxels, the algorithm accepts a density prior on the target (spatial) dataset,
which allows variation in the cell density across space. However, the source dataset
is implicitly given a uniform volume prior, i.e. each entity (single cell) is expected to occupy the same
volume of space in the target. This makes eminent sense for single-cell data, of course, since
cells are likely to be roughly uniform in size.

Bone Fight introduces a volume prior on the source dataset, which sets the expected "volume" of each entity in the
dataset. For single-cell data, a uniform prior makes sense (all cells occupy the same spatial volume) and the
results should be identical with standard Tangram.

However, exploiting the volume prior makes it possible to align not just single cells with spatial data, but also:

* Clusters of single cells with spatial data
* Spatial data with single cells (for reverse label transfer)
* Clusters of single cells with clusters of single cells (e.g. cross-species alignment)
* Spatial data with spatial data (e.g. consecutive sections)
...etc...

For example, you can align single-cell clusters (instead of individual single cells) by providing
the cluster-level expression matrix as input, along with a volume prior in the form of 
the vector of cluster sizes (number of cells per cluster). This is orders of magnitude faster
than fitting single-cell data directly, and can be significantly more robust (since cluster-level
expression profiles are more robust than single-cell profiles). 

As another example, Bone Fight could be used to map one set of single-cell clusters to another,
under the assumption that they represent the same underlying reality. For example, this could
be used to align mouse and human single-cell data from the same tissue.

## Views

To emphasize the symmetry of source and target datasets - both now come with volume priors - they 
are both described by View objects. You create a View by providing an expression tensor and a
volume prior.

To train a model, simply create the source and target views, set the hyperparameters, and call `fit()`:

```python
import general_tangram as gt
a = gt.View()
b = gt.View()
model = gt.GeneralTangram(a, b)
model.fit()
y = model.transform(x)
```

In Tangram, labels can only be transferred from single-cell data to spatial data. Using Bone Fight, you
can very easily train the reverse model, simply by exchanging the roles of the source and
target views; this lets you transfer labels from spatial to single-cell data. More generally, 
it enables volumetric alignment of any two omics datasets that correspond to different views of 
the same underlying reality (e.g. a particular tissue). 

## Multidimensional tensors

To avoid a lot of tedious data shuffling, Bone Fight accepts source and target views described by
tensors of any rank (number of dimensions), which are preserved during label transfer. For example, the target 
tensor can be an image stack of shape (x, y, z, n_features), and label transfer (using `BoneFight.transform()`)
preserves this shape, which can then be used directly for downstream analysis. This works especially well
with [Napari](https://napari.org) for browsing the results.


## Installation

```
git clone https://github.com/linnarsson-lab/BoneFight
cd BoneFight
pip install -e .
```
