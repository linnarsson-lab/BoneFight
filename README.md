# Bone Fight

![Ostomachion](Ostomachion2.jpg)

This package is based on [Tangram](https://github.com/broadinstitute/Tangram) (Biancalani* T., Scalia* G. et al. 2020 - Deep learning and alignment of spatially-resolved whole transcriptomes of single cells in the mouse brain with Tangram. biorXiv doi:10.1101/2020.08.29.272831). Photo above, courtesy of [Oh so souvenir](https://ohsosouvenir.com/products/gadgets/ostomachion-puzzle-game-detail).

Bone Fight extends and generalizes standard Tangram in several ways (see below). It's named after a game similar to Tangram, first described by 
Aristotle who named it Ostomachion - "bone fight". It's believed that the pieces were made of bone, and that
two players would compete to see how could create a target figure the quickest.

## Volume priors

Standard Tangram takes a density prior on the target (spatial) dataset. However, the source dataset
is implicitly given a uniform volume prior, i.e. each entity (single cell) is expected to occupy the same
volume in the target.

General Tangram introduces a volume prior on the source dataset, which sets the expected "volume" of each entity in the
dataset. For single-cell data, a uniform prior makes sense (all cells occupy the same spatial volume) and the
results should be identical with standard Tangram.

However, exploiting the volume prior makes it possible to align
not just single cells with spatial data, but also:

* Clusters of single cells with spatial data
* Spatial data with single cells (for reverse label transfer)
* Clusters of single cells with clusters of single cells (e.g. cross-species alignment)
* Spatial data with spatial data (e.g. consecutive sections)
...etc...

For example, you can align single-cell clusters (instead of individual single cells) by providing
the cluster-level expression matrix as input, along with a volume prior in the form of 
the vector of cluster sizes (number of cells per cluster). This is orders of magnitude faster
than fitting single-cell data directly, and can be significantly more robust (since cluster-level
expression profiles are more robust). On the other hand, it's impossible to transfer single-cell 
level labels in this setting (only cluster-level labels can be transferred).

## Views

To emphasize the new symmetry of source and target datasets - both come with volume priors - they 
are now both described by View objects. You create a View by providing an expression tensor and a
volume prior.

To train a model, simply create the source and target views, set the hyperparameters, and call `fit()`.

In standard Tangram, labels can only be transferred from single-cell data to spatial data. In General
Tangram, you can very easily train the reverse model, simply by exchanging the roles of the source and
target views; this lets you transfer labels from spatial to single-cell data.

## Tensors as source and target

To avoid a lot of error-prone data shuffling, General Tangram takes source and target 
tensors of any rank (number of dimensions), which are preserved during label transfer. For example, the output 
tensor can be an image stack of shape (x, y, z, n_features), and label transfer (using `GeneralTangram.transform()`)
generates results in this same shape, which can then be used directly for downstream analysis. This works well
with Napari for browsing the results.


## Installation

```
git clone https://github.com/linnarsson-lab/GeneralTangram
cd GeneralTangram
pip install -e .
```

## Example use

```python
import general_tangram as gt
a = gt.View()
b = gt.View()
model = gt.GeneralTangram(a, b)
model.fit()
y = model.transform(x)
```

