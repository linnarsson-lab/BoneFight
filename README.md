# Bone Fight

A Python package for generalized volumetric alignment of omics datasets.

Bone Fight can be used to create a probabilistic map that relates one comprehensive omics dataset
to another; this map can then be used to transfer properties from the source dataset
to the target dataset. The source and target datasets are assumed to be 'comprehensive', i.e. they
both cover the same sample (tissue, organ, organoid, cell line, etc.) completely, albeit with potentially 
different resolution and parcellation into voxels, clusters, single cells etc.

![Ostomachion](Ostomachion2.jpg)

Bone Fight is based on [Tangram](https://github.com/broadinstitute/Tangram) (Biancalani* T., Scalia* G. et al. 2020 - Deep learning and alignment of spatially-resolved whole transcriptomes of single cells in the mouse brain with Tangram. biorXiv doi:10.1101/2020.08.29.272831; photo courtesy of [Oh so souvenir](https://ohsosouvenir.com/products/gadgets/ostomachion-puzzle-game-detail)).

All credit for the fundamental Tangram algorithm goes to the original authors and the Regev lab. Bone Fight extends and generalises standard 
Tangram in several useful ways (see below). It's named after a game similar to Tangram, first described by 
Aristotle who called it [Ostomachion](https://en.wikipedia.org/wiki/Ostomachion) - "bone fight". It's believed that the pieces were made of bone, and that two players would compete to see who would be the quickest to create a target figure.

## Volume priors

Tangram was designed to map single-cell data to spatial gene expression data. In order
to allocate cells to voxels, the algorithm accepts a density prior on the target (spatial) dataset,
which allows variation in the cell density across space. In contrast, the source dataset
is implicitly given a uniform volume prior, i.e. each entity (single cell) is expected to occupy the same
volume of space in the target. This makes eminent sense for single-cell data, of course, since
cells are likely to be roughly uniform in size.

However, both single-cell and spatial datasets can be large, e.g. in the millions of cells and voxels.
The projection matrix generated by Tangram is shaped (n_cells, n_voxels), which can quickly run
into terabytes of memory. Optimizing such a weight matrix by gradient descent would require
extreme hardware.

Bone Fight introduces a volume prior on the source dataset, which sets the expected "volume" of each entity in the
dataset. For single-cell data, a uniform prior makes sense (all cells occupy the same spatial volume) and the
results should be identical to standard Tangram.

However, exploiting the volume prior makes it possible to align not just single cells with spatial data, but also:

* Clusters of single cells with spatial data
* Spatial data with single cells (for reverse label transfer)
* Clusters of single cells with clusters of single cells (e.g. cross-species alignment)
* Spatial data with spatial data (e.g. consecutive sections)
...etc...

For example, you can align single-cell clusters (instead of individual single cells) by providing
the cluster-level expression matrix as input, along with a volume prior in the form of 
the vector of cluster sizes (number of cells per cluster). This requires orders of magnitude
less memory and is orders of magnitude faster than fitting single-cell data directly. 
It can also be significantly more robust (since cluster-level
expression profiles are more robust than single-cell profiles). 

As another example, Bone Fight could be used to map one set of single-cell clusters to another,
under the assumption that they represent the same tissue. For example, this could
be used to align mouse and human single-cell data from the same tissue.

## Views

To emphasize the symmetry of source and target datasets - both now come with volume priors - they 
are both described by View objects. You create a View by providing an expression tensor and a
volume prior.

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


## Usage

### Preparations

BoneFight needs a source dataset and a target dataset. Both datasets comprise an expression
tensor and a volume tensor. 

The expression tensors can be multidimensional, but the last dimension
must contain the features (e.g. genes). For example, the source tensor can be a single-cell
expression matrix of shape (n_cells, n_genes), and the target tensor can be a spatial 
dataset of shape (x, y, n_genes). 

The feature dimension is assumed to be the same for the source and target dataset. In other
words, the last dimension of source and target tensors must contain the same features
in the same order. For example, if the spatial dataset contains 100 genes, then you must
include only those same 100 genes, in the same order, in the single-cell dataset. 

The volume tensor is used to define the expected volume (in a general sense) of each entity
in a dataset. For example, in a spatial dataset you may have some voxels that are known not to
contain any cells; set their volumes to zero. You may also have some prior knowledge about the
expected density of cells in a voxel, e.g. based on total spot counts, or counts of nuclei.

Similarly, for a dataset of single-cell clusters, it's reasonable to set the volume of each
cluster to the number of cells in that cluster. If you know some clusters are over- or undersampled,
you can compensate by adjusting their volumes accordingly.

The absolute values of the volume vectors are not important; they will be normalized so that they
sum to one internally anyway. Thus you can also think of volume priors as density priors.

### Training a model

To train a model, create the source and target views:

```python
import bone_fight as bf
a = bf.View(tensor_a, volumes_a)
b = bf.View(tensor_b, volumes_b)
model = bf.BoneFight(a, b)
model.fit()
```

The `.fit()` method returns the model itself, so you can write more succinctly:

```python
model = bf.BoneFight(a, b).fit()
```

You can also set the number of epochs (the number of iterations of the optimization function)
and the learning rate:

```python
model = bf.BoneFight(a, b).fit(100, 0.1)
```

After the model has been trained, you can check convergence by plotting the loss
as a function of the epoch, which is stored as `model.losses`.


### Label transfer

Once your model has been trained, you can transfer additional properties (called 'labels')
from the source dataset to the target.

The first thing you want to transfer is probably the spatial distribution of each source entity.
You can do this by transforming the identity matrix (diagonal ones):

```python
import numpy as np
# Assume source dataset is (n_clusters, n_genes)
# Assume target dataset is (x, y, n_genes)

# Create an identity matrix, shape (n_clusters, n_clusters)
labels = np.eye(n_clusters)

# Transform it, and the result will be (x, y, n_clusters)
y = model.transform(labels)
```

Note that the shape of the target dataset is preserved (this is useful for
spatial data).

You can also transfer any other property of the source dataset in the same way. 
For example, you can transfer genes from a source
single-cell dataset to a target spatial dataset where they were not measured. Just
create a tensor for the 'labels' (i.e. gene expression vectors)
you want to transfer, and call `model.transform(labels)`. Note that the shape of
the labels tensor must match the shape of the source dataset in the first N - 1
dimensions, and the last dimension should be the number of different labels you want
to transfer. For example, if you want to transfer a single gene expression vector, 
the label tensor should be shaped (n_cells, 1), not (n_cells,).

To transfer categorical labels, use ['one-hot'](https://en.wikipedia.org/wiki/One-hot) encoding.

Labels can only be transfered from the source to the target. For reverse transfer, train 
the reverse model by exchanging the views.
