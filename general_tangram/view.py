from typing import Tuple
import numpy as np


class View:
	def __init__(self, X: np.ndarray, volume: np.ndarray) -> None:
		"""
		A view consists of a set of tiles measured on a shared feature space,
		each occupying a known volume (amount of space).

		Args:
			X			The expression tensor, of shape (<dims>, n_features)
			volume		The volume tensor, of shape (<dims>,)

		Remarks:
			The expression tensor may be multidimensional (e.g. an image, or image stack), but
			the last dimension must contain the features (e.g. genes).

			The volume tensor must have the same shape as the expression tensor without the 
			last (features) dimension.

			The volume tensor will be normalized so that it sums to 1.
		"""
		assert X.shape[:-1] == volume.shape, f"Shape of volume tensor {volume.shape} must match shape of expression tensor {X.shape[:-1]}"
		self.X = X.astype("float32")
		self.volume = (volume / volume.sum()).astype("float32")

	@property
	def shape(self) -> Tuple[int, ...]:
		return self.X.shape

	@property
	def tiles_shape(self) -> Tuple[int, ...]:
		return self.X.shape[:-1]

	@property
	def n_tiles(self) -> int:
		return np.prod(self.X.shape[:-1])

	@property
	def n_features(self) -> int:
		return self.X.shape[-1]
