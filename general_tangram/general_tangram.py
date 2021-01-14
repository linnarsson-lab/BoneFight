from typing import List, Tuple
import numpy as np
import torch
from torch.nn.functional import softmax, cosine_similarity
from tqdm import trange
from .view import View


class GeneralTangram:
	def __init__(self, a: View, b: View, lambda_v: float = 1, lambda_t: float = 1, lambda_f: float = 1, lambda_r: float = 0, device: str = "cpu") -> None:
		"""
		Construct a generalized Tangram model

		Args:
			a			The source View
			b			The target View
			lambda_v	The volume hyperparameter (default: 1)
			lambda_t	The tiles similarity hyperparameter (default: 1)
			lambda_f	The features similarity hyperparameter (default: 1)
			lambda_r	The regularization hyperparameter
		"""
		self.a = a
		self.b = b
		assert a.n_features == b.n_features, f"Both views must have the same number of features but {a.n_features} ≠ {b.n_features}"
		self.lambda_v = lambda_v
		self.lambda_t = lambda_t
		self.lambda_f = lambda_f
		self.lambda_r = lambda_r
		self.device = device

		self._M = torch.tensor(np.random.uniform(0, 1, size=(a.n_tiles, b.n_tiles)), device=device, requires_grad=True, dtype=torch.float32)
		self._A = torch.tensor((a.X * a.volume[..., None]).reshape((-1, a.n_features)), device=device, dtype=torch.float32)  # volume-weighted expression matrix
		self._B = torch.tensor(b.X.reshape(-1, b.n_features), device=device, dtype=torch.float32)
		self._volume_a = torch.tensor(a.volume.flatten(), device=device, dtype=torch.float32)
		self._volume_b = torch.tensor(b.volume.flatten(), device=device, dtype=torch.float32)
		
		self.M = np.zeros(0)  # Placeholder for the training result
		self.losses: List[int] = []

	def _loss_fn(self, verbose=True):
		M_probs = softmax(self._M, dim=1)  # Rows now sum to 1

		# Predict the volume distribution of view b
		v_pred = self._volume_a @ M_probs
		v_distribution = v_pred / v_pred.sum()
		volume_loss = self.lambda_v * torch.nn.KLDivLoss(reduction='sum')(torch.log(v_distribution), self._volume_b)

		# Predict the expression matrix of view b from view a
		B_pred = M_probs.t() @ self._A
		f_expression_loss = -self.lambda_f * cosine_similarity(B_pred, self._B, dim=1).mean()
		t_expression_loss = -self.lambda_t * cosine_similarity(B_pred, self._B, dim=0).mean()

		# Entropy regularization
		regularization_loss = -self.lambda_r * (torch.log(M_probs) * M_probs).sum()

		return volume_loss + t_expression_loss + f_expression_loss + regularization_loss

	def fit(self, num_epochs: int, learning_rate: float = 0.1) -> None:
		"""
		Fit the generalized Tangram model

		Args:
			num_epochs		Number of epochs
			learning_rate	Learning rate (default: 0.1)

		Remarks:
			After optimization, the mapping matrix M of shape (b.n_tiles, a.n_tiles) is saved as self.M
		"""
		optimizer = torch.optim.Adam([self._M], lr=learning_rate)
		with trange(num_epochs) as t:
			for _ in t:
				loss = self._loss_fn()
				lossn = loss.detach().numpy()
				t.set_postfix(loss=lossn)
				self.losses.append(lossn)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		# take final softmax w/o computing gradients
		with torch.no_grad():
			self.M = softmax(self._M, dim=1).cpu().numpy().T

	def transform(self, X: np.ndarray) -> np.ndarray:
		"""
		Transform a feature tensor X from the source view (a) to the target view (b)

		Args:
			X	Input tensor with shape a.tiles_shape + (n_features,)

		Returns:
			The transformed tensor, with shape b.tiles_shape + (n_features,)
		
		Remarks:
			The first dimensions of the input matrix should correspond to those
			of the input tensor (a); for example, an image stack would have 
			three dimensions, while a single-cell dataset would have one. The 
			final dimension corresponds to all the distinct features that 
			you want to transfer; for example, to transfer a single feature,
			the last dimension should have length 1. 
		"""
		Y = self.M @ X.reshape(-1, X.shape[-1])
		return Y.reshape(self.b.tiles_shape + (X.shape[-1],))
