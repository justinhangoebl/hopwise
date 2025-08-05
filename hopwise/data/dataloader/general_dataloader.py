# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""hopwise.data.dataloader.general_dataloader
################################################
"""

from logging import getLogger

import pandas as pd
import torch

from hopwise.data.dataloader.abstract_dataloader import AbstractDataLoader, NegSampleDataLoader
from hopwise.data.interaction import Interaction


class TrainDataLoader(AbstractDataLoader):
    """TrainDataLoader is used for training. It can generate negative interaction when :attr:`training` is True.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self.sampler = sampler
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        self._batch_size = batch_size
        self.step = batch_size
        self.sample_size = len(self._dataset.inter_feat)

    @property
    def pr_end(self):
        return len(self._dataset.inter_feat)

    def _shuffle(self):
        self._dataset.shuffle()

    def _next_batch_data(self):
        interaction = self._dataset[self.pr:self.pr + self.step]
        self.pr += self.step
        return interaction

    def collate_fn(self, index):
        index = torch.tensor(index)
        interaction = self._dataset[index]
        interaction = self.sampler.sample_by_user_ids(interaction, self._dataset)
        return interaction


class FullSortRecEvalDataLoader(AbstractDataLoader):
    """FullSortRecEvalDataLoader is used for full sort evaluation. It can generate all candidate items for each user.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self.sampler = sampler
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field

        # Get user dataframe for evaluation
        if isinstance(dataset.inter_feat, pd.DataFrame):
            self.user_df = dataset.inter_feat.groupby(self.uid_field).first().reset_index()
        else:
            # Handle Interaction object - convert to pandas DataFrame for groupby operation
            inter_dict = {}
            for field in dataset.inter_feat.interaction:
                inter_dict[field] = dataset.inter_feat[field].numpy()
            inter_df = pd.DataFrame(inter_dict)
            self.user_df = inter_df.groupby(self.uid_field).first().reset_index()

        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        self._batch_size = batch_size
        self.step = batch_size
        self.sample_size = len(self.user_df)

    @property
    def pr_end(self):
        return len(self.user_df)

    def _shuffle(self):
        # No shuffling for evaluation
        pass

    def _next_batch_data(self):
        user_df = self.user_df[self.pr:self.pr + self.step]
        self.pr += self.step
        return user_df

    def collate_fn(self, index):
        index = torch.tensor(index)
        user_df = self.user_df.iloc[index]

        # Create interaction with all items for each user
        user_ids = user_df[self.uid_field].values
        n_users = len(user_ids)
        n_items = self._dataset.item_num

        # Create full interaction matrix
        user_ids_expanded = torch.tensor(user_ids).repeat_interleave(n_items)
        item_ids_expanded = torch.arange(n_items).repeat(n_users)

        interaction = Interaction({
            self.uid_field: user_ids_expanded,
            self.iid_field: item_ids_expanded
        })

        # Create history index for masking historical interactions
        # Get historical interactions for these users
        history_index = None
        if hasattr(self._dataset, 'inter_feat'):
            # Create a sparse matrix to track user-item interactions
            user_item_matrix = {}
            inter_feat = self._dataset.inter_feat

            # Build user-item interaction history
            if hasattr(inter_feat, 'interaction'):
                # Handle Interaction object
                hist_users = inter_feat[self.uid_field].numpy()
                hist_items = inter_feat[self.iid_field].numpy()
            else:
                # Handle DataFrame
                hist_users = inter_feat[self.uid_field].values
                hist_items = inter_feat[self.iid_field].values

            for u, i in zip(hist_users, hist_items):
                if u not in user_item_matrix:
                    user_item_matrix[u] = set()
                user_item_matrix[u].add(i)

            # Create history mask
            history_mask = []
            for user_idx, user_id in enumerate(user_ids):
                user_start = user_idx * n_items
                if user_id in user_item_matrix:
                    for item_id in user_item_matrix[user_id]:
                        if item_id < n_items:  # Ensure item_id is valid
                            history_idx = user_start + item_id
                            # Ensure the index is within bounds for the current batch
                            if history_idx < n_users * n_items:
                                history_mask.append(history_idx)

            if history_mask:
                history_index = torch.tensor(history_mask)

        # Create positive user and item indices
        # For full sort evaluation, we need to identify positive interactions
        positive_u = torch.arange(n_users)
        positive_i = user_df[self.iid_field].values if self.iid_field in user_df.columns else torch.zeros(n_users, dtype=torch.long)

        return interaction, history_index, positive_u, torch.tensor(positive_i)


class NegSampleDataLoader(AbstractDataLoader):
    """NegSampleDataLoader is used for negative sampling during training.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self.sampler = sampler
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        self._batch_size = batch_size
        self.step = batch_size
        self.sample_size = len(self._dataset.inter_feat)

    def collate_fn(self, index):
        index = torch.tensor(index)
        interaction = self._dataset[index]
        interaction = self.sampler.sample_by_user_ids(interaction, self._dataset)
        return interaction


class NegSampleEvalDataLoader(NegSampleDataLoader):
    """NegSampleEvalDataLoader is used for negative sampling evaluation.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self.sampler = sampler
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        self._batch_size = batch_size
        self.step = batch_size
        self.sample_size = len(self._dataset.inter_feat)

    def collate_fn(self, index):
        index = torch.tensor(index)
        interaction = self._dataset[index]
        interaction = self.sampler.sample_by_user_ids(interaction, self._dataset)

        # For negative sampling evaluation, we need to return 4 values
        # interaction, row_idx, positive_u, positive_i
        user_ids = interaction[self.uid_field]
        item_ids = interaction[self.iid_field]

        # Create row indices (user indices in the batch)
        unique_users, inverse_indices = torch.unique(user_ids, return_inverse=True)
        row_idx = inverse_indices

        # Create positive user and item indices
        positive_u = torch.arange(len(unique_users))
        positive_i = item_ids

        return interaction, row_idx, positive_u, positive_i


class FullSortLPEvalDataLoader(AbstractDataLoader):
    """FullSortLPEvalDataLoader is used for link prediction evaluation.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self.sampler = sampler
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        self._batch_size = batch_size
        self.step = batch_size
        self.sample_size = len(self._dataset.inter_feat)

    def collate_fn(self, index):
        index = torch.tensor(index)
        interaction = self._dataset[index]
        return interaction
