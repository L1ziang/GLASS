import torch


__all__ = [
    'Attack_Dataset',
]


class Attack_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.intermediate_data = None  # Inputs
        self.actual_data = None  # Targets

    def push(self, intermediate, actual):
        assert intermediate.size(0) == actual.size(0)

        if self.intermediate_data is None:
            self.intermediate_data = intermediate
        else:
            self.intermediate_data = torch.cat([self.intermediate_data, intermediate])

        if self.actual_data is None:
            self.actual_data = actual
        else:
            self.actual_data = torch.cat([self.actual_data, actual])

    def __len__(self):
        if self.intermediate_data is None:
            return 0
        else:
            return self.intermediate_data.size(0)

    def __getitem__(self, idx):
        return self.intermediate_data[idx], self.actual_data[idx]