import random
import torch
from torch_geometric.loader import DataLoader

torch.backends.cudnn.enabled = True
save_id = random.randint(0, 1000)


class Predictor:
    def __init__(self, model, test_dataset, batch_size=256, num_workers=8):
        self.model = model.cuda()
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers) if test_dataset is not None else None
        self.supervise_interaction = False
        self.train_output, self.val_output, self.test_output = None, None, None

    def test(self):
        test_output = self.test_iterations()
        return test_output

    def to_cuda(self, data):
        data.batch = data.batch.cuda()
        data.x = data.x.cuda()
        data.edge_attr = data.edge_attr.cuda()
        data.edge_index = data.edge_index.cuda()
        data.mol_node_num = data.mol_node_num.cuda()
        return data

    @torch.no_grad()
    def test_iterations(self):
        self.model.eval()
        dataloader = self.test_dataloader
        ys_pred = []
        for data in dataloader:
            mol_batch = self.to_cuda(data)
            y_pred = self.model(mol_batch)
            ys_pred.append(y_pred.cpu())
        return ys_pred

