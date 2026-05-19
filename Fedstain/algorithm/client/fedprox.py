import torch

from algorithm.client.fedavg import FedAvgClient
from utils.optimizers_shcedulers import get_optimizer
from utils.tools import trainable_params, get_best_device, local_time


class FedProxClient(FedAvgClient):
    def __init__(self, args, dataset, client_id, logger):
        super(FedProxClient, self).__init__(args, dataset, client_id, logger)

    def train(self):
        self.move2new_device()
        self.classification_model.train()
        criterion = torch.nn.CrossEntropyLoss()
        global_params = trainable_params(self.classification_model, detach=True)
        for epoch in range(self.args.num_epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if len(data) <= 1:
                    continue
                self.optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.classification_model(data)
                loss = criterion(output, target)
                loss.backward()
                for w, w_t in zip(trainable_params(self.classification_model), global_params):
                    w.grad.data += self.args.mu * (w.data - w_t.data)
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()

        average_loss = total_loss / len(self.train_loader)
        self.classification_model.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        self.logger.log(f"{local_time()}, Client {self.client_id}, Avg Loss: {average_loss:.4f}")
