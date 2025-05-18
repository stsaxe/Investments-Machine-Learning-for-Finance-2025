import torch

from modeling.AbstractModelWrapper import AbstractModelWrapper


class Predictor(AbstractModelWrapper):
    def predict(self, test_loader: torch.utils.data.DataLoader) -> tuple[list, list]:
        self.check_model()
        self.check_device()

        predictions = []
        targets = []

        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            for x, t in test_loader:
                x = x.to(self.device)
                t = t.to(self.device)

                out = self.model(x)

                t = t.squeeze()
                out = out.squeeze()

                predictions.extend(out.detach().cpu().tolist())
                targets.extend(t.detach().cpu().tolist())

        torch.cuda.empty_cache()

        return predictions, targets
