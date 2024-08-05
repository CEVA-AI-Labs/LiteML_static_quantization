import torch
from liteml.ailabs_liteml.retrainer import RetrainerConfig, RetrainerModel
from ImageNet import get_dataset
from models.resnet import ResNet18_Weights
from models.resnet import resnet18
import tqdm
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def evaluate(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for (idx, batch) in enumerate(tqdm.tqdm(dataloader)):
            if idx % 100 == 0:
                torch.cuda.empty_cache()
            data = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(data)
            _, preds = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (preds == labels).sum().item()
        accuracy = 100 * correct / total
    return accuracy


def main():
    # Load model
    weights = ResNet18_Weights(ResNet18_Weights.IMAGENET1K_V1)
    model = resnet18(weights=weights).to(device)

    # Load dataset
    val_loader, calibration_loader, calibration_loader_key = get_dataset(dataset_folder='/AI_Labs/datasets/ImageNet', batch_size=16)
    # Load configuration file
    conf = RetrainerConfig('configuration.yaml')
    # Add calibration_loader and calibration_loader_key to the config
    conf["QAT"]["data_quantization"][ "calibration_loader"] = calibration_loader
    conf["QAT"]["data_quantization"]["calibration_loader_key"] = calibration_loader_key

    # Wrap model with LiteML's RetrainerModel
    q_model = RetrainerModel(model, config=conf).to(device)
    acc = evaluate(q_model, val_loader)
    print(f'Top1 accuracy = {acc:.2f}')


if __name__ == '__main__':
    main()