import torch
from tqdm import tqdm
def train_one_epoch(model, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: str,logger=None):

    """
    Train the model for one epoch. 
    """
    model.train()
    stats = {}

    pbar = tqdm(loader, desc="Training", leave=False)

    for step, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        losses = model(images, targets)
        losses["loss_total"].backward()
        optimizer.step()

        for k, v in losses.items():
            stats[k] = stats.get(k, 0.0) + float(v.detach())

        # update tqdm display
        pbar.set_postfix({
            "total": f"{losses['loss_total'].item():.3f}",
            "det": f"{losses['loss_det'].item():.3f}",
            "kd": f"{losses['loss_kd'].item():.3f}",
        })

    # average stats
    num_batches = max(len(loader), 1)
    stats = {k: v / num_batches for k, v in stats.items()}

    return stats