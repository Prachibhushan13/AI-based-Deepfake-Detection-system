from pathlib import Path

import torch
from torch import nn

from torch_model import EfficientNetBiLSTM


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    model = EfficientNetBiLSTM(
        sequence_length=20,
        frame_chunk_size=2,
        pretrained=True,
    ).to(device)

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    # One synthetic video: 20 RGB frames at 224x224.
    videos = torch.rand(1, 20, 3, 224, 224, device=device)
    labels = torch.tensor([1.0], device=device)

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-3,
    )
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(videos)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()

    print("Output shape:", tuple(logits.shape))
    print("Loss:", round(loss.item(), 5))
    print("GPU memory allocated (GB):", round(
        torch.cuda.memory_allocated() / 1024**3, 2
    ))

    checkpoint = Path("backend/artifacts/models/torch_smoke_test.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    print("Checkpoint saved:", checkpoint)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
