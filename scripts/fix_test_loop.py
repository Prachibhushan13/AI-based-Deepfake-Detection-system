with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2_new.py', 'rb') as f:
    content = f.read()

# Fix the test evaluation loop - fix indentation
content = content.replace(
    b'            features = features.to(device, non_blocking=True)\r\n'
    b'            targets = targets.to(device, non_blocking=True)\r\n'
    b'                    model_output = model(features)\r\n'
    b'                    loss = nn.functional.binary_cross_entropy_with_logits(model_output, targets)\r\n'
    b'            if loss.dim() > 0:\r\n'
    b'                loss = loss.mean()\r\n'
    b'            test_loss += loss.item() * len(targets)\r\n'
    b'                    test_logits.extend(model_output.detach().cpu().tolist())\r\n'
    b'            test_labels.extend(targets.detach().cpu().tolist())\r',
    b'            features = features.to(device, non_blocking=True)\r\n'
    b'            targets = targets.to(device, non_blocking=True)\r\n'
    b'            model_output = model(features)\r\n'
    b'            loss = nn.functional.binary_cross_entropy_with_logits(model_output, targets)\r\n'
    b'            if loss.dim() > 0:\r\n'
    b'                loss = loss.mean()\r\n'
    b'            test_loss += loss.item() * len(targets)\r\n'
    b'            test_logits.extend(model_output.detach().cpu().tolist())\r\n'
    b'            test_labels.extend(targets.detach().cpu().tolist())\r'
)

with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2_new.py', 'wb') as f:
    f.write(content)

print('Fixed test loop indentation')