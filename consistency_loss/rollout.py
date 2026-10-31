"""Recursive long-horizon evaluation of a fixed-horizon forecasting model.

The model keeps its training horizon. Only the evaluation data loader uses
the longer target horizon; predictions, never targets, update the context.
"""
import torch


def rollout_predict(model, history, train_pred_len, test_pred_len):
    """Extend [B, context_len, C] history into [B, test_pred_len, C] predictions.

    Each model call predicts train_pred_len steps. Feed that block back into
    the fixed-length context; the final block may exceed the requested horizon.
    The caller sets evaluation mode and disables gradients (see rollout_test).
    """
    if train_pred_len <= 0 or test_pred_len <= 0:
        raise ValueError("forecast horizons must be positive")
    context_len = history.shape[1]
    num_blocks = (test_pred_len + train_pred_len - 1) // train_pred_len
    predictions = []
    for _ in range(num_blocks):
        prediction = model(history, None, None, None)
        if prediction.shape[1] < train_pred_len:
            raise ValueError("model output is shorter than its training horizon")
        prediction = prediction[:, -train_pred_len:, :].detach()
        predictions.append(prediction)
        history = torch.cat((history, prediction), dim=1)[:, -context_len:, :]
    return torch.cat(predictions, dim=1)[:, :test_pred_len, :]


def rollout_test(model, test_loader, device, train_pred_len, test_pred_len):
    """Return global MSE/MAE, restoring the model's previous training mode.

    Sum errors over all samples, times and channels before dividing. Averaging
    batch means instead would over-weight a smaller final batch.
    """
    was_training = model.training
    model.eval()
    squared_error = absolute_error = 0.0
    elements = 0
    try:
        with torch.no_grad():
            for batch_x, batch_y, *_ in test_loader:
                x = batch_x.float().to(device)
                # TSLib may prepend label_len historical points to batch_y.
                target = batch_y.float().to(device)[:, -test_pred_len:, :]
                prediction = rollout_predict(model, x, train_pred_len, test_pred_len)
                if prediction.shape != target.shape:
                    raise ValueError("rollout target and prediction shapes differ")
                error = prediction - target
                squared_error += error.square().sum().item()
                absolute_error += error.abs().sum().item()
                elements += error.numel()
    finally:
        model.train(was_training)
    if not elements:
        raise ValueError("rollout test loader is empty")
    return squared_error / elements, absolute_error / elements
