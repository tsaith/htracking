
def evaluate(model, criterion, loader, device):

    model.eval()   # Set model to evaluate mode

    num_samples = 0
    corrects = 0
    loss = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            batch_size = target.size(0)

            output = model(data)
            batch_loss = criterion(output, target)
            _, predicted = torch.max(output, 1)

            num_samples += batch_size
            loss += batch_loss.item() * batch_size
            corrects += (predicted == target).sum().item()

    loss /= num_samples
    acc = float(corrects) / num_samples

    return loss, acc

def predict(data, batch_size, model, device):

    model.eval()   # Set model to evaluate mode
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                         shuffle=False, num_workers=32, pin_memory=True)

    proba = []
    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)

            output = model(batch_data)
            p =  torch.exp(output)
            proba.append(p.cpu().numpy())

    proba = np.concatenate(proba, axis=0)

    return proba

