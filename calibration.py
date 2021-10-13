import torch

class Identity(torch.nn.Module):
    def __init__(self): 
        super(Identity, self).__init__()

    def forward(self, x): 
        return x

    def predict(self, x):
        return x

class LogisticRegressionModel(torch.nn.Module): 
  
    def __init__(self): 
        super(LogisticRegressionModel, self).__init__() 
        self.linear = torch.nn.Linear(1, 1)  # One in and one out 
  
    def forward(self, x):
        return self.linear(x)

def LRcalibrator(netG, netD, data_loader, device, nz=100, calib_frac=0.1):
    n_batches = int(calib_frac * len(data_loader))
    batch_size = data_loader.batch_size
    # define a shortcut
    def gen_scores(batch_size):
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        x = netG(noise)
        return netD(x)
    print('prepare real scores ...')
    scores_real = []
    for i, (data, _) in enumerate(data_loader):
        scores_real.append(netD(data.to(device)))
        if i > n_batches:
            break
    scores_real = torch.cat(scores_real, dim=0)
    print('prepare fake scores ...')
    scores_fake = torch.cat([gen_scores(batch_size) for _ in range(n_batches)], dim=0)

    print('training LR calibrator ...')
    model = LogisticRegressionModel().to(device)
    x = torch.cat([scores_real, scores_fake], dim=0)
    y = torch.cat([torch.ones_like(scores_real),
                torch.zeros_like(scores_fake)], dim=0)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
    for epoch in range(5000):
        optimizer.zero_grad()
        with torch.enable_grad():
            pred_y = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, y)
            loss.backward(retain_graph=True)
        optimizer.step()
        if epoch % 1000 == 0: 
            print('Epoch: %d; Loss:%.3f' % (epoch, loss.item()))
    return model