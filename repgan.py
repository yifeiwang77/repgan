import torch

def batch_l2_norm_squared(z):
    return z.pow(2).sum(dim=tuple(range(1, len(z.shape))))

def repgan(netG, 
        netD, 
        calibrator, 
        device, 
        nz=100,
        batch_size=100, 
        clen=640, 
        tau=0.01, 
        eta=0.1):
    '''
    1) network config
    netG: generator network. Input: latent (B x latent_dim x 1 x 1). Output: images (B x C x H x W)
    netD: discriminator network. Input: images (B x C x H x W). Output: raw score (B x 1)
    calibrator: calibrator network for calibrating the discriminator score. Input: raw score (B x 1). Ouput: calibrated score: (B x 1)
    nz: the dimension of the latent z of the generator
    2) sampling config
    batch_size: number of samples per batch
    clen: length the Markov chain (only the last sample at the end of the chain is left)
    tau: step size in L2MC
    eta: scale of  white noise in L2MC. Default: sqrt(tau)
    3) update rule
    - (a) Langevin. z' = zk + tau/2 * grad + eta * epsilon
    - (b) MH test. Calculate alpha, and flip a coin with probability alpha.
    '''
    # shortcut for getting score and updated means (latent + tau/2 * grads) from latent
    def latent_grad_step(latent): 
        with torch.autograd.enable_grad():
            latent.requires_grad = True
            score = calibrator(netD(netG(latent))).squeeze() # latent -> sample -> score
            obj = torch.sum(score - batch_l2_norm_squared(latent)/2) # get L2MC grads
            grads = torch.autograd.grad(obj, latent)[0] 
        mean = latent.data + tau/2 * grads
        return score, mean

    # initialize the chain and step once
    old_latent = torch.randn(batch_size, nz, 1, 1, device=device, requires_grad=True) # assume gaussian prior, others also work
    old_score, old_mean = latent_grad_step(old_latent) # current score and next-step mean
    one = old_score.new_tensor([1.0])

    # MCMC transitions
    for _ in range(clen):
        # 1) proposal and step once
        prop_noise = torch.randn_like(old_latent) # draw the proposed noise in q(z'|z_k)
        prop_latent = old_mean + eta * prop_noise
        prop_score, prop_mean = latent_grad_step(prop_latent) # proposed score and next-step mean

        # 2a) calculating MH ratio (in log space for stability)
        score_diff = prop_score - old_score  # note that the scores are before-sigmoid logits (also work for WGAN)
        latent_diff = batch_l2_norm_squared(old_latent)/2 - batch_l2_norm_squared(prop_latent)/2
        noise_diff = batch_l2_norm_squared(prop_noise)/2 - batch_l2_norm_squared(old_latent-prop_mean)/(2*eta**2)
        alpha = torch.min((score_diff + latent_diff + noise_diff).exp(), one)

        # 2b) decide the acceptance with alpha
        accept = torch.rand_like(alpha) <= alpha

        # 2c) update stats with the binary mask
        old_latent.data[accept] = prop_latent.data[accept]
        old_score.data[accept] = prop_score.data[accept]
        old_mean.data[accept] = prop_mean.data[accept]

    # get the final samples
    fake_samples = netG(old_latent)
    return fake_samples