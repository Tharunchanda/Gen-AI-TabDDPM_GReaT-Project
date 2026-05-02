from copy import deepcopy
import torch
import os
import numpy as np
import zero
import _bootstrap
from tab_ddpm import GaussianMultinomialDiffusion
from utils_train import get_model, make_dataset, update_ema
import lib
import pandas as pd

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=torch.device('cuda:1'), constraint_anneal_warmup=0.2, distribution_anneal_warmup=0.2):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000
        self.constraint_anneal_warmup = constraint_anneal_warmup  # Fraction of steps for warmup
        self.initial_constraint_weight = diffusion.constraint_weight
        self.distribution_anneal_warmup = distribution_anneal_warmup
        self.initial_distribution_weight = diffusion.distribution_weight

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _anneal_constraint_weight(self, step):
        """Gradually increase constraint weight during training (warmup schedule)."""
        if self.initial_constraint_weight <= 0.0:
            return
        
        frac_done = step / self.steps
        warmup_frac = min(frac_done / self.constraint_anneal_warmup, 1.0)
        # Use smooth warmup: starts at 0.1x and ramps up to full weight
        self.diffusion.constraint_weight = self.initial_constraint_weight * (0.1 + 0.9 * warmup_frac)

    def _anneal_distribution_weight(self, step):
        if self.initial_distribution_weight <= 0.0:
            return
        frac_done = step / self.steps
        warmup_frac = min(frac_done / self.distribution_anneal_warmup, 1.0)
        self.diffusion.distribution_weight = self.initial_distribution_weight * (0.1 + 0.9 * warmup_frac)

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
            
            self._anneal_constraint_weight(step)
            self._anneal_distribution_weight(step)
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(
                        f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss} '
                        f'(constraint_weight: {self.diffusion.constraint_weight:.6f}, '
                        f'distribution_weight: {self.diffusion.distribution_weight:.6f})'
                    )
                self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1

def train(
    parent_dir,
    real_data_path = 'data/higgs-small',
    steps = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    constraint_regularization = None,
    distribution_regularization = None,
    num_numerical_features = 0,
    device = torch.device('cuda:1'),
    seed = 0,
    change_val = False
):
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)

    dataset = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    print(K)

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    print(d_in)
    
    print(model_params)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    model.to(device)

    # train_loader = lib.prepare_beton_loader(dataset, split='train', batch_size=batch_size)
    train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)

    constraint_config = None
    if constraint_regularization is not None and num_numerical_features > 0:
        enabled = bool(constraint_regularization.get('enabled', False))
        weight = float(constraint_regularization.get('weight', 0.0))
        lower_q = float(constraint_regularization.get('lower_quantile', 0.0005))
        upper_q = float(constraint_regularization.get('upper_quantile', 0.9995))
        if enabled and weight > 0.0:
            X_num_train = dataset.X_num['train']
            # Compute bounds from normalized data (should be approximately N(0,1))
            # Using 0.0005/0.9995 quantiles gives ~3.3 standard deviations
            lower_bounds = np.quantile(X_num_train, lower_q, axis=0).astype(np.float32)
            upper_bounds = np.quantile(X_num_train, upper_q, axis=0).astype(np.float32)
            
            # For robustness, ensure bounds are at least reasonable (>= 4 sigma apart)
            bound_ranges = upper_bounds - lower_bounds
            min_range = 0.5  # For normalized data, minimum useful range
            for i in range(len(bound_ranges)):
                if bound_ranges[i] < min_range:
                    center = (lower_bounds[i] + upper_bounds[i]) / 2
                    lower_bounds[i] = center - min_range / 2
                    upper_bounds[i] = center + min_range / 2
            
            constraint_config = {
                'weight': weight,
                'lower_bounds': lower_bounds.tolist(),
                'upper_bounds': upper_bounds.tolist(),
            }
            print(
                f"Constraint regularization enabled: weight={weight}, "
                f"quantiles=({lower_q}, {upper_q})"
            )
            print(f"  Lower bounds: {lower_bounds[:3]}... (min: {lower_bounds.min():.4f})")
            print(f"  Upper bounds: {upper_bounds[:3]}... (max: {upper_bounds.max():.4f})")

    distribution_config = None
    if distribution_regularization is not None and num_numerical_features > 0:
        enabled = bool(distribution_regularization.get('enabled', False))
        weight = float(distribution_regularization.get('weight', 0.0))
        if enabled and weight > 0.0:
            X_num_train = dataset.X_num['train']
            target_mean = X_num_train.mean(axis=0).astype(np.float32)
            target_std = X_num_train.std(axis=0).astype(np.float32)
            target_std = np.maximum(target_std, 1e-6)
            distribution_config = {
                'weight': weight,
                'mean_weight': float(distribution_regularization.get('mean_weight', 1.0)),
                'std_weight': float(distribution_regularization.get('std_weight', 1.0)),
                'target_mean': target_mean.tolist(),
                'target_std': target_std.tolist(),
            }
            print(
                f"Distribution regularization enabled: weight={weight}, "
                f"mean_weight={distribution_config['mean_weight']}, std_weight={distribution_config['std_weight']}"
            )




    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        constraint_regularization=constraint_config,
        distribution_regularization=distribution_config,
        device=device
    )
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device
    )
    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))