"""Runnable CIFAR-10 transformation comparison on frozen WRN-40-2 features."""
import argparse
from dataclasses import dataclass, asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import torch
from .core import (stratified_indices, test_pools, pair_indices, transform_images,
                   GaussianDensity, BootstrapHeads, score_bundle, thresholds_from_known,
                   shift_auc, paired_auc_difference)


@dataclass
class Config:
    data_root: str = 'data'
    out: str = 'results/wrn_compare'
    checkpoint: str | None = None
    download: bool = False
    device: str = 'auto'
    fit_per_class: int = 500
    test_per_class: int = 100
    members: int = 5
    seeds: tuple = (0, 1, 2)
    strengths: tuple = (0., .25, .5, .75, 1.)
    kinds: tuple = ('same_class_mix', 'cross_class_mix', 'gaussian_noise')
    noise_std: float = .3
    pca_dim: int = 32
    batch_size: int = 64
    fpr: float = .05
    bootstrap: int = 200
    threads: int = 4

    def validate(self):
        if self.fit_per_class < 2 or self.test_per_class < 2 or self.members < 2:
            raise ValueError('Need >=2 examples/class/pool and >=2 ensemble members')
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError('Need unique nonnegative seeds')
        if (not self.strengths or len(set(self.strengths)) != len(self.strengths)
                or any(not np.isfinite(a) or not 0 <= a <= 1 for a in self.strengths)):
            raise ValueError('Strengths must be unique and in [0,1]')
        if not self.kinds or set(self.kinds)-{'same_class_mix','cross_class_mix','gaussian_noise'}:
            raise ValueError('Unknown or empty transformation list')
        if not 0 < self.fpr < 1 or self.bootstrap < 2 or self.batch_size < 1 or self.threads < 1:
            raise ValueError('Invalid fpr/bootstrap/batch_size/threads')
        if self.pca_dim < 1 or not np.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError('Invalid PCA dimension or noise scale')


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_wrn(config):
    from pytorch_ood.model import WideResNet
    cached = Path(torch.hub.get_dir())/'checkpoints'/'wrn-cifar10-pt.pt'
    checkpoint = Path(config.checkpoint).expanduser() if config.checkpoint else cached
    if checkpoint.exists():
        model = WideResNet(num_classes=10, depth=40, widen_factor=2, pretrained=None)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()})
    elif config.download and config.checkpoint is None:
        model = WideResNet(num_classes=10, pretrained='cifar10-pt')
    else:
        raise FileNotFoundError(f'WRN checkpoint missing: {checkpoint}; provide checkpoint or enable download')
    device = config.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    # Read preprocessing from this checkpoint's library recipe, not guessed defaults.
    norm = WideResNet.transform_for('cifar10-pt').transforms[-1]
    return model, device, norm, checkpoint


def extract(model, images, device, norm, batch_size):
    features, logits = [], []
    mean = torch.tensor(norm.mean, device=device)[None, :, None, None]
    std = torch.tensor(norm.std, device=device)[None, :, None, None]
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            x = torch.as_tensor(images[start:start+batch_size].transpose(0, 3, 1, 2).copy(), device=device)
            z = model.features((x-mean)/std)
            features.append(z.cpu().numpy())
            logits.append(model.fc(z).cpu().numpy())
    return np.concatenate(features), np.concatenate(logits)


def run_experiment(config):
    config.validate()
    torch.set_num_threads(config.threads)
    out = Path(config.out); out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'manifest.json'
    source = {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    manifest = dict(status='running', config=asdict(config), source_sha256=source,
                    interpretation='Transformed-vs-clean discrimination, not semantic OOD detection',
                    model='Frozen cifar10-pt WRN-40-2; no spectral normalization/DDU retraining',
                    EU='Stratified bootstrap linear-head mutual information; backbone fixed',
                    real_wild_ood='NOT RUN; CIFAR-10 transformations only',
                    python=platform.python_version(),
                    versions={p: importlib.metadata.version(p) for p in
                              ('numpy','torch','torchvision','pytorch-ood','scikit-learn')})
    manifest_path.write_text(json.dumps(manifest, indent=2))
    try:
        from torchvision.datasets import CIFAR10
        train = CIFAR10(config.data_root, train=True, download=config.download)
        test = CIFAR10(config.data_root, train=False, download=config.download)
        model, device, norm, checkpoint = load_wrn(config)
        manifest.update(device=device, checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint))
        rows, summaries, contrasts, assignments, accuracies = [], [], [], [], []
        for seed in config.seeds:
            print(f'seed={seed}: extracting clean WRN features', flush=True)
            train_y, test_y = np.asarray(train.targets), np.asarray(test.targets)
            fit_ix = stratified_indices(train_y, config.fit_per_class, seed)
            pools = test_pools(test_y, config.test_per_class, seed+100)
            base_ix, cal_ix = pools['evaluation'], pools['calibration']
            for name, ix in dict(fit=fit_ix, **pools).items():
                assignments.extend(dict(seed=seed, pool=name, dataset='train' if name=='fit' else 'test',
                                        index=int(i), class_id=int((train_y if name=='fit' else test_y)[i])) for i in ix)
            pixels = lambda dataset, ix: dataset.data[ix].astype(np.float32)/255.
            fit_x = pixels(train, fit_ix)
            clean_x, cal_x = pixels(test, base_ix), pixels(test, cal_ix)
            fit_z, fit_logits = extract(model, fit_x, device, norm, config.batch_size)
            clean_z, clean_logits = extract(model, clean_x, device, norm, config.batch_size)
            cal_z, cal_logits = extract(model, cal_x, device, norm, config.batch_size)
            np.savez_compressed(out/f'clean_features_seed{seed}.npz',
                                fit_features=fit_z, fit_labels=train_y[fit_ix], fit_index=fit_ix,
                                calibration_features=cal_z, calibration_logits=cal_logits, calibration_index=cal_ix,
                                evaluation_features=clean_z, evaluation_logits=clean_logits, evaluation_index=base_ix,
                                calibration_labels=test_y[cal_ix], evaluation_labels=test_y[base_ix])
            density = GaussianDensity(config.pca_dim).fit(fit_z, train_y[fit_ix])
            heads = BootstrapHeads(config.members, seed=seed).fit(fit_z, train_y[fit_ix])
            calibration, _ = score_bundle(cal_z, cal_logits, density, heads)
            clean_scores, clean_pred = score_bundle(clean_z, clean_logits, density, heads)
            thresholds = thresholds_from_known(calibration, config.fpr)
            scales = {key: max(float(np.quantile(v,.75)-np.quantile(v,.25)),1e-8)
                      for key, v in calibration.items()}
            accuracies.append(dict(seed=seed, n=len(base_ix),
                                   WRN_accuracy=float((clean_logits.argmax(1)==test_y[base_ix]).mean()),
                                   bootstrap_head_accuracy=float((clean_pred==test_y[base_ix]).mean())))
            rng = np.random.default_rng(seed+200)
            noise = rng.normal(size=clean_x.shape).astype(np.float32)
            for kind in config.kinds:
                partner_ix = (pair_indices(test_y, base_ix, pools['partner'], kind, seed+300)
                              if kind.endswith('_mix') else np.full(len(base_ix), -1))
                partner_x = pixels(test, partner_ix) if kind.endswith('_mix') else None
                for strength in config.strengths:
                    print(f'seed={seed}: {kind}, strength={strength:g}', flush=True)
                    changed = transform_images(clean_x, strength, kind, partner_x, noise, config.noise_std)
                    if strength == 0:
                        z, logits, scores, pred = clean_z, clean_logits, clean_scores, clean_pred
                    else:
                        z, logits = extract(model, changed, device, norm, config.batch_size)
                        scores, pred = score_bundle(z, logits, density, heads)
                    for name, score in scores.items():
                        threshold = thresholds[name]
                        summaries.append(dict(seed=seed, transformation=kind, strength=float(strength),
                                              score=name, n=len(score), mean=float(score.mean()),
                                              mean_change=float((score-clean_scores[name]).mean()),
                                              standardized_change=float((score-clean_scores[name]).mean()/scales[name]),
                                              rejection_rate=float((score > threshold).mean()),
                                              clean_rejection_rate=float((clean_scores[name] > threshold).mean()),
                                              shift_AUROC=shift_auc(clean_scores[name],score),
                                              threshold=threshold, calibration_IQR=scales[name]))
                    for i, row_id in enumerate(base_ix):
                        row = dict(seed=seed, transformation=kind, strength=float(strength),
                                   row_id=int(row_id), source_class=int(test_y[row_id]),
                                   partner_id=int(partner_ix[i]),
                                   partner_class=int(test_y[partner_ix[i]]) if partner_ix[i]>=0 else -1,
                                   WRN_prediction=int(logits[i].argmax()), head_prediction=int(pred[i]),
                                   source_label_agreement=bool(logits[i].argmax()==test_y[row_id]))
                        for name, score in scores.items():
                            row[name] = float(score[i])
                            row[name+'_depth'] = float((score[i]-thresholds[name])/scales[name])
                            row[name+'_rejected'] = bool(score[i] > thresholds[name])
                        rows.append(row)
                    contrast = paired_auc_difference(clean_scores['density'], scores['density'],
                                                     clean_scores['EU'], scores['EU'], seed, config.bootstrap)
                    contrasts.append(dict(seed=seed, transformation=kind, strength=float(strength), **contrast))
                    if seed == config.seeds[0]:
                        np.savez_compressed(out/f'preview_{kind}_{strength:g}.npz', images=changed[:8],
                                            source_labels=test_y[base_ix[:8]])
            # Save progress after every completed seed.
            pd.DataFrame(rows).to_csv(out/'rows.csv', index=False)
            pd.DataFrame(summaries).to_csv(out/'summary_per_seed.csv', index=False)
            pd.DataFrame(contrasts).to_csv(out/'density_minus_EU.csv', index=False)
            pd.DataFrame(assignments).to_csv(out/'pool_assignments.csv', index=False)
            pd.DataFrame(accuracies).to_csv(out/'clean_accuracy.csv', index=False)
        summary = pd.DataFrame(summaries)
        summary.groupby(['transformation','strength','score']).agg(
            n_seeds=('seed','nunique'), shift_AUROC_mean=('shift_AUROC','mean'),
            shift_AUROC_std=('shift_AUROC','std'), rejection_mean=('rejection_rate','mean'),
            rejection_std=('rejection_rate','std')).reset_index().to_csv(out/'summary.csv', index=False)
        from .plots import plot_results
        plot_results(out)
        manifest.update(status='completed', rows=len(rows), seeds_completed=list(config.seeds),
                        limitation='One frozen backbone; seed SD is not backbone-training variance. '
                        'Pair-bootstrap intervals condition on the fitted heads/density and calibration. '
                        'A single-seed run has undefined seed SD (blank in CSV).')
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return summary
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        manifest_path.write_text(json.dumps(manifest, indent=2))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default='data')
    p.add_argument('--out', default='results/wrn_compare')
    p.add_argument('--checkpoint')
    p.add_argument('--download', action='store_true')
    p.add_argument('--device', default='auto')
    p.add_argument('--smoke', action='store_true', help='One seed, small real-CIFAR subset; no scientific conclusion')
    args = p.parse_args()
    config = Config(**{k:v for k,v in vars(args).items() if k!='smoke'})
    if args.smoke:
        config.fit_per_class, config.test_per_class = 20, 5
        config.seeds, config.strengths, config.bootstrap = (0,), (0., .5, 1.), 40
    run_experiment(config)


if __name__ == '__main__':
    main()
