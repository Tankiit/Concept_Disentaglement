"""Frozen CLIP ViT-B/32 image features for CIFAR-10 and CIFAR-10-C, extracted on Modal.

The encoder is never trained. Images are resized to 224 bicubic and CLIP-normalised on the GPU,
and the L2-normalised 512-d embedding is stored as float16, one NPZ per split:

    cifar10_train.npz, cifar10_test.npz          X (N x 512), y (N,)
    cifar10c_<corruption>.npz                    X (50000 x 512), y, severity (1..5)

Run:   modal run real/modal_extract.py
       modal run real/modal_extract.py::more      # DINOv2 ViT-S/14 on CIFAR-10; both backbones on CIFAR-100
Fetch: modal volume get reach-aistats features cache/

Naming: CLIP files keep the original names (cifar10_train.npz, cifar100_train.npz, ...); other
backbones are prefixed (dinov2_cifar10_train.npz). CIFAR-100 files also store y_coarse.
"""
import modal

app = modal.App('reach-aistats-features')
volume = modal.Volume.from_name('reach-aistats', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('wget')
         .pip_install('torch==2.4.1', 'torchvision==0.19.1', 'open_clip_torch==2.26.1', 'numpy<2'))

CORRUPTIONS = ('gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
               'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
               'elastic_transform', 'pixelate', 'jpeg_compression')
C10C_URL = 'https://zenodo.org/records/2535967/files/CIFAR-10-C.tar'


@app.function(image=image, gpu='A10G', volumes={'/vol': volume}, timeout=3*3600)
def extract():
    import os, subprocess, numpy as np, torch, torch.nn.functional as F, open_clip, torchvision
    out = '/vol/features'; os.makedirs(out, exist_ok=True)
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.cuda().eval().half()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(images):  # images: N x 32 x 32 x 3 uint8
        feats = []
        for start in range(0, len(images), 2000):
            x = torch.from_numpy(images[start:start+2000]).cuda().permute(0, 3, 1, 2).float()/255.
            x = F.interpolate(x, size=224, mode='bicubic', align_corners=False).clamp(0, 1)
            z = model.encode_image(((x-mean)/std).half()).float()
            feats.append(F.normalize(z, dim=-1).cpu().numpy().astype(np.float16))
        return np.concatenate(feats)

    for train in (True, False):
        name = f"{out}/cifar10_{'train' if train else 'test'}.npz"
        if os.path.exists(name):
            continue
        ds = torchvision.datasets.CIFAR10('/tmp/c10', train=train, download=True)
        np.savez(name, X=embed(ds.data), y=np.asarray(ds.targets, dtype=np.int16))
        print('saved', name, flush=True)
    volume.commit()

    if not all(os.path.exists(f'{out}/cifar10c_{c}.npz') for c in CORRUPTIONS):
        subprocess.run(['wget', '-q', '-O', '/tmp/c10c.tar', C10C_URL], check=True)
        subprocess.run(['tar', '-xf', '/tmp/c10c.tar', '-C', '/tmp'], check=True)
    labels = None
    for c in CORRUPTIONS:
        name = f'{out}/cifar10c_{c}.npz'
        if os.path.exists(name):
            continue
        labels = np.load('/tmp/CIFAR-10-C/labels.npy').astype(np.int16)
        images = np.load(f'/tmp/CIFAR-10-C/{c}.npy')
        np.savez(name, X=embed(images), y=labels, severity=np.repeat(np.arange(1, 6), 10000).astype(np.int8))
        print('saved', name, flush=True)
        volume.commit()


@app.function(image=image, gpu='A10G', volumes={'/vol': volume}, timeout=3600)
def extract_dataset(backbone: str, dataset: str):
    """Frozen features for one (backbone, dataset): train and test splits."""
    import os, numpy as np, torch, torch.nn.functional as F, torchvision
    out = '/vol/features'; os.makedirs(out, exist_ok=True)
    if backbone == 'clip':
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        forward = model.encode_image
        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    elif backbone == 'resnet50':  # supervised ImageNet ResNet-50, penultimate (avgpool) features
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        forward = model
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:  # DINOv2 ViT-S/14, self-supervised, no text supervision; CLS token
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        forward = model
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    model = model.cuda().eval().half()
    mean = torch.tensor(mean, device='cuda').view(1, 3, 1, 1)
    std = torch.tensor(std, device='cuda').view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(images):
        feats = []
        for start in range(0, len(images), 1000):
            x = torch.from_numpy(images[start:start+1000]).cuda().permute(0, 3, 1, 2).float()/255.
            x = F.interpolate(x, size=224, mode='bicubic', align_corners=False).clamp(0, 1)
            z = forward(((x-mean)/std).half()).float()
            feats.append(F.normalize(z, dim=-1).cpu().numpy().astype(np.float16))
        return np.concatenate(feats)

    prefix = '' if backbone == 'clip' else f'{backbone}_'
    for train in (True, False):
        name = f"{out}/{prefix}{dataset}_{'train' if train else 'test'}.npz"
        if os.path.exists(name):
            continue
        if dataset == 'cifar10_1':
            # CIFAR-10.1 v6 (Recht et al. 2019): a natural re-collection of CIFAR-10, test only.
            if train:
                continue
            import urllib.request
            base = 'https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_'
            for part in ('data', 'labels'):
                urllib.request.urlretrieve(base+part+'.npy', f'/tmp/c101_{part}.npy')
            images = np.load('/tmp/c101_data.npy')
            labels = np.load('/tmp/c101_labels.npy').astype(np.int16)
            np.savez(name, X=embed(images), y=labels)
            print('saved', name, flush=True)
            volume.commit()
            continue
        if dataset == 'cifar10':
            ds = torchvision.datasets.CIFAR10('/tmp/data', train=train, download=True)
            extra = {}
        else:
            ds = torchvision.datasets.CIFAR100('/tmp/data', train=train, download=True)
            # torchvision exposes fine labels; derive coarse ones from the standard mapping file.
            import pickle
            meta = pickle.load(open(f"/tmp/data/cifar-100-python/{'train' if train else 'test'}", 'rb'),
                               encoding='latin1')
            extra = dict(y_coarse=np.asarray(meta['coarse_labels'], dtype=np.int16))
        np.savez(name, X=embed(ds.data), y=np.asarray(ds.targets, dtype=np.int16), **extra)
        print('saved', name, flush=True)
        volume.commit()


@app.local_entrypoint()
def main():
    extract.remote()


@app.local_entrypoint()
def resnet():
    extract_dataset.remote('resnet50', 'cifar10')


@app.local_entrypoint()
def natural():
    extract_dataset.remote('clip', 'cifar10_1')


@app.local_entrypoint()
def more():
    jobs = [('dinov2', 'cifar10'), ('clip', 'cifar100'), ('dinov2', 'cifar100')]
    list(extract_dataset.starmap(jobs))
