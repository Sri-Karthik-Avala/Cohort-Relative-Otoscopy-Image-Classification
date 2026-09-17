__author__ = "Karthik"

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 1234


def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def find_data():
    here = os.getcwd()
    try:
        sdir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        sdir = here
    bases = []
    for root in [here, sdir, os.path.dirname(here), os.path.dirname(sdir)]:
        for sub in ["", "public", "dataset/public", "dataset", "input", "data",
                    "input/public", "data/public"]:
            bases.append(os.path.join(root, sub) if sub else root)
    seen = set()
    bases = [b for b in bases if not (b in seen or seen.add(b))]

    train_csv = test_csv = None
    for b in bases:
        tc = os.path.join(b, "train.csv")
        te = os.path.join(b, "test.csv")
        if os.path.exists(tc) and os.path.exists(te):
            train_csv, test_csv = tc, te
            break
    if train_csv is None:
        for b in bases:
            hits = glob.glob(os.path.join(b, "**", "train.csv"), recursive=True)
            for h in hits:
                if os.path.exists(os.path.join(os.path.dirname(h), "test.csv")):
                    train_csv = h
                    test_csv = os.path.join(os.path.dirname(h), "test.csv")
                    break
            if train_csv:
                break
    if train_csv is None:
        raise FileNotFoundError("train.csv / test.csv not found")

    csv_dir = os.path.dirname(train_csv)
    img_dir = _first_existing([
        os.path.join(csv_dir, "images"),
        os.path.join(csv_dir, "public", "images"),
        os.path.join(os.path.dirname(csv_dir), "images"),
        os.path.join(csv_dir, "train"),
    ])
    if img_dir is None:
        for b in [csv_dir, os.path.dirname(csv_dir)] + bases:
            for d in glob.glob(os.path.join(b, "**", "images"), recursive=True):
                if os.path.isdir(d):
                    img_dir = d
                    break
            if img_dir:
                break
    if img_dir is None:
        cand = {}
        for b in [csv_dir, os.path.dirname(csv_dir)]:
            for d, _, files in os.walk(b):
                n = sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg", ".png")))
                if n > 50:
                    cand[d] = n
        if cand:
            img_dir = max(cand, key=cand.get)
    if img_dir is None:
        raise FileNotFoundError("images directory not found")

    out_dir = os.path.join(here, "working")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "submission.csv")
    return img_dir, train_csv, test_csv, out_path


def resolve_image_path(img_dir, iid):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG", ""):
        p = os.path.join(img_dir, iid + ext)
        if os.path.exists(p):
            return p
    return os.path.join(img_dir, iid + ".jpg")


def load_base(img_dir, iid, size):
    from PIL import Image
    try:
        im = Image.open(resolve_image_path(img_dir, iid)).convert("RGB")
        im = im.resize((size, size), Image.BICUBIC)
        return np.asarray(im, dtype=np.uint8)
    except Exception:
        return np.full((size, size, 3), 128, dtype=np.uint8)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_view(base, vidx, rng):
    from PIL import Image
    h = base.shape[0]
    if vidx == 0:
        arr = base
    elif vidx == 1:
        arr = base[:, ::-1]
    elif vidx == 2:
        arr = base[::-1, :]
    elif vidx == 3:
        arr = base[::-1, ::-1]
    else:
        scale = rng.uniform(0.65, 1.0)
        cs = max(8, int(round(h * np.sqrt(scale))))
        y0 = rng.randint(0, h - cs + 1)
        x0 = rng.randint(0, h - cs + 1)
        crop = base[y0:y0 + cs, x0:x0 + cs]
        crop = np.asarray(Image.fromarray(crop).resize((h, h), Image.BICUBIC))
        arr = crop[:, ::-1] if rng.random() < 0.5 else crop
    arr = np.ascontiguousarray(arr).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)


def strip_head(model, name):
    import torch.nn as nn
    if "convnext" in name:
        model.classifier[2] = nn.Identity()
    elif "efficientnet" in name:
        model.classifier = nn.Identity()
    elif "densenet" in name:
        model.classifier = nn.Identity()
    elif "resnet" in name:
        model.fc = nn.Identity()
    elif "mobilenet" in name:
        model.classifier = nn.Identity()
    elif "regnet" in name:
        model.fc = nn.Identity()
    return model


def backbone_specs(cuda):
    import torchvision.models as M
    s = []
    if cuda:
        s.append(("convnext_tiny", lambda: M.convnext_tiny(weights=M.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)))
        s.append(("efficientnet_b0", lambda: M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1)))
        s.append(("efficientnet_v2_s", lambda: M.efficientnet_v2_s(weights=M.EfficientNet_V2_S_Weights.IMAGENET1K_V1)))
        s.append(("densenet201", lambda: M.densenet201(weights=M.DenseNet201_Weights.IMAGENET1K_V1)))
    else:
        s.append(("convnext_tiny", lambda: M.convnext_tiny(weights=M.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)))
        s.append(("efficientnet_b0", lambda: M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1)))
        s.append(("densenet201", lambda: M.densenet201(weights=M.DenseNet201_Weights.IMAGENET1K_V1)))
    return s


def extract_one(model, bases, n_views, device, batch, seed):
    import torch
    N = len(bases)
    out = None
    rng = np.random.RandomState(seed)
    model = model.eval().to(device)
    with torch.no_grad():
        for v in range(n_views):
            acc = []
            for i in range(0, N, batch):
                chunk = bases[i:i + batch]
                ten = np.stack([make_view(b, v, rng) for b in chunk])
                x = torch.from_numpy(ten).float().to(device)
                f = model(x).float().cpu().numpy()
                if f.ndim > 2:
                    f = f.reshape(f.shape[0], -1)
                acc.append(f)
            fv = np.concatenate(acc, 0)
            out = fv if out is None else out + fv
    return out / float(n_views)


def extract_deep(ids, img_dir):
    spaces = {}
    try:
        import torch
    except Exception:
        return spaces
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    size = 224
    n_views = 8 if cuda else 4
    batch = 96 if cuda else 24
    bases = [load_base(img_dir, i, size) for i in ids]
    for name, ctor in backbone_specs(cuda):
        try:
            model = strip_head(ctor(), name)
            emb = extract_one(model, bases, n_views, device, batch, SEED)
            spaces[name] = emb.astype(np.float32)
            del model
            if cuda:
                torch.cuda.empty_cache()
            print("  feat[%s] %s" % (name, str(emb.shape)))
        except Exception as e:
            print("  skip %s (%s)" % (name, str(e)[:60]))
    return spaces


def extract_classical(ids, img_dir):
    size = 64
    feats = []
    for iid in ids:
        b = load_base(img_dir, iid, size).astype(np.float32) / 255.0
        r, g, bl = b[..., 0], b[..., 1], b[..., 2]
        mx = b.max(2); mn = b.min(2); v = mx; s = (mx - mn) / (mx + 1e-6)
        gray = 0.299 * r + 0.587 * g + 0.114 * bl
        gy, gx = np.gradient(gray)
        mag = np.sqrt(gx * gx + gy * gy)
        ori = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
        vec = []
        for ch in [r, g, bl, s, v, gray]:
            hist, _ = np.histogram(ch, bins=8, range=(0, 1), density=True)
            vec.append(hist)
        oh, _ = np.histogram(ori, bins=9, range=(0, 1), weights=mag, density=True)
        vec.append(oh)
        gh, _ = np.histogram(mag, bins=8, range=(0, mag.max() + 1e-6), density=True)
        vec.append(gh)
        grid = []
        gs = size // 4
        for yy in range(4):
            for xx in range(4):
                tile = b[yy * gs:(yy + 1) * gs, xx * gs:(xx + 1) * gs]
                grid.append(tile.reshape(-1, 3).mean(0))
                grid.append([tile.reshape(-1, 3).std()])
        vec.append(np.concatenate([np.atleast_1d(x).ravel() for x in grid]))
        f = np.fft.fftshift(np.abs(np.fft.fft2(gray)))
        cy = cx = size // 2
        yy, xx = np.ogrid[:size, :size]
        rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        bands = []
        for k in range(6):
            mk = (rad >= k * size / 12.0) & (rad < (k + 1) * size / 12.0)
            bands.append(np.log1p(f[mk].mean() if mk.any() else 0.0))
        vec.append(np.array(bands))
        feats.append(np.concatenate([np.atleast_1d(x).ravel() for x in vec]).astype(np.float32))
    F = np.stack(feats)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    mu = F.mean(0, keepdims=True); sd = F.std(0, keepdims=True) + 1e-6
    return (F - mu) / sd


def l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def cohort_anomaly(X, df):
    score = np.zeros(len(df), dtype=np.float64)
    for g, idx in df.groupby("group_id").indices.items():
        idx = np.asarray(idx)
        Xi = X[idx]
        n = len(idx)
        S = Xi @ Xi.T
        D = 1.0 - S
        np.fill_diagonal(D, np.nan)
        mean_d = np.nanmean(D, axis=1)
        csum = Xi.sum(0)
        loo = l2norm((csum[None, :] - Xi) / max(n - 1, 1))
        loo_d = 1.0 - np.sum(Xi * loo, axis=1)
        raw = 0.5 * mean_d + 0.5 * loo_d
        med = np.median(raw)
        mad = np.median(np.abs(raw - med)) + 1e-6
        score[idx] = (raw - med) / (1.4826 * mad)
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def matthews(true, pred):
    true = np.asarray(true).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(np.sum((pred == 1) & (true == 1)))
    tn = int(np.sum((pred == 0) & (true == 0)))
    fp = int(np.sum((pred == 1) & (true == 0)))
    fn = int(np.sum((pred == 0) & (true == 1)))
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def facets(pred, true, groups):
    pred = np.asarray(pred).astype(int)
    true = np.asarray(true).astype(int)
    m = max(0.0, min(1.0, matthews(true, pred)))
    fl = set(np.where(pred == 1)[0])
    t = set(np.where(true == 1)[0])
    u = fl | t
    j = 1.0 if not u else len(fl & t) / len(u)
    err = (pred != true).astype(int)
    d = pd.DataFrame({"g": groups, "e": err})
    c = float(np.mean(np.exp(-1.1 * d.groupby("g").e.sum().values)))
    base = (m * j * c) ** (1.0 / 3.0) if min(m, j, c) > 0 else 0.0
    return m, j, c, base


def apply_threshold(score, groups, thr, cap):
    pred = (score > thr).astype(int)
    d = pd.DataFrame({"g": groups})
    for g, idx in d.groupby("g").indices.items():
        idx = np.asarray(idx)
        fl = idx[pred[idx] == 1]
        if len(fl) > cap:
            keep = fl[np.argsort(-score[fl])[:cap]]
            pred[idx] = 0
            pred[keep] = 1
    return pred


def optimize_threshold(score, y, groups, cap):
    best = (-1.0, float(np.quantile(score, 0.85)))
    grid = np.unique(np.quantile(score, np.linspace(0.55, 0.98, 220)))
    for thr in grid:
        b = facets(apply_threshold(score, groups, thr, cap), y, groups)[3]
        if b > best[0]:
            best = (b, thr)
    return best[1], best[0]


def main():
    seed_everything()
    img_dir, train_csv, test_csv, out_path = find_data()
    print("images:", img_dir)
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    full = pd.concat([tr[["image_id", "group_id"]], te[["image_id", "group_id"]]],
                     ignore_index=True)
    ids = list(full.image_id)
    ntr = len(tr)

    print("extracting features ...")
    spaces = extract_deep(ids, img_dir)
    spaces["classical"] = extract_classical(ids, img_dir)
    use = list(spaces)
    print("spaces:", list(spaces.keys()), "| consensus over:", use)

    scores = [cohort_anomaly(l2norm(spaces[n].astype(np.float32)), full) for n in use]
    consensus = np.mean(scores, axis=0)
    cons_tr, cons_te = consensus[:ntr], consensus[ntr:]

    ytr = tr.is_intruder.values.astype(int)
    gtr = tr.group_id.values
    gte = te.group_id.values
    cap = max(int(tr.groupby("group_id").is_intruder.sum().max()), 1)

    thr, base = optimize_threshold(cons_tr, ytr, gtr, cap)
    m, j, c, _ = facets(apply_threshold(cons_tr, gtr, thr, cap), ytr, gtr)
    print("train base=%.4f m=%.3f j=%.3f c=%.3f thr=%.4f cap=%d" % (base, m, j, c, thr, cap))

    pred = apply_threshold(cons_te, gte, thr, cap)
    if pred.sum() == 0:
        rate = max(float(ytr.mean()), 1.0 / len(ytr))
        thr2 = float(np.quantile(cons_te, 1.0 - rate))
        pred = apply_threshold(cons_te, gte, thr2, cap)

    sub = pd.DataFrame({"image_id": te.image_id.values, "is_intruder": pred.astype(int)})
    sub = sub.drop_duplicates("image_id").set_index("image_id").loc[te.image_id.values].reset_index()
    sub.to_csv(out_path, index=False)
    print("wrote %s rows=%d positives=%d (%.1f%%)" %
          (out_path, len(sub), int(sub.is_intruder.sum()), 100.0 * sub.is_intruder.mean()))


if __name__ == "__main__":
    main()
