import os, json, hashlib, numpy as np, sys
root = os.path.abspath(os.path.dirname(__file__) + '/../..')
dataset = sys.argv[1]
src = os.path.join(root, "output_deepmvi", dataset, "imputed.npy")
dst_dir = os.path.join(root, "output_imputegap", dataset)
os.makedirs(dst_dir, exist_ok=True)
arr = np.load(src)
np.save(os.path.join(dst_dir, "imputed.npy"), arr)
sha1 = hashlib.sha1(open(os.path.join(dst_dir, "imputed.npy"), "rb").read()).hexdigest()
with open(os.path.join(dst_dir, "sha1.json"), "w") as f:
    json.dump({"sha1": sha1}, f, indent=4)
meta = {
    "dataset": dataset,
    "n_samples": int(arr.shape[0]),
    "n_features": int(arr.shape[1]),
    "imputer": "DeepMVI",
    "version": "1.0",
}
with open(os.path.join(dst_dir, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=4)
print("Exported:", dst_dir)
