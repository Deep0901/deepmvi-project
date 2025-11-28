import numpy as np, os, sys
root="/home/deeps/deepmvi-seminar"
imp=os.path.join(root,"output_deepmvi","airq","imputed.npy")
gt_candidates=[os.path.join(root,"data","airq","X.npy"),
               os.path.join(root,"data","airq","airq.npy"),
               os.path.join(root,"data","airq","GT.npy")]
if not os.path.exists(imp):
    print("ERROR: imputed file not found:", imp); sys.exit(2)
imp_arr=np.load(imp)
gt=None
for p in gt_candidates:
    if os.path.exists(p):
        try:
            gt=np.load(p)
            print("Using GT:",p)
            break
        except Exception:
            pass
if gt is None:
    print("No ground truth found among:", gt_candidates); sys.exit(0)
if imp_arr.shape != gt.shape:
    print("Shape mismatch: imputed", imp_arr.shape, "gt", gt.shape); sys.exit(3)
mask = ~np.isnan(gt)
diff = (imp_arr - gt)[mask]
mae = float(np.mean(np.abs(diff)))
rmse = float((np.mean(diff**2))**0.5)
print(f"MAE={mae:.6e}, RMSE={rmse:.6e}, points={diff.size}")
with open("run_logs/airq_metrics.txt","w") as f:
    f.write(f"MAE={mae}\\nRMSE={rmse}\\npoints={diff.size}\\n")
print("Wrote run_logs/airq_metrics.txt")
