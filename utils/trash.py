# I can't remember why i used this before and what i changed but it might be important so im keeping it here for now anyways
# 
# def train_unet(
#     train_loader,
#     val_loader,
#     epochs=8,
#     lr=1e-3,
#     lambda_holes=2.0,
#     beta_tv=0.01,
#     normalized=False,     # match your dataset's normalize flag
#     max_depth_m=80.0,
#     ckpt_path="unet_depth_inpaint_best.pt",
#     device=None,
# ):
#     device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"device:{device}")
#     model = DepthUNet(in_ch=1, base_ch=32).to(device)
#     opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
#     scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

#     best_rmse = float('inf')
#     max_val = 1.0 if normalized else max_depth_m

#     for ep in tqdm(range(1, epochs+1)):
#         model.train()
#         t0 = time.time()
#         running = {"l_all":0.0, "l_h":0.0, "l_tv":0.0}
#         nsteps = 0

#         for i, batch in enumerate(train_loader):
#             # print(f"starting batch...")
#             inp = batch["input_depth"].to(device)     # (B,1,H,W)
#             tgt = batch["target_depth"].to(device)
#             vm  = batch["valid_mask"].to(device)      # bool
#             hole_mask = (inp == 0.0) & vm             # supervise holes where GT exists

#             opt.zero_grad(set_to_none=True)
#             with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
#                 pred = model(inp)
#                 # Losses
#                 l_all  = masked_l1(pred, tgt, vm)
#                 l_hole = masked_l1(pred, tgt, hole_mask) if hole_mask.any() else pred.new_tensor(0.0)
#                 l_tv   = tv_smoothness(pred, focus_mask=hole_mask | vm) * beta_tv
#                 loss   = l_all + lambda_holes * l_hole + l_tv

#             scaler.scale(loss).backward()
#             scaler.step(opt)
#             scaler.update()

#             running["l_all"] += float(l_all.detach().cpu())
#             running["l_h"]   += float(l_hole.detach().cpu())
#             running["l_tv"]  += float(l_tv.detach().cpu())
#             nsteps += 1
#             # print(f"batch {i} of {len(train_loader)} complete.")

#         # Validate
#         val_metrics = evaluate(model, val_loader, device, normalized=normalized, max_depth_m=max_depth_m)

#         # Save best-by hole RMSE
#         if val_metrics["holes_rmse"] < best_rmse:
#             best_rmse = val_metrics["holes_rmse"]
#             torch.save({"model": model.state_dict(),
#                         "epoch": ep,
#                         "val": val_metrics,
#                         "cfg": {"lambda_holes":lambda_holes, "beta_tv":beta_tv, "normalized":normalized}},
#                        ckpt_path)

#         dt = time.time() - t0
#         print(f"[Epoch {ep}/{epochs}] {dt:.1f}s  "
#               f"train L_all={running['l_all']/nsteps:.4f}  "
#               f"L_h={running['l_h']/nsteps:.4f}  L_tv={running['l_tv']/nsteps:.4f}  |  "
#               f"VAL holes: MAE={val_metrics['holes_mae']:.3f} RMSE={val_metrics['holes_rmse']:.3f} PSNR={val_metrics['holes_psnr']:.2f}  "
#               f"| all: MAE={val_metrics['all_mae']:.3f} RMSE={val_metrics['all_rmse']:.3f} PSNR={val_metrics['all_psnr']:.2f}")

#     print(f"Best hole RMSE: {best_rmse:.3f} (ckpt saved to {ckpt_path})")
#     return model
