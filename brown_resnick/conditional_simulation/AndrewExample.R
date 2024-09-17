library(SpatialExtremes)

s1 <- s2 <- seq(-4, 4, length.out = 64L)
sgrid <- expand.grid(s1 = s1, s2 = s2)

sgrid$y_true <- np$load(paste0("src/runs/Section5_2_SBNN-IP/output/true_field.npy"))
mean_sd <- np$load(paste0("src/intermediates/max_stable_data_Gumbel_mean_std.npy"))
obs$z_frechet <- (obs$z * mean_sd[2] + mean_sd[1]) %>% exp()




lscale <- 3
system.time({X <- lapply(1, function(i) 
        condrmaxstab(30, coord = sgrid[idx_pred_locs, 1:2] %>% as.matrix(),
            cond.coord = obs[idx_obs_locs ,1:2] %>% as.matrix(),
            cond.data = obs[idx_obs_locs, "z_frechet"],
            cov.mod = "powexp", 
            nugget = 0, range = lscale,
            smooth = 1.5))})