library(SpatialExtremes)
library(reticulate)
library(ggplot2)

produce_mask <- function(observed_indices, n)
{
    mask <- array(0, dim = c((n**2)))
    mask[observed_indices] <- rep(1, length(observed_indices))
    return(mask)
}

n.sim <- 20
n.cond <- 5
range <- 20
smooth <- 1
nugget <- 0
n <- 25
s1 <- seq(-10,10, length.out = n)
s2 <- seq(-10,10, length.out = n)
s <- cbind(s1, s2)
spatial_grid <- expand.grid(s1 = s1, 
                  s2 = s2)
cov_mod <- "brown"
observations <- SpatialExtremes::rmaxstab(n = 1, coord = s, cov.mod = cov_mod, nugget = nugget,
                                              range = range, smooth = smooth, grid = TRUE)

observed_indices <- sort(sample(1:(n**2), n.cond, replace = FALSE))
mask <- produce_mask(observed_indices, n)
cond_data <- observations[observed_indices]
cond_coord <- as.matrix(spatial_grid[observed_indices,])
coord <- as.matrix(spatial_grid[-observed_indices,])

cond <- condrmaxstab(n.sim, coord, cond.coord = cond_coord, cond.data = cond_data, range = range,
smooth = smooth, nugget = .0001, cov.mod = "brown")
condsim <- cond$sim

np <- import("numpy")
condsim_file <- "data/25_by_25/all/ref_image13/conditional_simulations_range_20_smooth_1_observed_5_20.npy"
refimagename <- "data/25_by_25/all/ref_image13/ref_image.npy"
maskname <- "data/25_by_25/all/ref_image13/mask.npy"
np$save(condsim_file, condsim)
np$save(refimagename, observations)
np$save(maskname, mask)

#spatial_grid$z <- NA
#spatial_grid$z[-observed_indices] <- condsim
#ggplot(spatial_grid) + geom_tile(aes(s1,s2, fill = log(z))) + 
                       scale_colour_distiller(palette = "Spectral"); 