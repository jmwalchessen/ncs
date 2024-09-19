library(SpatialExtremes)
library(reticulate)

produce_mask <- function(observed_indices, n)
{
    mask <- array(0, dim = c((n**2)))
    mask[observed_indices] <- rep(1, length(observed_indices))
    return(mask)
}

concatenate_conditional_simulations <- function(condsim, nrep, ref_image, mask)
{
    #make sure mask and ref_image are all flat
    c(mask) <- c(n**2)
    c(ref_image) <- c(n**2)
    condsim[,mask == 1] <- ref_image[mask == 1]
}

produce_validation_data <- function(model_name, ref_image_name, n, nrep, range, smooth, cov_mod, condsim_file)
{
    s1 <- seq(-10,10, length.out = n)
    s2 <- seq(-10,10, length.out = n)
    s <- cbind(s1, s2)
    spatial_grid <- expand.grid(s1 = s1, 
                  s2 = s2)
    np <- import("numpy")
    folder_name <- paste(paste("data", model_name, sep = "/"), ref_image_name, sep = "/")
    ref_image <- np$load((paste(folder_name, "ref_image.npy", sep = "/")))
    mask <- np$load((paste(folder_name, "mask.npy", sep = "/")))
    dim(ref_image) <- c(n**2)
    dim(mask) <- c(n**2)
    observed_indices <- (1:(n**2))[mask == 1]
    cond_data <- ref_image[observed_indices]
    cond_coord <- as.matrix(spatial_grid[observed_indices,])
    coord <- as.matrix(spatial_grid[-observed_indices,])

    cond <- condrmaxstab(k = nrep, coord = coord, cond.coord = cond_coord, cond.data = cond_data, range = range,
                         smooth = smooth, nugget = .0001, cov.mod = cov_mod)
    condsim <- cond$sim
    condsim_file <- paste(paste(folder_name, "mcmc/approximate_conditional", sep = "/"), condsim_file, sep = "/")
    np$save(condsim_file, condsim)
}




model_name <- "model3"
ref_image_name <- "ref_image3"
n <- 32
nrep <- 4000
range <- 12
smooth <- 1
cov_mod <- "brown"
condsim_file <- paste("approximate_conditional_simulation_range", as.character(range), sep = "_")
condsim_file <- paste(paste(paste(paste(condsim_file, "smooth", sep = "_"), as.character(smooth), sep = "_"), as.character(nrep), sep = "_"), "npy", sep = ".")
produce_validation_data(model_name, ref_image_name, n, nrep, range, smooth, cov_mod, condsim_file)