library("reticulate")
library(devtools)
working_directory <- (strsplit(getwd(), "/sde_diffusion")[[1]])[1]
spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
devtools::install(spatialextremes_directory)

produce_random_mask <- function(mask_file_name, obsn, n)
{
    observed_indices <- sort(sample(1:(n**2), obsn, replace = FALSE))
    mask <- produce_mask(observed_indices, n)
    np <- import("numpy")
    np$save(mask_file_name, mask) 
}

produce_mask <- function(observed_indices, n)
{
    mask <- array(0, dim = c((n**2)))
    mask[observed_indices] <- rep(1, length(observed_indices))
    dim(mask) <- c(n,n)
    return(mask)
}

produce_reference_images <- function(nsim, range, smooth, cov_mod, ref_image_file)
{
    n <- 32
    s1 <- seq(-10,10, length.out = n)
    s2 <- seq(-10,10, length.out = n)
    s <- cbind(s1, s2)
    ref_images <- SpatialExtremes::rmaxstab(nsim, coord = s, cov.mod = cov_mod, grid = TRUE, range = range, smooth = smooth)
    dim(ref_images) <- c(nsim, n, n)
    np <- import("numpy")
    np$save(ref_image_file, ref_images)  
}

nsim <- 1000
range <- 2
smooth <- 1
cov_mod <- "brown"
ref_image_file <- "data/mcmc/mask1/reference_images_range_2_smooth_1.npy"
mask_file_name <- "data/mcmc/mask1/mask.npy"
obsn <- 10
n <- 32
produce_reference_images(nsim, range, smooth, cov_mod, ref_image_file)