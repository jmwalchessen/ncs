library(parallel)
library(reticulate)
library(devtools)
working_directory <- (strsplit(getwd(), "/sde_diffusion")[[1]])[1]
spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
devtools::install(spatialextremes_directory)


produce_ref_image <- function(model_name, ref_image_name, cov_mod, range, smooth, nugget, n)
{
    s1 <- seq(-10,10, length.out = n)
    s2 <- seq(-10,10, length.out = n)
    s <- cbind(s1, s2)
    observations <- SpatialExtremes::rmaxstab(n = 1, coord = s, cov.mod = cov_mod, nugget = nugget,
                                              range = range, smooth = smooth, grid = TRUE)
    dim(observations) <- c(n,n)
    np <- import("numpy")
    ref_file_name <- paste(paste(paste("data", model_name, sep = "/"), ref_image_name, sep = "/"), "ref_image.npy", sep = "/")
    np$save(ref_file_name, observations)
}


produce_mask <- function(observed_indices, n)
{
    mask <- array(0, dim = c((n**2)))
    mask[observed_indices] <- rep(1, length(observed_indices))
    dim(mask) <- c(n,n)
    return(mask)
}

produce_random_mask <- function(model_name, ref_image_name, obsn, n)
{
    observed_indices <- sort(sample(1:(n**2), obsn, replace = FALSE))
    mask <- produce_mask(observed_indices, n)
    mask_name <- paste(paste(paste("data", model_name, sep = "/"), ref_image_name, sep = "/"), "mask.npy", sep = "/")
    np <- import("numpy")
    np$save(mask_name, mask) 
    
}

model_name <- "model3"
ref_image_name <- "ref_image3"
cov_mod = "brown"
range <- 12
smooth <- 1
nugget <- 0
obsn <- 5
n <- 32
produce_ref_image(model_name, ref_image_name, cov_mod, range, smooth, nugget, n)
produce_random_mask(model_name, ref_image_name, obsn, n)