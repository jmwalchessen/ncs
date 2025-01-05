library("reticulate")
library(devtools)
library(Rlab)
working_directory <- (strsplit(getwd(), "/sde_diffusion")[[1]])[1]
spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
devtools::install(spatialextremes_directory)

produce_random_mask <- function(p, n)
{
    mask <- rbern(n**2, p)
    dim(mask) <- c(n,n)
    return(mask)
}

produce_random_masks <- function(nsim, p, n)
{
    masks <- array(0, dim = c(nsim,n,n))
    for(i in 1:nsim)
    {
        masks[i,,] <- produce_random_mask(p,n)
    }
    return(masks)
}

#produce one reference image and one mask (pair)
produce_reference_images_and_masks <- function(nsim, range, smooth, cov_mod, ref_image_file, mask_file, n, p)
{
    n <- 32
    s1 <- seq(-10,10, length.out = n)
    s2 <- seq(-10,10, length.out = n)
    s <- cbind(s1, s2)
    ref_images <- SpatialExtremes::rmaxstab(nsim, coord = s, cov.mod = cov_mod, grid = TRUE, range = range, smooth = smooth)
    masks <- produce_random_masks(nsim, p, n)
    dim(ref_images) <- c(nsim, n, n)
    np <- import("numpy")
    np$save(ref_image_file, ref_images)
    np$save(mask_file, masks)  
}

generate_pit_reference_images_and_masks <- function()
{
    ps <- c(.01,.05,.1,.25,.5)
    smooth <- 1.5
    range <- 3.
    nsim <- 4000
    cov_mod <- "brown"
    for(i in 1:length(ps))
    {
        p <- ps[i]
        ref_name <- paste("data/model4/random", as.character(p), sep = "")
        ref_image_file <- paste(paste(paste(ref_name, "reference_images_range_3_smooth_1.5_random", sep = "/"),
                                            as.character(p), sep = "_"), "_4000.npy", sep = "_")
        mask_file <- paste(ref_name, "mask.npy", sep = "/")
        obsn <- 10
        n <- 32
        produce_reference_images_and_masks(nsim, range, smooth, cov_mod, ref_image_file, mask_file, n, p)
    }
}

generate_pit_reference_images_and_masks()