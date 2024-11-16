library(dbscan)
library("parallel")
library("reticulate")
library(SpatialExtremes)
library(R.utils)

args <- commandArgs(trailingOnly=TRUE)
range <- as.numeric(args[1])
smooth <- as.numeric(args[2])
nugget <- as.numeric(args[3])
ref_image_name <- as.character(args[4])
mask_file_name <- as.character(args[5])
condsim_file_name <- as.character(args[6])
cov_mod <- as.character(args[7])
neighbors <- as.numeric(args[8])
n <- as.numeric(args[9])
nrep <- as.numeric(args[10])
missing_index_start <- as.numeric(args[11])
missing_index_end <- as.numeric(args[12])


produce_mask <- function(observed_indices, n)
{
    mask <- array(0, dim = c((n**2)))
    mask[observed_indices] <- rep(1, length(observed_indices))
    return(mask)
}

located_neighboring_pixels <- function(observed_spatial_grid, k, key_location)
{
    knn <- kNN(observed_spatial_grid, k = k, query = key_location)
    id_matrix <- as.matrix(knn$id)
    return(id_matrix)
}

lcs_per_pixel <- function(observed_spatial_grid, observations, k, key_location,
                          cov_mod, nugget, range, smooth, nrep)
{
    print(key_location)
    id_matrix <- located_neighboring_pixels(observed_spatial_grid, k, key_location)
    cond_data <- observations[id_matrix]
    cond_coord <- observed_spatial_grid[id_matrix,]
    output <- SpatialExtremes::condrmaxstab(nrep, coord = key_location,
              cond.coord = cond_coord,
              cond.data = cond_data,
              cov.mod = cov_mod, 
              nugget = nugget, 
              range = range,
              smooth = smooth)
    condsim <- output$sim
}

produce_lcs_per_pixel_via_core <- function(n, range, smooth, nugget, cov_mod, seed1, seed2,
                                              neighbors, m, nrep, missing_index)
{
    s1 <- s2 <- seq(-10, 10, length.out = n)
    s <- cbind(s1, s2)
    spatial_grid <- expand.grid(s1 = s1, 
                  s2 = s2)
    set.seed(seed1)
    observations <- SpatialExtremes::rmaxstab(n = 1, coord = s, cov.mod = cov_mod,
                                              range = range, smooth = smooth, grid = TRUE)
    dim(observations) <- c(n**2)

    set.seed(seed2)
    observed_indices <- sort(sample((n**2), (n**2-m), replace = FALSE))
    observed_spatial_grid <- spatial_grid[observed_indices,]
    unobserved_indices <- (1:n**2)[-observed_indices]
    unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
    mask <- produce_mask(observed_indices, n)
    key_location <- unobserved_spatial_grid[missing_index,]
    condsim <- lcs_per_pixel(observed_spatial_grid, observations,
                             neighbors, key_location, cov_mod,
                             nugget, range, smooth, nrep)
    return(condsim)
}


produce_lcs_per_pixel_via_mask <- function(n, range, smooth, nugget, cov_mod, mask_file_name,
                                           ref_image_name, neighbors, nrep, missing_index)
{
    np <- import("numpy")
    s1 <- s2 <- seq(-10, 10, length.out = n)
    s <- cbind(s1, s2)
    spatial_grid <- expand.grid(s1 = s1, 
                  s2 = s2)

    mask <- np$load(mask_file_name)
    observations <- np$load(ref_image_name)
    dim(observations) <- c(n**2)
    dim(mask) <- c(n**2)
    observed_indices <- (1:n**2)[mask == 1]
    observed_spatial_grid <- spatial_grid[observed_indices,]
    observations <- observations[observed_indices]
    unobserved_indices <- (1:n**2)[-observed_indices]
    unobserved_observations <- observations[unobserved_indices]
    print(log(unobserved_observations[missing_index]))
    unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
    key_location <- unobserved_spatial_grid[missing_index,]
    condsim <- lcs_per_pixel(observed_spatial_grid, observations, neighbors,
                             key_location, cov_mod, nugget, range, smooth, nrep)
    return(condsim)
}



for(missing_index in missing_index_start:missing_index_end)
{
    y <- produce_lcs_per_pixel_via_mask(n, range, smooth, nugget, cov_mod, mask_file_name, ref_image_name,
                                                   neighbors, nrep, missing_index)
    current_condsim_file <- paste(paste(condsim_file_name, as.character(missing_index), sep = "_"), "npy", sep = ".")
    np <- import("numpy")
    np$save(current_condsim_file, y)
}
rm(list = ls())