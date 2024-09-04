library(dbscan)
library("parallel")
library("reticulate")
library("devtools")
working_directory <- (strsplit(getwd(), "/conditional_simulation")[[1]])[1]
spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
devtools::install(spatialextremes_directory)

n <- 32
s1 <- s2 <- seq(-10, 10, length.out = n)
k <- 5
spatial_grid <- expand.grid(s1 = s1, 
                  s2 = s2)
#dim(spatial_grid) <- c(n**2, 2)
#dist_matrix <- as.matrix(dist, 2)(spatial_grid, method = "euclidean"))
knn <- kNN(spatial_grid, k = k)
id_matrix <- as.matrix(knn$id)

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


MCMC_interpolation_per_pixel <- function(observed_spatial_grid, observations, k, key_location, cov_mod, nugget, range, smooth, nrep)
{
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

MCMC_interpolation <- function(n, unobserved_indices, observations, k, cov_mod, nugget, range, smooth, nrep)
{
    unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
    observed_indices <- (1:n**2)[-unobserved_indices]
    observed_spatial_grid <- spatial_grid[observed_indices,]
    m <- dim(unobserved_spatial_grid)[1]
    condsim <- array(0, dim = c(m, nrep))
    for (i in 1:m)
    {
        key_location <- unobserved_spatial_grid[i,]
        condsim[i,] <- MCMC_interpolation_per_pixel(observed_spatial_grid, observations, k, key_location, cov_mod, nugget, range, smooth, nrep)
    }
    return(condsim)
}


np <- import("numpy")

obsn <- 512
seed_value <- 34234
set.seed(seed_value)
n <- 32
s1 <- s2 <- seq(-10, 10, length.out = n)
s <- cbind(s1, s2)
spatial_grid <- expand.grid(s1 = s1, 
                  s2 = s2)
range <- 3
smooth <- 1.6
nugget <- 0
cov_mod <- "powexp"
k <- 5
observations <- SpatialExtremes::rmaxstab(1, coord = s, cov.mod = "powexp",  nugget = nugget, range = range,
smooth = smooth, grid = TRUE)
dim(observations) <- c(n**2)
observed_indices <- sort(sample((n**2), obsn, replace = FALSE))
unobserved_indices <- (1:n**2)[-observed_indices]
mask <- produce_mask(observed_indices, n)
nrep <- 4000
imcmc <- MCMC_interpolation(n, unobserved_indices, observations, k, cov_mod, nugget, range, smooth, nrep)
np$save("data/powexp/MCMC_interpolation/ref_image1/conditional_simulations_neighbors5_powexp_range_3_smooth_1.6_4000.npy", imcmc)
np$save("data/powexp/MCMC_interpolation/ref_image1/observed_simulation_powexp_range_3_smooth_1.6.npy", observations)
np$save("data/powexp/MCMC_interpolation/ref_image1/mask.npy", mask)