library(dbscan)
library("parallel")
library("reticulate")
library("devtools")
library("SpatialExtremes")


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

MCMC_interpolation <- function(n, unobserved_indices, observations, k, cov_mod, nugget, range,
                               smooth, nrep, spatial_grid)
{
    unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
    observed_indices <- (1:n**2)[-unobserved_indices]
    observed_spatial_grid <- spatial_grid[observed_indices,]
    m <- dim(unobserved_spatial_grid)[1]
    condsim <- array(0, dim = c(m, nrep))
    for (i in 1:m)
    {
        print(i)
        key_location <- as.array(c(unobserved_spatial_grid[i,1], unobserved_spatial_grid[i,2]))
        dim(key_location) <- c(1,2)
        condsim[i,] <- MCMC_interpolation_per_pixel(observed_spatial_grid, observations, k, key_location, cov_mod, nugget, range, smooth, nrep)
    }
    return(condsim)
}