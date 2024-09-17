library(SpatialExtremes)
library(reticulate)

produce_mask <- function(observed_indices, n)
{
    mask <- array(0, dim = c((n**2)))
    mask[observed_indices] <- rep(1, length(observed_indices))
    return(mask)
}


produce_validation_data <- function(

n <- 25
s1 <- seq(-10,10, length.out = n)
s2 <- seq(-10,10, length.out = n)
s <- cbind(s1, s2)
spatial_grid <- expand.grid(s1 = s1, 
                  s2 = s2)
cov_mod <- "brown"
observations <- SpatialExtremes::rmaxstab(n = 1, coord = s, cov.mod = cov_mod, nugget = nugget,
                                              range = range, smooth = smooth, grid = TRUE))

