

produce_mask <- function(observed_indices, n)
{
  mask <- array(0, dim = c((n**2)))
  mask[observed_indices] <- rep(1, length(observed_indices))
  return(mask)
}

produce_random_mask <- function(m, n)
{
  
  observed_indices <- samples(1:(n**2), m)
  mask <- produce_mask(observed_indices, n)
}

joint_lcs <- function(mask_file_name, ref_image_name, n, nrep) 
{
  s1 <- s2 <- seq(-10, 10, length.out = n)
  s <- cbind(s1, s2)
  spatial_grid <- expand.grid(s1 = s1, 
                              s2 = s2)
  
  mask <- np$load(mask_file_name)
  ref_image <- exp(np$load(ref_image_name))
  ref_image <- flatten_matrix(ref_image, n)
  mask <- flatten_matrix(mask, n)
  observed_indices <- (1:n**2)[mask == 1]
  observed_spatial_grid <- spatial_grid[observed_indices,]
  observations <- ref_image[observed_indices]
  unobserved_indices <- (1:n**2)[-observed_indices]
  unobserved_observations <- ref_image[unobserved_indices]
  unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
  output <- SpatialExtremes::condrmaxstab(nrep, coord = unobserved_spatial_grid,
                                          cond.coord = observations,
                                          cond.data = observed_spatial_grid,
                                          cov.mod = cov_mod, 
                                          nugget = nugget, 
                                          range = range,
                                          smooth = smooth)
  condsim <- output$sim
  return(condsim)
}