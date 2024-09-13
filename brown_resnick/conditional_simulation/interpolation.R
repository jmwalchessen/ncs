library(dbscan)
library("parallel")
library("reticulate")
library("devtools")
#working_directory <- (strsplit(getwd(), "/conditional_simulation")[[1]])[1]
#spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
#devtools::install(spatialextremes_directory)
library(SpatialExtremes)
library(R.utils)

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
    dim(mask) <- c(n,n)
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


#observed_spatial_grid, observations, k, key_location, cov_mod, nugget, range, smooth, nrep
MCMC_interpolation_per_pixel_interrupted <- function(argsList)
{
    observed_spatial_grid <- argsList$observed_spatial_grid
    observations <- argsList$observations
    k <- argsList$k
    key_location <- argsList$key_location
    cov_mod <- argsList$cov_mod
    nugget <- argsList$nugget
    range <- argsList$range
    smooth <- argsList$smooth
    nrep <- argsList$nrep


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
    condsim <- array(NA, dim = c(m, nrep))
    for (i in 1:m)
    {
        print(i)
        key_location <- unobserved_spatial_grid[i,]
        x <- MCMC_interpolation_per_pixel(observed_spatial_grid, observations, k, key_location, cov_mod, nugget, range, smooth, nrep)
        condsim[i,] <- x
    }
    return(condsim)
}



try_with_time_limit <- function(observed_spatial_grid, observations, k, key_location, cov_mod, nugget, range, smooth, nrep, cpu = Inf, elapsed = Inf)
{

  y <- try({setTimeLimit(cpu, elapsed);
  MCMC_interpolation_per_pixel(observed_spatial_grid, observations, k, key_location, cov_mod, nugget, range, smooth, nrep)}, silent = TRUE) 
  if(inherits(y, "try-error")) NA else y 
}


interruptor <- function(FUN,args, time.limit, ALTFUN){

  results <- 
    tryCatch({
      withTimeout({FUN(args)}, timeout=time.limit)
    }, error = function(e){
      if(grepl("reached elapsed time limit",e$message))
        ALTFUN(args) else
          paste(e$message,"EXTRACTERROR")
      })

  #if(grepl("EXTRACTERROR",results)){
    #print(gsub("EXTRACTERROR","",results))
    #results <- array(1,args$nrep)
  #} 

  return(results)
} 

alternative_MCMC_interpolation_per_pixel <- function(observed_spatial_grid, observations, k, key_location, cov_mod, nugget, range, smooth, nrep)
{
    return(array(NA, dim = c(1, nrep)))
}

afunc <- function(x,y)
{
    return(x+y)
}

func <- function(l)
{
    return(l$x-l$y)
}

funcinterupt <- function(l)
{
    interruptor(func, l, 5, afunc)
}


MCMC_interpolation_interupted_partial <- function(n, unobserved_indices, observations, k, cov_mod, nugget, range, smooth, nrep, unobserved_start, unobserved_end)
{
    unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
    observed_indices <- (1:n**2)[-unobserved_indices]
    observed_spatial_grid <- spatial_grid[observed_indices,]
    condsim <- array(NA, dim = c((unobserved_end - unobserved_start+1), nrep))
    print(dim(condsim))
    for (i in unobserved_start:unobserved_end)
    {
        print(i)
        key_location <- unobserved_spatial_grid[i,]
        x <- interruptor(FUN = MCMC_interpolation_per_pixel_interrupted, args = list(observed_spatial_grid = observed_spatial_grid, observations = observations,k = k,
                         key_location = key_location, cov_mod = cov_mod, nugget = nugget, range = range, smooth = smooth, nrep = nrep), time.limit = 5, ALTFUN = alternative_MCMC_interpolation_per_pixel)
        print(x)
        condsim[(i-unobserved_start+1),] <- x
    }
    return(condsim)
}


MCMC_interpolation_interupted_full <- function(n, unobserved_indices, observations, k, cov_mod, nugget, range, smooth, nrep,
                                               gap)
{
    unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
    observed_indices <- (1:n**2)[-unobserved_indices]
    observed_spatial_grid <- spatial_grid[observed_indices,]
    m <- dim(unobserved_spatial_grid)[1]
    condsim <- array(NA, dim = c(m, nrep))
    partial <- as.integer(m/gap)
    staggering <- seq(1,m,gap)
    for (i in 1:partial)
    {
        unobserved_start <- staggering[i]
        unobserved_end <- staggering[i+1]
        print(unobserved_start)
        print(unobserved_end)
        condsim[unobserved_start:unobserved_end,] <- MCMC_interpolation_interupted_partial(n, unobserved_indices, observations, k, cov_mod, nugget,
                                                                         range, smooth, nrep, unobserved_start, unobserved_end)
    }
    return(condsim)
}


#np <- import("numpy")

#obsn <- 300
#seed_value <- 34234
#set.seed(seed_value)
#n <- 25
#s1 <- s2 <- seq(-10, 10, length.out = n)
#s <- cbind(s1, s2)
#spatial_grid <- expand.grid(s1 = s1, 
#                  s2 = s2)
#range <- 1.6
#smooth <- 1.6
#nugget <- 0
#cov_mod <- "brown"
#k <- 5
#observations <- SpatialExtremes::rmaxstab(1, coord = s, cov.mod =  cov_mod, 
#                                          nugget = nugget, range = range,
#                                          smooth = smooth, grid = TRUE)
#dim(observations) <- c(n**2)
#observed_indices <- sort(sample((n**2), obsn, replace = FALSE))
#unobserved_indices <- (1:n**2)[-observed_indices]
#mask <- produce_mask(observed_indices, n)
#nrep <- 5
#gap <- 20
#imcmc <- MCMC_interpolation_interupted_full(n, unobserved_indices, observations, k, cov_mod, nugget, range, smooth, nrep, gap)


#np$save("data/brown/MCMC_interpolation/ref_image1/conditional_simulations_neighbors5_brown_range_1.6_smooth_1.6_4000_25_by_25.npy", imcmc)
#np$save("data/brown/MCMC_interpolation/ref_image1/observed_simulation_brown_range_1.6_smooth_1.6_25_by_25.npy", observations)
#np$save("data/brown/MCMC_interpolation/ref_image1/mask.npy", mask)

np <- import("numpy")
n <- 25
obsn <- 624
observed_indices <- sort(sample((n**2), obsn, replace = FALSE))
mask <- produce_mask(observed_indices, n)
np$save("brown_resnick/data/25_by_25/mask.npy", mask)