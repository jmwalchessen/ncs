library(Rlab)
source("brown_resnick_data_generation.R")

number_of_replicates <- 4000

produce_mask <- function(n, p)
{
  mask <- rbern(n = (n**2), prob = p)
  dim(mask) <- c(n,n)
  return(mask)
}

produce_masks <- function(nrep, n, p)
{
  masks <- array(0, c(nrep,n,n))
  for(i in 1:nrep)
  {
    masks[i,,] <- produce_mask(n, p)
  }
  return(masks)
}

masked_true_brown_resnick_data_generation <- function(nrep, number_of_replicates_per_call, calls, range, smooth,
                                                      mask_file, masked_brfile, n, p)
{
  np <- import("numpy")
  n.size <- 1024
  nn <- sqrt(n.size)
  x <- y <- seq(-10, 10, length = nn)
  coord <- expand.grid(x, y)
  repnumberslist <- rep(number_of_replicates_per_call, calls)
  masks <- produce_masks(nrep,n,p)
  brimages <- simulate_data_across_cores(repnumberslist, nn, coord, range, smooth)
  dim(brimages) <- c(nrep,n,n)
  masked_brimages <- masks*brimages
  np$save(mask_file, masks)
  np$save(masked_brfile, masked_brimages)
}

nrep <- 4
number_of_replicates_per_call <- 2
calls <- 2
range <- 1.0
smooth <- 1.5
n <- 32
p <- .05
mask_file <- "data/ncs/model4/true_masks_range_1.0_smooth_1.5_4.npy"
masked_brfile <- "data/ncs/model4/true_masked_brown_resnick_range_1.0_smooth_1.5_4.npy"
masked_true_brown_resnick_data_generation(nrep, number_of_replicates_per_call, calls, range, smooth, mask_file,
                                          masked_brfile, n, p)
