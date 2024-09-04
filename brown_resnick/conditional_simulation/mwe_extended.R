library("dplyr")
library("parallel")
library("reticulate")
library("devtools")
working_directory <- (strsplit(getwd(), "/conditional_simulation")[[1]])[1]
spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
devtools::install(spatialextremes_directory)

#m is the number of missing locations


#1 if observed
produce_mask <- function(observed_indices, n)
{
    mask <- array(0, dim = c((n**2)))
    mask[observed_indices] <- rep(1, length(observed_indices))
    return(mask)
}

seed_value <- 34234
set.seed(seed_value)
n <- 32
s1 <- s2 <- seq(-10, 10, length.out = n)
s <- cbind(s1, s2)
nrep <- 50
df <- expand.grid(s1 = s1, 
                  s2 = s2) %>%
  mutate(z = c(SpatialExtremes::rmaxstab(1, coord = s, cov.mod = "powexp", 
                        nugget = 0, range = 3,
                        smooth = 1.6, grid = TRUE)))

## Simulate data
N <- 1L
#lscales <- runif(N, min = 1, max = 3)

#X <-   rmaxstab(1, coord = s, cov.mod = "powexp", 
           #nugget = 0, range = 1,
           #smooth = 1.5, grid = TRUE)
obsn <- 10
#make sure observed are in 32 by 32 part
idx_pred_locs <- -sample((n**2), obsn, replace = FALSE)
#idx_pred_locs <- -(1:obsn)
startTime <- Sys.time()
#ask Andrew if there is a reason nugget is not zero in condrmaxstab but is in rmaxstab
output <- SpatialExtremes::condrmaxstab(nrep, coord = df[idx_pred_locs, 1:2] %>% as.matrix(),
             cond.coord = df[-idx_pred_locs ,1:2] %>% as.matrix(),
             cond.data = df[-idx_pred_locs, "z"],
             cov.mod = "powexp", 
             nugget = 0, 
             range = 3,
             smooth = 1.6)

np <- import("numpy")
#output returns a list with the following elements: sim (only size of missing locations)
endTime <- Sys.time()
print(endTime - startTime)
condsim <- (output["sim"])[[1]]
condsim_array <- as.array(condsim)
mask <- produce_mask(-idx_pred_locs, n)
np$save("data/mwe/ref_image2/preprocessed_conditional_simulations_powexp_range_3_smooth_1.6.npy", condsim_array)
np$save("data/mwe/ref_image2/observed_simulation_powexp_range_3_smooth_1.6.npy", df$z)
np$save("data/mwe/ref_image2/mask.npy", mask)
np$save("data/mwe/ref_image2/seed_value.npy", array(seed_value))
