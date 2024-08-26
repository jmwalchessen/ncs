library("dplyr")
library("parallel")
library("reticulate")
library("devtools")
working_directory <- (strsplit(getwd(), "/conditional_simulation")[[1]])[1]
spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
devtools::install(spatialextremes_directory)

diff <- .645161285
start_pt <- -10-16*diff
end_pt <- 10+16*diff
#m is the number of missing locations
process_condrmaxstab <- function(condsim, m, nrep)
{
    processed_array <- array(0, dim = c(m,nrep))
    for(irep in 1:nrep)
    {
        processed_array[,irep] <- condsim[(m*(irep-1)+1):(m*irep)]
    }
    return(processed_array)
}

set.seed(395234)
n <- 64
s1 <- s2 <- seq(-10, 10, length.out = n)
s <- cbind(s1, s2)
nrep <- 2
df <- expand.grid(s1 = s1, 
                  s2 = s2) %>%
  mutate(z = c(SpatialExtremes::rmaxstab(1, coord = s, cov.mod = "brown", 
                        nugget = 0, range = 1.6,
                        smooth = 1.6, grid = TRUE)))

## Simulate data
N <- 1L
#lscales <- runif(N, min = 1, max = 3)

#X <-   rmaxstab(1, coord = s, cov.mod = "powexp", 
           #nugget = 0, range = 1,
           #smooth = 1.5, grid = TRUE)
obsn <- 50
#make sure observed are in 32 by 32 part
idx_pred_locs <- -sample((n**2), obsn, replace = FALSE)
startTime <- Sys.time()
#ask Andrew if there is a reason nugget is not zero in condrmaxstab but is in rmaxstab
output <- SpatialExtremes::condrmaxstab(nrep, coord = df[idx_pred_locs, 1:2] %>% as.matrix(),
             cond.coord = df[-idx_pred_locs ,1:2] %>% as.matrix(),
             cond.data = df[-idx_pred_locs, "z"],
             cov.mod = "brown", 
             nugget = 0, 
             range = 1.6,
             smooth = 1.6)

np <- import("numpy")
#output returns a list with the following elements: sim (only size of missing locations)
endTime <- Sys.time()
print(endTime - startTime)
condsim <- (output["sim"])[[1]]
condsim_array <- as.array(condsim)
m <- (n**2)-obsn
condsim <- process_condrmaxstab(condsim, m, nrep)
zsim <- array(0, dim = c((n**2),nrep))
#dim of condsim is number of missing locations
zsim[idx_pred_locs,] <- condsim
obsz <- rep(df$z, times = nrep)
dim(obsz) <- c((n**2),nrep)
zsim[-idx_pred_locs,] <- obsz[-idx_pred_locs,]
#results <- cbind(df$z, df$zsim)
#print(dim(df$zsim))
print(dim(obsz))
np$save("data/mwe/conditional_simulations_brown_range_1.6_smooth_1.6.npy", condsim_array)
np$save("data/mwe/observed_simulation_brown_range_1.6_smooth_1.6.npy", df$z)