library("SpatialExtremes")
library("dplyr")
library("parallel")
library("reticulate")

set.seed(1)

s1 <- s2 <- seq(-10, 10, length.out = 64)
print(s1)
s <- cbind(s1, s2)
df <- expand.grid(s1 = s1, 
                  s2 = s2) %>%
  mutate(z = c(rmaxstab(1, coord = s, cov.mod = "powexp", 
                        nugget = 0, range = 3,
                        smooth = 1.5, grid = TRUE)) %>% log())

## Simulate data
N <- 1L
#lscales <- runif(N, min = 1, max = 3)

#X <-   rmaxstab(1, coord = s, cov.mod = "powexp", 
           #nugget = 0, range = 1,
           #smooth = 1.5, grid = TRUE)

idx_pred_locs <- -(1:50)
startTime <- Sys.time()
#ask Andrew if there is a reason nugget is not zero in condrmaxstab but is in rmaxstab
output <- condrmaxstab(1, coord = df[idx_pred_locs, 1:2] %>% as.matrix(),
             cond.coord = df[-idx_pred_locs ,1:2] %>% as.matrix(),
             cond.data = df[-idx_pred_locs, "z"],
             cov.mod = "powexp", 
             nugget = 0, 
             range = 1,
             smooth = 1.5)

np <- import("numpy")
#output returns a list with the following elements: sim (only size of missing locations)
endTime <- Sys.time()
print(endTime - startTime)
condsim <- array((output["sim"])[[1]])
condsim <- as.array(condsim)
df$zsim <- rep(0, nrow(df))
df$zsim[idx_pred_locs] <- condsim
df$zsim[-idx_pred_locs] <- df[-idx_pred_locs, "z"]
#dim of condsim is number of missing locations
results <- cbind(df$z,df$zsim)
np$save("data/mwe/conditional_simulations1.npy", results)