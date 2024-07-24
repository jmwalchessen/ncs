library("SpatialExtremes")
library("dplyr")
library("parallel")
library("devtools")
working_directory <- (strsplit(getwd(), "/conditional_simulation")[[1]])[1]
spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
devtools::install(spatialextremes_directory)

set.seed(1)

s1 <- s2 <- seq(-10, 10, length.out = 32)
s <- cbind(s1, s2)
df <- expand.grid(s1 = s1, 
                  s2 = s2) %>%
  mutate(z = c(SpatialExtremes::rmaxstab(1, coord = s, cov.mod = "brown", 
                        nugget = 0, range = 1.6,
                        smooth = 1.6, grid = TRUE)))

## Simulate data
#N <- 1L
#lscales <- runif(N, min = 1, max = 3)

#X <-   rmaxstab(1, coord = s, cov.mod = "brown", 
           #nugget = 0, range = 1.6,
           #smooth = 1.6, grid = TRUE)

idx_pred_locs <- -(1:2)
print(typeof(idx_pred_locs))
nugget <- 0
condvector <- condrmaxstab(1, coord = df[idx_pred_locs, 1:2] %>% as.matrix(),
                           cond.coord = df[-idx_pred_locs ,1:2] %>% as.matrix(),
                           cond.data = df[-idx_pred_locs, "z"], cov.mod = "brown",
                           nugget = 0, range = 1.6, smooth = 1.6)
print(condvector["sim"])
