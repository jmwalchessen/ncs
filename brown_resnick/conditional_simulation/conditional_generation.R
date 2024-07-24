library("dplyr")
library("parallel")
library("devtools")
library("Rlab")
working_directory <- (strsplit(getwd(), "/conditional_simulation")[[1]])[1]
spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
devtools::install(spatialextremes_directory)

seed_value <- 235542
set.seed(seed_value)
n <- 32
s1 <- s2 <- seq(-10, 10, length.out = n)
s <- cbind(s1, s2)
df <- (expand.grid(s1 = s1,Vs2 = s2) %>% mutate(z = c(SpatialExtremes::rmaxstab(1, coord = s, cov.mod = "brown", nugget = 0, range = 1.6, smooth = 1.6, grid = TRUE))))

mask <- as.matrix(rbern(n = (n**2), p = .01))
observed_indices <- which(mask == 1)
missing_indices <- -observed_indices

k <- 10
condsim <- SpatialExtremes::condrmaxstab(k, coord = df[missing_indices, 1:2] %>% as.matrix(),
                           cond.coord = df[observed_indices,1:2] %>% as.matrix(),
                           cond.data = df[observed_indices, "z"], cov.mod = "brown",
                           nugget = 0, range = 1.6, smooth = 1.6)
condsim <- condsim$sim
condsim <- as.array(condsim)
#array is (k,m) where m=#of missing observations
print(rep(df$z[observed_indices], times = 3))
#cond_simulations <- array(0, dim = c(k,(n**2)))
#cond_simulations[,observed_indices] <- df$z[observed_indices]
#cond_simulations[,missing_indices] <- condsim$sim
