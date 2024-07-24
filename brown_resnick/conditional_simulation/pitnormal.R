library(stats)
library(dplyr)
library("devtools")
working_directory <- (strsplit(getwd(), "/conditional_simulation")[[1]])[1]
spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
devtools::install(spatialextremes_directory)
seed_value <- 235542
set.seed(seed_value)
n <- 32
s1 <- s2 <- seq(-10, 10, length.out = n)
s <- cbind(s1, s2)
logbr_samples <- SpatialExtremes::rmaxstab(1000, coord = s, cov.mod = "brown", 
                        nugget = 0, range = 1.6,
                        smooth = 1.6, grid = TRUE) %>% log()
empirical_cdf <- ecdf(logbr_samples)
pit_samples <- empirical_cdf(logbr_samples)
    