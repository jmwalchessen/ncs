 library("SpatialExtremes")
library("dplyr")
library("parallel")
library("reticulate")
library("devtools")
working_directory <- (strsplit(getwd(), "/conditional_simulation")[[1]])[1]
spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
devtools::install(spatialextremes_directory)

set.seed(234825)

s1 <- s2 <- seq(-10, 10, length.out = 64)
s <- cbind(s1, s2)
rsamples <- SpatialExtremes::rmaxstab(50, coord = s, cov.mod = "brown", range = 1.6, smooth = 1.6, grid = TRUE)
print(dim(rsamples))
np <- import("numpy")
np$save("data/mwe/unconditional_samples.npy", as.array(rsamples))