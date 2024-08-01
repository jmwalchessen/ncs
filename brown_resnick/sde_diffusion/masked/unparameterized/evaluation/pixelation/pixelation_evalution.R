
library(reticulate)
library(devtools)
working_directory <- (strsplit(getwd(), "/sde_diffusion")[[1]])[1]
#spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
#devtools::install(spatialextremes_directory)
library(gstat)

number_of_replicates <- 1
n.size <- 1024
nn <- sqrt(n.size)
x <- y <- seq(-10, 10, length = nn)
coord <- expand.grid(x, y)
range <- 1.6
smooth <- 1.6
#z <- SpatialExtremes::rmaxstab(n = number_of_replicates, coord = coord, cov.mod = "brown", range = range, smooth = smooth)
print(coord)

