library(SpatialExtremes)

args = commandArgs(trailingOnly=TRUE)
range <- as.numeric(args[1])
smooth <- as.numeric(args[2])
number_of_replicates <- as.numeric(args[3])
seed <- as.numeric(args[4])

n.size <- 961
nn <- sqrt(n.size)
x <- y <- seq(-10, 10, length = nn)
coord <- expand.grid(x, y)

y <- rmaxstab(n = number_of_replicates, coord = coord, cov.mod = "brown", range = range, smooth = smooth)
print(as.list(y))