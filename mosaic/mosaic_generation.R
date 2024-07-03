library(spatstat)
X <- rpoislinetess(lambda = 3)
plot(as.im(X), main="rpoislinetess(3)")
plot(X, add=TRUE)