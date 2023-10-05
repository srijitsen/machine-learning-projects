
####################################### MA(2) Model ############################3


################### (a) ####################

# Set the values of p(1) and p(2)
p1 <- 1/2
p2 <- 1/3

# Generate 200 iid values of x(t) from a normal distribution with mean 0 and variance 1
x <- rnorm(200, mean = 0, sd = 1)

# Initialize the vector to store the values of y(t)
y <- numeric(200)

# Calculate the values of y(t) using the MA(2) model
for (t in 3:200) {
  y[t] <- x[t] + p1*x[t-1] + p2*x[t-2]
}

# Plot the values of y(t) against t
plot(1:200, y, type = "l", xlab = "t", ylab = "y(t)")

################## (b) #######################
# Compute the mean and variance of y(t)
mean_y <- mean(y)
var_y <- var(y)

# Compute the ACF of y(t) from the generated samples
acf_estimated <- acf(y, lag.max = 20, plot = FALSE)$acf

# Compute the ACF of y(t) and plot the ACF against k
acf_y <- acf(y, lag.max = 20, plot = TRUE)

############### (c) ####################

# Define the ACF values
acf_theoretical <- c(13/18, 1/2 * 13/18, 1/3 * 13/18, rep(0, 17))

# Plot the ACF using a stem-and-leaf plot
stem(acf_theoretical, scale = 2)

############## (d)  ##################


acf_theoretical <- c(13/18, 1/2 * 13/18, 1/3 * 13/18, rep(0, 17))

# Plot the theoretical ACF and estimated ACF together
plot(acf_theoretical, type = "h", ylim = c(-0.2, 1), xlab = "Lag (k)", ylab = "ACF")
lines(acf_estimated, type = "h", col = "red")
legend("topright", legend = c("Theoretical ACF", "Estimated ACF"), lty = c(1,1), col = c("black", "red"))

####################################### MA(2) Model ############################

################# (a) ################

# Parameters
p <- 0.8 # autoregressive coefficient
n <- 200 # number of time steps

# Compute theoretical ACF
lags <- 0:10 # lags to compute ACF for
acf_theoretical <- p^lags # theoretical ACF values


############## (b) ##############

acf <- p^lags # ACF values

# Generate time series
set.seed(123) # set random seed for reproducibility
eps <- rnorm(n) # generate iid noise
y <- arima.sim(list(ar=p), n, innov=eps)
y[1] <- 0 # set y(0) to 0

# Plot time series
plot(y, type="l", lwd=2, xlab="Time", ylab="y(t)", main="AR(1) Model")


######################### (c) ######################

# Compute estimated ACF
acf_estimated <- acf(y, lag.max=10, plot=FALSE)$acf[1:11]

# Plot ACFs
plot(lags, acf_theoretical, type="h", lwd=2, xlab="Lag", ylab="Autocorrelation",
     main="Theoretical and Estimated ACF for AR(1) Model")
points(lags, acf_estimated, type="h", lwd=2, col="red")

# Compute RMSE
rmse <- sqrt(mean((acf_theoretical - acf_estimated)^2))
cat("RMSE:", rmse, "\n")



