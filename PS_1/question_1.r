# Question 1

# Task 1
library(readr)
data_directory <- file.path(getwd(),"IO_ProblemSets","PS_1","data","MROZ_mini.csv")
MROZ_mini <- read_csv(data_directory)

int <- matrix(rep(1, nrow(MROZ_mini)), ncol = 1)

expvar <- cbind(int, MROZ_mini$educ)

xtx <- t(expvar) %*% expvar

inv <- 1/(xtx[1,1]*xtx[2,2] - xtx[1,2]*xtx[2,1]) * matrix(c(xtx[2,2], -xtx[1,2], -xtx[2,1], xtx[1,1]), ncol = 2)

# solve(xtx)

beta_ols = inv %*% t(expvar) %*% MROZ_mini$lwage

y_ols = expvar %*% beta_ols

u_ols = MROZ_mini$lwage - y_ols

#summary(lm(MROZ_mini$lwage ~ MROZ_mini$educ))


# Task 2
meanx <- 1/nrow(MROZ_mini) * sum(MROZ_mini$educ)
meany <- 1/nrow(MROZ_mini) * sum(MROZ_mini$lwage)

beta1_ols = (sum((MROZ_mini$educ-meanx)*(MROZ_mini$lwage-meany)))/(sum((MROZ_mini$educ-meanx)^2))
beta0_ols = meany - beta1_ols*meanx

# Verification
beta_ols[1] - beta0_ols
beta_ols[2] - beta1_ols
