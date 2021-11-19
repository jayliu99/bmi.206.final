setwd('/Users/jayliu/BMI.206/grp_prj')
eps = 1e-8 # Add this to resolve log(0)

bridges = as.numeric(read.table("bridges.csv", header=FALSE, sep=","))
cascades = as.numeric(read.table("cascades.csv", header=FALSE, sep=","))

hist(bridges+eps) # Not normally distributed, very left-skewed
summary(bridges)
table(cascades) # Most nodes have casacde number = 0

y = log(bridges+eps) # Add eps to deal with 0 values for bridging coeff.
hist(y) # More normally distributed

#Fit linear model w/o log transform
mod1 = lm(bridges~cascades)
qqnorm(mod1$residuals)
qqline(mod1$residuals)

#Fit linear model with log transform
y = log(bridges + eps)
mod2 = lm(y~cascades)
qqnorm(mod2$residuals)
qqline(mod2$residuals)

#Summary statistics
summary(mod2)
hist(mod2$residuals)
boxplot(y~cascades,xlab="Cascade#",ylab="Bridging Coeff.")
abline(mod2, col='red')
legend("bottomright",legend=paste("R2 = ", format(summary(mod2)$r.squared,digits=3)))




