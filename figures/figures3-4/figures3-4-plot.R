library(tidyverse)
library(grid)
set.seed(1)

# Load data
df1 = read_csv("figure-3-data.csv")
df2 = read_csv("figure-4-data.csv") %>% subset(Method=="tar")

# Combine (because OLS data in figure 3 is the same as in figure 4)
df <- rbind(df1, df2)
method.labels <- c('ols' = "OLS", 
                   'ar' = "AR($A$) = xPAR($W,Z$)",
                   'par5' = "${PAR}_{\\lambda_1}(W$)", 
                   'par10' = "${PAR}_{\\lambda_2}(W$)", 
                   'tar' = "TAR($A$)"
                   )
region.labels <- c("ols"="$C_{OLS}$", 
                   "ar"="$C_A(\\lambda_1) = C_{W,Z}(\\lambda_1)$", 
                   "par5"="$C_W(\\lambda_1)$", 
                   "par10"="$C_W(\\lambda_2)$", 
                   "tar"="Targeted distr.")

#### Functions for making circles and ellipses
ellipse <- function(center = c(0, 0), a = 1, b = 1, npoints=100, method="ols", rotation=diag(2), lty="solid"){
  xx = seq(-a, a, length.out = npoints)
  yy = b/a * sqrt(a**2 - xx**2)
  out = t(rotation%*%rbind(c(xx, rev(xx)), c(yy, -rev(yy))))
  out = sweep(out, 2, center, '+')
  return(data.frame(x=out[,1], y=out[,2], Method=method,  lty=lty))
}

# The theoretical ellipse is specified by eigvals and eigenvectors of E[AAT] + lamb*Omega_W
matrix.info <- read.table("extra-files/figure-3-matrix-eigvals.csv", sep=";", header=F)
radius.par5 = sqrt(matrix.info[,1])
radius.par10 = sqrt(matrix.info[,2])

U5 = as.matrix(matrix.info[,3:4])
U10 = as.matrix(matrix.info[,5:6])

# Compute guarantee sets
regions <- rbind(ellipse(lty="ols"), # OLS
                 ellipse(a=sqrt(1+5), b=sqrt(1+5), lty="ar"), #OLS <- AR
                 ellipse(a=radius.par5[1],b=radius.par5[2], method="par5", rotation=U5, lty="par5"), #PAR5
                 ellipse(a=sqrt(1+5), b=sqrt(1+5), method="par5", lty="ar"), #PAR5 <- AR
                 ellipse(a=radius.par10[1],b=radius.par10[2], method="par10", rotation=U10, lty="par10"), #PAR10
                 ellipse(a=sqrt(1+5), b=sqrt(1+5), method="par10", lty="ar"), #PAR10 <- AR
                 ellipse(a=sqrt(1+5), b=sqrt(1+5), method="ar", lty="ar")) #AR
regions$Method <- factor(regions$Method, levels=names(method.labels))
regions$lty <- factor(regions$lty, levels=c("ols", "par5", "par10", "ar"))

regions$Primary <- (as.character(regions$lty) == as.character(regions$Method))

# The theoretical ellipse is specified by eigvals and eigenvectors of E[AAT] + lamb*Omega_W
matrix.info.target <- read.table("extra-files/figure-4-matrix-eigvals.csv", sep=";", header=F)
shift = as.vector(matrix.info.target[,1])
radius = as.vector(matrix.info.target[,2])
lamb = as.vector(matrix.info.target[,3])[1]
rotat = as.matrix(matrix.info.target[,4:5])

targ <- ellipse(center=shift, a=sqrt(radius[1]), b=sqrt(radius[2]), rotation=rotat, method="tar", lty="tar") %>% select(-Method)
targ$lty <- factor(targ$lty, levels=names(region.labels))
targ$Method <- factor(targ$Method, levels=names(region.labels))

# Cut points to be inside interval (to cut away whitespace in tikzDevice)
lims = 5
df_ = df %>%
  subset(Method %in% c("ols", "par5", "par10", "ar")) %>%
  subset((-lims < A0) & (A0 < lims) & (-lims < A1) & (A1 < lims))

midpoint <- mean(log10(subset(df_, Method=="ols")$MSE))*1.2

# Order factors for plotting order
df_$Method <- factor(df_$Method, levels=names(method.labels))

# Plot
p <- ggplot(df_) +
  geom_point(aes(x=A0, y=A1, color=MSE), alpha=1, size=2) +
  geom_path(data=regions, aes(x,y, lty = lty, alpha=Primary), size=0.6, color="#000066", show.legend = T) +
  labs(x = "$\\nu_1$", y="$\\nu_2$")+
  theme_bw(base_size = 9) +
  scale_color_gradient2(
    low="#1c0f00", mid="#fcf78f", high="#D1654C",
    trans = "log10",
    midpoint = midpoint,
    limits = c(1, 10), oob=scales::squish,
  ) +
  scale_linetype_manual(values=c("solid", "11", "22", "33"),
                        breaks = c("ols", "ar", "par5", "par10"),
                        labels=as_labeller(region.labels)
  )+
  scale_x_continuous(breaks = c(-5, 0, 5)) + scale_y_continuous(breaks=c(-5, 0, 5)) + 
  scale_alpha_manual(values=c(0.4, 1), name=NULL, breaks=NULL) +
  guides(lty=guide_legend(title=NULL),
         alpha=guide_legend(title=NULL),
         color=guide_colorbar(order=1, title="MSPE",
                              barheight=unit(1.5, "cm")
                              )) +
  coord_fixed(ratio = 1, xlim=c(-lims, lims), ylim = c(-lims, lims)) +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_blank(),
        legend.spacing.x = unit(0.05, 'cm'),
        legend.text = element_text(margin = margin(l = 2, unit = "pt")),
        legend.margin=margin(-10,0,0,0),
        plot.margin = margin(0, 0, 0, -5),
        legend.title.align = 0) +
  facet_wrap(~Method, ncol=5, labeller=as_labeller(method.labels))
print(p)


### Target AR plot
# Cut points to be inside interval
lims = 5
df_ = df %>%
  subset(Method %in% c("tar", "ols")) %>%
  subset((-lims < A0) & (A0 < lims) & (-lims < A1) & (A1 < lims))

# Order factors for plotting order
df_$Method <- factor(df_$Method, levels=names(method.labels))

p <- ggplot(df_) +
  geom_point(aes(x=A0, y=A1, color=MSE), alpha=1, size=2) +
  geom_path(data=subset(regions, Method=="ols" & lty=="ols"), aes(x,y, lty = lty), size=0.6, color="#000066", alpha = 1, show.legend = T) +
  geom_path(data=targ, aes(x,y, lty=lty), size=0.6, color="#000066", alpha = 1, show.legend = T) +
  geom_point(aes(x=0, y=0), shape=4, size=0.8, color="#000066", show.legend=F) +
  labs(x = "$\\nu_1$", y="$\\nu_2$")+
  theme_bw(base_size = 9) +
  scale_color_gradient2(
    low="#1c0f00", mid="#fcf78f", high="#D1654C",
    trans = "log10",
    midpoint = midpoint,
    limits = c(1, 10), oob=scales::squish,
  ) +
  scale_linetype_manual(values=c("solid", "11", "22", "33", "11"),
                        breaks = c("ols", "ar", "par5", "par10", "tar"),
                        labels=as_labeller(region.labels)
  )+
  scale_x_continuous(breaks = c(-5, 0, 5)) + scale_y_continuous(breaks=c(-5, 0, 5)) + 
  guides(lty=guide_legend(title=NULL),
         color=guide_colorbar(order=1, title="MSPE",
                              barheight=unit(1.5, "cm")
                              )) +
  coord_fixed(ratio = 1, xlim=c(-lims, lims), ylim = c(-lims, lims)) +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_blank(),
        legend.spacing.x = unit(0.05, 'cm'),
        legend.text = element_text(margin = margin(l = 2, unit = "pt")),
        legend.margin=margin(-10,0,0,0),
        plot.margin = margin(0, 0, 0, -5),
        legend.title.align = 0) +
  facet_wrap(~Method, ncol=5, labeller=as_labeller(method.labels))
print(p)