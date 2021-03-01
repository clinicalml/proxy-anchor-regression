library(ggplot2)
library(dplyr)
library(readr)
library(tikzDevice)

# Naming
method.names <- c("par5" = "PAR($W$)", "cross5" = "xPAR($W,Z$)", "ar5" = "AR($A$)")
predictor.names <- c("$X_{\\rm{causal}}^1$", "$X_{\\rm{causal}}^2$", "$X_{\\rm{causal}}^3$",
                     "$X_{\\rm{anti-causal}}^1$", "$X_{\\rm{anti-causal}}^2$", "$X_{\\rm{anti-causal}}^3$")

# Read data
df = read_csv("experiment3-data.csv") %>%
  transform(Method = factor(Method, levels=names(method.names)), Causal <- factor(Causal)) %>% 
  subset(Method!="ar5") %>% 
  group_by(X.coord, Method, Causal) %>% 
  summarize(sd = sd(abs(Weight)), Weight = mean(abs(Weight)), ymin = Weight - sd, ymax = Weight+sd) %>%
  ungroup()

# Plot
p <- ggplot(subset(df, Method != "ar5"), aes(x=X.coord, y=abs(Weight), fill = Method)) + 
  geom_bar(stat="identity", position=position_dodge(width=0.9)) + 
  geom_errorbar(aes(ymin = ymin,ymax = ymax), stat="identity", position = position_dodge(width=0.9), size=0.1, width=0.25) +
  coord_flip() +
  labs(x=NULL, y = "$|$Regression coefficients$|$") + 
  scale_x_continuous(breaks=c(1:6), labels=predictor.names) +
  scale_fill_brewer(palette="Dark2", breaks=c("cross5", "par5"), labels=as_labeller(method.names)) +
  theme_bw(base_size=9) +
  theme(legend.key.size = unit(0.8,"line")) +
  guides(fill = guide_legend(title=NULL, override.aes = list(size = 0.3)))
print(p)
