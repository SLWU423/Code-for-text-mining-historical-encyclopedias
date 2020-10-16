library(NLP)
library(tm)
library(topicmodels)

library(Rwordseg)
library(jiebaRD)
library(jiebaR)


library(RColorBrewer)
library(wordcloud)
library(igraph)

library(ggplot2)

library(slam)
library(dplyr)
library(tidytext)


installDict("/Desktop/trail/Dict/NongXue.scel", dictname = "NongXue")
installDict("/Desktop/trail/Dict/GuangYiNongYe.scel", dictname = "GuangYiNongYe")
installDict("/Desktop/trail/Dict/NongYeCiHuiDaQuan.scel", dictname = "NongYeCiHuiDaQuan")
installDict("/Desktop/trail/Dict/NongYeLeiCiKu.scel", dictname = "NongYeLeiCiKu")
installDict("/Desktop/trail/Dict/NongYeXue.scel", dictname = "NongYeXue")


cutter = worker(type = "mix", stop_word = "/Desktop/stopwords_1906.txt")

cutter["...中国农业发展史_1前言.txt"]
cutter["...中国农业发展史_2原始农业.txt"]
cutter["...中国农业发展史_3夏商.txt"]
cutter["...中国农业发展史_4西周春秋战国.txt"]
cutter["...中国农业发展史_5传统农业.txt"]
cutter["...中国农业发展史_6农书和结束语.txt"]
cutter["...中国农业科学技术史稿_1原始社会时期.txt"]
cutter["...中国农业科学技术史稿_2夏商西周时期.txt"]
cutter["...中国农业科学技术史稿_3春秋战国时期.txt"]
cutter["...中国农业科学技术史稿_4秦汉时期.txt"]
cutter["...中国农业科学技术史稿_5魏晋南北朝时期.txt"]
cutter["...中国农业科学技术史稿_6隋唐时期.txt"]
cutter["...中国农业科学技术史稿_7宋元时期.txt"]
cutter["...中国农业科学技术史稿_8明清时期.txt"]
cutter["...中国农业科学技术史稿_9结束语.txt"]
cutter["...中国科学技术史农学卷_1序和导言.txt"]
cutter["...中国科学技术史农学卷_2先秦时期.txt"]
cutter["...中国科学技术史农学卷_3秦汉魏晋时期.txt"]
cutter["...中国科学技术史农学卷_4隋唐宋元时期.txt"]
cutter["...中国科学技术史农学卷_5明清时期.txt"]
cutter["...中国科学技术史农学卷_6结束语.txt"]



txt<-Corpus(DirSource('...seg results'),
            readerControl = list(reader = readPlain, 
                                 language = "cn"))

inspect(txt)

txt.temp<-Corpus(VectorSource(txt), 
                 readerControl = list(reader = readPlain, 
                                      language = "cn"))

inspect(txt.temp)

#Data pre-processing
stopwordsCN = readLines("...stopwords_1906.txt")
stopwordsCN<-enc2utf8(stopwordsCN)
stopwordsCN<-stopwordsCN[Encoding(stopwordsCN)!="unknown"]

txt.temp<-tm_map(txt.temp, removeWords, stopwordsCN)
txt.temp<-tm_map(txt,removePunctuation)
txt.temp<-tm_map(txt,removeNumbers)
txt.temp<-tm_map(txt,stripWhitespace)
inspect(txt.temp)


#Finding frequencies for all extracted words
tdm.txt<- TermDocumentMatrix(txt.temp)
tdm.txt2<- as.matrix(tdm.txt)

tdm.txt.removed<- as.matrix(removeSparseTerms(tdm.txt, 0.95))

View(tdm.txt2)
write.csv(tdm.txt2, "...txt_TD.csv", fileEncoding = "UTF-8")

txt.freq<- data.frame(ST = rownames(tdm.txt2), Freq = rowSums(tdm.txt2), row.names = NULL)

View(txt.freq)
write.csv(txt.freq, "...txt_freq.csv", fileEncoding = "UTF-8")



#topic modelling using LDA 
#Finding suitable number of topics
dtm<- DocumentTermMatrix(txt.temp,control = list(weighting = weightTf))
#inspect(dtm)
term_tfidf<- tapply(dtm$v/row_sums(dtm)[dtm$i], dtm$j, mean)*log2(nDocs(dtm)/col_sums(dtm>0))
l1=term_tfidf>=quantile(term_tfidf,0.5)
dtm<-dtm[,l1]
dtm = dtm[row_sums(dtm)>0,]
summary(col_sums(dtm))

fold_num = 10
kv_num = c(5, 10*c(1:5, 10))
seed_num = 2003
try_num = 1

smp<- function(cross = fold_num, n, seed){
  set.seed(seed)
  dd=list()
  aa0=sample(rep(1:cross, ceiling(n/cross))[1:n], n)
  for (i in 1:cross) dd[[i]]=(1:n)[aa0==i]
  return(dd)}

selectK<- function(dtm, kv=kv_num, SEED=seed_num, cross=fold_num, sp){
  per_ctm=NULL
  log_ctm=NULL
  for(k in kv){
    per=NULL
    loglik=NULL
    for (i in 1:try_num){
      cat("R is running for", "topic", k, "fold", i, as.character(as.POSIXct(Sys.time(), "Asia/Shanghai")),"\n")
      te=sp[[i]]
      tr=setdiff(1:dtm$nrow, te)
#      VEM = LDA(dtm[tr,], k = k, control = list(seed = SEED)),
#      VEM_fixed = LDA(dtm[tr,], k = k, control = list(estimate.alpha = FALSE, seed = SEED)),
#      CTM = CTM(dtm[tr,], k = k, control = list(seed = SEED, var = list(tol = 10^-4), em = list(tol = 10^-3)))
      Gibbs = LDA(dtm[tr,], k = k, method = "Gibbs", control = list(seed = SEED, burnin = 1000, thin=100, iter=1000))
      per=c(per, perplexity(Gibbs, newdata=dtm[te,]))
      loglik=c(loglik,logLik(Gibbs, newdata=dtm[te,]))
    }
    per_ctm=rbind(per_ctm, per)
    log_ctm=rbind(log_ctm, loglik)
  }
  return(list(perplex=per_ctm, loglik=log_ctm))
}

sp=smp(n=dtm$nrow,seed=seed_num)

system.time((ctmK=selectK(dtm=dtm, kv=kv_num, SEED=seed_num,cross=fold_num,sp=sp)))

m_per=apply(ctmK[[1]], 1, mean)
m_log=apply(ctmK[[2]], 2, mean)

k=c(kv_num)
df=ctmK[[1]]
logLik=ctmK[[2]]

#Plotting perplexity and loglikelihhod graphs to determine optimum number of topics
write.csv(data.frame(k, df, logLik), "...Perplexity_Gibbs.csv", fileEncoding = "UTF-8")

png("...Perplexity_gibbs.png", width=5, height=5, units="in", res=700)

matplot(k,df,type=c("b"), xlab="Number of topics", ylab="Perplexity", pch=1:try_num, col=1, main='')
legend("topright", legend=paste("fold", 1:try_num), col=1, pch=1:try_num)
dev.off()

png("...LogLikelihood_gibbs.png", width=5, height=5, units="in", res=700)

matplot(k, logLik, type = c("b"), xlab = "Number of topics",ylab = "Log-Likelihood", pch=1:try_num,col = 1, main = '')       
legend("topright", legend = paste("fold", 1:try_num), col=1, pch=1:try_num)
dev.off()



#Using the suitable number of topic to model
k = 20
SEED <- 2003
jss_TM2 <- list(
  VEM = LDA(dtm, k = k, control = list(seed = SEED)),
  VEM_fixed = LDA(dtm, k = k, control = list(estimate.alpha = FALSE, seed = SEED)),
  Gibbs = LDA(dtm, k = k, method = "Gibbs", control = list(seed = SEED, burnin = 1000, thin = 100, iter = 2000)),
  CTM = CTM(dtm, k = k, control = list(seed = SEED, var = list(tol = 10^-4), em = list(tol = 10^-3))) )   
#save(jss_TM2, file = paste(getwd(), "/jss_TM2.Rdata", sep = ""))
#save(jss_TM, file = paste(getwd(), "/jss_TM1.Rdata", sep = ""))

#Function "terms" group the top 30 terms
termsForSave1<- terms(jss_TM2[["VEM"]],30)
termsForSave2<- terms(jss_TM2[["VEM_fixed"]], 30)
termsForSave3<- terms(jss_TM2[["Gibbs"]], 30)
termsForSave4<- terms(jss_TM2[["CTM"]], 30)

#drawing term frequency (under different topics) graphs
SaveAllTerms1<- LDA(dtm, k = k, control=list(seed=SEED))
SaveAllTerms2<- LDA(dtm, k = k, control = list(estimate.alpha = FALSE, seed = SEED))
SaveAllTerms3<- LDA(dtm, k = k, method = "Gibbs", control = list(seed = SEED, burnin = 1000, thin = 100, iter = 2000))
SaveAllTerms4<- CTM(dtm, k = k, control = list(seed = SEED, var = list(tol = 10^-4), em = list(tol = 10^-3)))

tidyAllTerms1<- tidy(SaveAllTerms1)
tidyAllTerms2<- tidy(SaveAllTerms2)
tidyAllTerms3<- tidy(SaveAllTerms3)
tidyAllTerms4<- tidy(SaveAllTerms4)

top_terms<- tidyAllTerms3 %>%
  group_by(topic) %>%
  top_n(50, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)
top_terms

png("...term frequency in 20 topics_Gibbs-2000 iter.png",
    width=40, height=30,
    units="in", res=500)

top_terms %>%
  mutate(term = reorder(term, beta))%>%
  ggplot(aes(term,beta, fill = factor(topic)))+
  theme_grey(base_family = 'STKaiti')+
  theme(text = element_text(size=8))+
  geom_bar(alpha=0.8, stat="identity", show.legend=FALSE)+
  facet_wrap(~topic, scales="free")+
  coord_flip()

dev.off()
#drawing term frequency (under different topics) graphs ends



#drawing topic graphs in network form
tfs = as.data.frame(termsForSave4, stringsAsFactors = F); tfs[,1]
adjacent_list = lapply(1:10, function(i) embed(tfs[,i], 2)[, 2:1])
edgelist = as.data.frame(do.call(rbind, adjacent_list), stringsAsFactors =F)
#View(edgelist)

topic = unlist(lapply(1:30, function(i) rep(i, 9)))
#edgelist$topic = topic

par(family = 'STKaiti')
g <-graph.data.frame(edgelist,directed=T)
l<-layout.fruchterman.reingold(g)
#edge.color="black"
nodesize = centralization.degree(g)$res
V(g)$size = log(centralization.degree(g)$res)
V(g)$attribute <- V(g)$name
E(g)$color =  unlist(lapply(sample(colors()[26:137], 30), function(i) rep(i, 9))); unique(E(g)$color)

png("...topic_graph_gibbs.png",
            width=20, height=25,
            units="in", res=700)
plot(g,edge.curved=TRUE,
     vertex.label.cex =1,edge.arrow.size=0.2,layout=l,vertex.label.family = 'STKaiti')
dev.off()
#End of codes
