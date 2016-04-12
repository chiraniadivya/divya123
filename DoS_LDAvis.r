#----------------------------------------------------------------------------------------------------#
# import necessary R packages
#----------------------------------------------------------------------------------------------------#
rm(list=ls())
library(tm)

#----------------------------------------------------------------------------------------------------#
# load text data
#----------------------------------------------------------------------------------------------------#
txt_input <- read.csv("C:/Users/A564812/Downloads/help_desk/opinion.csv",header=F,sep=",")
txt_data <- as.character(txt_input[,1])

#----------------------------------------------------------------------------------------------------#
# Filtering noise
#----------------------------------------------------------------------------------------------------#
noiseFilter <- function(x){
  x <- tolower(x)      ## keep lower case
  
  #### filter symbols
  #x <- gsub("\\+"," ",x)  ## substitute + by space
  x <- gsub("\\-"," ",x)
  #x <- gsub("\\("," ",x)   
  #x <- gsub("\\)"," ",x)
  x <- gsub("\\[name\\]", " ",x)
  x <- removePunctuation(x, preserve_intra_word_dashes = TRUE) ## remove punctuation
  
  #### filter digits   
  #x <- gsub("[[:digit:]]{2}[[:alpha:]]{3}[[:digit:]]{2}", " ", x) # remove calendar dates
  #x <- gsub("a[[:digit:]]"," ",x)
  #x <- gsub("x[[:digit:]]"," ",x)
  #x <- gsub("w[[:digit:]]"," ",x)
  x <- gsub("[[:digit:]]", " ",x) ## remove digits
  
  #### filter stop words     
  x <- removeWords(x, stopwords("english"))    ## remove common English stop words
  x <- gsub("agent"," ",x)
  x <- gsub("client", " " , x)
  x <- gsub("sent","send", x)
  X <- gsub("acct","account",x)
  x <- gsub("e mail","email",x)
  
  #### filter extra space   
  x <- stripWhitespace(x) 
  x <- gsub("^\\s+|\\s+$", "", x) # remove white space
  
  return(x)
}
txt_data <- noiseFilter(txt_data)

#### word stemming

stemFlag <- T
if(stemFlag){
  for (i in 1:length(txt_data)){
    txt_data[i] <- paste(stemDocument(unlist(strsplit(txt_data[i]," "))), collapse=" ")
  }
}

#### initialize all uni-gram tokens and all bi-tri tokens: uncomment at the first run
uniToken_all <- colnames(as.matrix(DocumentTermMatrix(VCorpus(VectorSource(txt_data)))))
write.csv(data.frame(x=uniToken_all,EXCLUDE=replicate(length(uniToken_all),"K")),file="C:/Users/a564812/Downloads/help_desk/uniVocab_all.csv",row.names = F)

TrigramTokenizer <- function(a){RWeka::NGramTokenizer(a, RWeka::Weka_control(min = 2, max = 3))}
bitriToken_all <- colnames(as.matrix(DocumentTermMatrix(VCorpus(VectorSource(txt_data)), control=list(tokenize=TrigramTokenizer))))
write.csv(data.frame(x=bitriToken_all,EXCLUDE=replicate(length(bitriToken_all),"K")),file="C:/Users/a564812/Downloads/help_desk/bitriVocab_all.csv",row.names = F)

#### remove user defined stop words
myStopWords <- read.csv("C:/Users/a564812/Downloads/help_desk/list_of_stop_words.csv",header = T)
myStopWords <- stemDocument(c("p", "d", "y", "z", "k","g", "b","f","q","s","t","r","e","j","l","w","x",
                              "rd","st","ac","also","us","th","will","soelberg","kulaga", 'mmk', 'mail', 'client', 'overnight', 'fax', 'send', 
                              'letter', 'bazeley', 'stefani', 'at', 'center', 'message', 'date', 'please' , 'asap', 'team', 'pdf', 
                              as.character(myStopWords$x)[which(myStopWords$EXCLUDE == 'X')]))
txt_data <- gsub("^\\s+|\\s+$", "", stripWhitespace(removeWords(txt_data,unique(myStopWords)))) 

#----------------------------------------------------------------------------------------------------#
# Create term frequency matrix
#----------------------------------------------------------------------------------------------------#
#### load dictionary
myDict <- read.csv("C:/Users/a564812/Downloads/help_desk/bitriVocab_all.csv",header = T)
myDict <- as.character(myDict[myDict$EXCLUDE != 'X',1])

#### build Documents' TFM
topic.train <- VCorpus(VectorSource(txt_data))  
TrigramTokenizer <- function(a){RWeka::NGramTokenizer(a, RWeka::Weka_control(min = 2, max = 3))}
ctrl <- list(tokenize=TrigramTokenizer, dictionary = myDict, bounds = list(global=c(5,Inf))) 
topic.train <- DocumentTermMatrix(topic.train,control=ctrl)

#### TF-IDF filter: comment if with a solid dictionary
#library(slam)
rowTotals <- apply(topic.train , 1, sum) #Find sum of words for each doc
colTotals <- apply(topic.train>0 , 2, sum) #Find the sum of docs for each word
#term_tfidf <- tapply(topic.train$v/row_sums(topic.train)[topic.train$i], topic.train$j, mean) * log2(nDocs(topic.train)/col_sums(topic.train> 0))
term_tfidf <- tapply(topic.train$v/rowTotals[topic.train$i], topic.train$j, mean) * log2(nDocs(topic.train)/colTotals)
if (summary(term_tfidf)[1] <0.1){
  tfidfFilter <- mean(summary(term_tfidf)[1:2])
  topic.train <- topic.train[,term_tfidf >= tfidfFilter]
}

#### remove document with no tokens in dictionary
topic.train <- topic.train[apply(topic.train , 1, sum) > 0, ] 
save(topic.train,file="C:/Users/a564812/Downloads/help_desk/vocab_token_bi_tri_exclusions/dtfm_bi_tri.RData")

#----------------------------------------------------------------------------------------------------#
# Topic Modeling: LDA
#----------------------------------------------------------------------------------------------------#
library(topicmodels)
library(SnowballC)

load("C:/Users/a564812/Downloads/help_desk/vocab_token_bi_tri_exclusions/dtfm_bi_tri.RData")
nTopic <- 3
#model.train <- LDA(topic.train,nTopic,method="VEM") # "Gibbs" or "VEM"
model.train <- CTM(topic.train,nTopic,method="VEM")
save(model.train, file = "C:/Users/a564812/Downloads/help_desk/LDA_output/LDA_model_60_bi_tri.RData")

#### save LDAvis model
myAPdata <- list()
v <- as.matrix(topic.train)
myAPdata$vocab <- model.train@terms
myAPdata$term.frequency <- as.integer(colSums(v))
myAPdata$phi <- exp(model.train@beta)
myAPdata$topic.proportion <- model.train@gamma
myAPdata$doc.length <- as.integer(rowSums(v))
save(myAPdata, file="C:/Users/a564812/Downloads/help_desk/LDA_output/LDAvis_60_bi_tri.RData")

#### get the index of valid documents and their topics
doc_index <- as.integer(rownames(v))
topic_index <- max.col(myAPdata$topic.proportion,ties.method = "first")


#----------------------------------------------------------------------------------------------------#
# Topic Modeling: LDA
#----------------------------------------------------------------------------------------------------#
#### LDA Visualization
library(LDAvis)
library(tsne)
library(servr)
load("C:/Users/a564812/Downloads/help_desk/LDAvis_10_bi_tri.RData")
source("C:/Users/a564812/Downloads/help_desk/myCreateJSON.R")
z <- myCreateJSON(phi=myAPdata$phi,
                  theta=myAPdata$topic.proportion,
                  vocab=myAPdata$vocab,
                  doc.length = myAPdata$doc.length,
                  term.frequency = myAPdata$term.frequency,
                  mds.method = function(x) tsne(svd(x)$u),
                  plot.opts = list(xlab="", ylab="")
                  )
serVis(z$myJSON,out.dir="C:/Users/a564812/Downloads/help_desk/LDAvis_webpage/LDAvis_10_bi_tri")
doc_topic <- apply(matrix(topic_index,ncol=1),1,FUN=function(x){z$topic_order[x]})
write.csv(data.frame(doc=txt_input[doc_index,1],topic=doc_topic), file="C:/Users/a564812/Downloads/help_desk/LDAvis_docTopic.csv", row.names=F)


term=terms(model.train,10)
term

write.csv(term,'term_5.csv')
write.csv(doc_index,'doc_index.csv')
z$topic_order
