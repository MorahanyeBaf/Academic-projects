---
title: "Automated Web Scraping in R"
---

.
```{r}

install.packages("rvest") 
library(rvest)

marketwatch_wbpg <- read_html(
  "https://www.marketwatch.com/story/bitcoin-jumps-after-credit-scare-2018-10-15"
)

marketwatch_wbpg %>%
  html_node("title") %>% 
  html_text()

marketwatch_wbpg %>%
  html_nodes("p") %>% 
  html_text()


marketwatch_bitcoin_articles <- read_html(
  "https://www.marketwatch.com/search?q=bitcoin&m=Keyword&rpp=15&mp=0&bd=false&rs=false"
)


urls <- marketwatch_bitcoin_articles %>%
  html_nodes("div.searchresult a") %>% 
  html_attr("href")

urls


datetime <- marketwatch_bitcoin_articles %>%
  html_nodes("div.deemphasized span") %>% 
  html_text()

datetime


datetime2 <- c()
for(i in datetime){
  correct_datetime <- grep("Today", i, invert=T, value=T)
  datetime2 <- append(datetime2, correct_datetime)
}

datetime <- datetime2

datetime


install.packages("lubridate") 
library(lubridate)


datetime_clean <- gsub("\\.","",datetime)

datetime_parse <- parse_date_time(
  datetime_clean, "%I:%M %p %m/%d/%Y"
)
datetime_parse


datetime_convert <- ymd_hms(
  datetime_parse, tz = "Africa/Johannesburg"
)
datetime_convert <- with_tz(
  datetime_convert, "Africa/Johannesburg"
)
datetime_convert


marketwatch_webpgs_datetimes <- data.frame(
  WebPg=urls, DateTime=datetime_convert
)
dim(marketwatch_webpgs_datetimes)


diff_in_hours <- difftime(
  Sys.time(), marketwatch_webpgs_datetimes$DateTime, units = "hours"
)
diff_in_hours
diff_in_hours <- as.double(diff_in_hours)
diff_in_hours
marketwatch_webpgs_datetimes$DiffHours <- diff_in_hours
head(marketwatch_webpgs_datetimes)


marketwatch_latest_data <- subset(
  marketwatch_webpgs_datetimes, DiffHours < 1
)
marketwatch_latest_data


titles <- c()
bodies <- c()
for(i in marketwatch_latest_data$WebPg){
  
  marketwatch_latest_wbpg <- read_html(i)
  title <- marketwatch_latest_wbpg %>%
    html_node("title") %>%
    html_text()
  titles <- append(titles, title)
  
  marketwatch_latest_wbpg <- read_html(i)
  body <- marketwatch_latest_wbpg %>%
    html_nodes("p") %>%
    html_text()
  one_body <- paste(body, collapse=" ")
  bodies <- append(bodies, one_body)
  
}

marketwatch_latest_data$Title <- titles
marketwatch_latest_data$Body <- bodies

names(marketwatch_latest_data)
marketwatch_latest_data$Title
marketwatch_latest_data$Body[1]



install.packages("stringr")
library(stringr)
clean_text_bodies <- str_squish(
  marketwatch_latest_data$Body
  )
clean_text_bodies[1]


install.packages("LSAfun") 
library(LSAfun)
summary <- c()
for(i in clean_text_bodies){
  top_info <- genericSummary(i,k=3);
  one_summary <- paste(top_info, collapse=" ")
  summary <- append(summary, one_summary)
}

summary

marketwatch_latest_data$Summary <- summary


install.packages("sendmailR")
library(sendmailR)

marketwatch_title_summary <- c()
for(i in 1:length(marketwatch_latest_data$Summary)){
  marketwatch_title_summary <- append(marketwatch_title_summary, marketwatch_latest_data$Title[i])
  marketwatch_title_summary <- append(marketwatch_title_summary, marketwatch_latest_data$Summary[i])
}

marketwatch_title_summary

from <- "<morahanyebaf9@gmail.com>"
to <- "<morahanyebaf9@gmail.com>"
subject <- "Hourly Summary of Bitcoin Events"
body <- marketwatch_title_summary             
mailControl <- list(smtpServer="ASPMX.L.GOOGLE.COM") 

sendmail(from=from,to=to,subject=subject,msg=body,control=mailControl)


```
