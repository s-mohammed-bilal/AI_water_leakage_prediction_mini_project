# Load Libraries
library(shiny)
library(ggplot2)
library(dplyr)
library(DT)
library(caret)
library(rpart)

# Load Dataset
data <- read.csv("data/detection.csv")

colnames(data) <- c(
  "Timestamp",
  "Sensor_ID",
  "Pressure",
  "Flow_Rate",
  "Temperature"
)

data <- as.data.frame(data)

# Feature Engineering
avg_flow <- data %>%
  group_by(Sensor_ID) %>%
  summarise(Avg_Flow = mean(Flow_Rate))

data <- left_join(data, avg_flow, by="Sensor_ID")

# Calculate Water Loss
data$Water_Loss <- data$Flow_Rate - data$Avg_Flow

# Leakage Rule
data$Leakage <- ifelse(abs(data$Water_Loss) > 1,"Yes","No")
data$Leakage <- as.factor(data$Leakage)

# Holdout Validation (80-20)
set.seed(123)
train_index <- createDataPartition(data$Leakage, p = 0.8, list = FALSE)
train_data <- data[train_index,]
test_data  <- data[-train_index,]

# Train Model
model <- train(
  Leakage ~ Pressure + Flow_Rate + Temperature + Water_Loss,
  data = train_data,
  method = "rpart"
)

# Test Model
pred_test <- predict(model, test_data)
conf_matrix <- confusionMatrix(pred_test, test_data$Leakage)
accuracy_value <- conf_matrix$overall["Accuracy"]
print(conf_matrix)

# UI
ui <- fluidPage(
  titlePanel(
    div(
      style = "text-align:center;",
      "AI Water Pipeline Leakage Prediction Dashboard"
    )
  ),
  sidebarLayout(
    sidebarPanel(
      selectInput("sensor","Sensor ID",choices = unique(data$Sensor_ID)),
      numericInput("flow","Flow Rate (L/s)",1.2),
      numericInput("pressure","Pressure (bar)",3),
      numericInput("temperature","Temperature (°C)",25),
      actionButton("predictBtn","Predict Leakage"),
      hr(),
      h4("Model Accuracy"),
      textOutput("accuracy")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Dataset",DTOutput("table")),
        tabPanel("Flow Rate Distribution",plotOutput("flowDist")),
        tabPanel("Pressure Distribution",plotOutput("pressureDist")),
        tabPanel("Water Loss Distribution",plotOutput("lossDist"))
      )
    )
  )
)

# Server
server <- function(input, output) {
  
  # Show Dataset
  output$table <- renderDT({
    datatable(data)
  })
  
  # Show Model Accuracy
  output$accuracy <- renderText({
    paste("Accuracy:", round(accuracy_value*100,2),"%")
    })
  
  # Flow Rate Distribution
  output$flowDist <- renderPlot({
    ggplot(data,aes(Flow_Rate))+
      geom_histogram(binwidth=0.1,fill="blue",alpha=0.6)+
      geom_density(aes(y=0.1*..count..),color="red",size=1)+
      ggtitle("Flow Rate Distribution")+
      xlab("Flow Rate (L/s)")+
      ylab("Count")
  })
  
  # Pressure Distribution
  output$pressureDist <- renderPlot({
    ggplot(data,aes(Pressure))+
      geom_histogram(binwidth=0.2,fill="green",alpha=0.6)+
      geom_density(aes(y=0.2*..count..),color="red",size=1)+
      ggtitle("Pressure Distribution")+
      xlab("Pressure (bar)")+
      ylab("Count")
  })
  
  # Water Loss Distribution
  output$lossDist <- renderPlot({
    ggplot(data,aes(Water_Loss))+
      geom_histogram(binwidth=0.1,fill="orange",alpha=0.6)+
      geom_density(aes(y=0.1*..count..),color="red",size=1)+
      ggtitle("Water Loss Distribution")+
      xlab("Water Loss (L/s)")+
      ylab("Count")
  })
  
  # Predict Leakage
  observeEvent(input$predictBtn,{
    avg_flow_sensor <- avg_flow$Avg_Flow[avg_flow$Sensor_ID == input$sensor]
    water_loss_input <- input$flow - avg_flow_sensor
    new_data <- data.frame(
      Pressure=input$pressure,
      Flow_Rate=input$flow,
      Temperature=input$temperature,
      Water_Loss=water_loss_input
    )
    
    pred <- predict(model,new_data)
    
    showModal(modalDialog(
      paste("Leakage Prediction:",pred)
    ))
  })
}

# Run App
shinyApp(ui,server)