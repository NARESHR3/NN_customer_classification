# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="660" height="865" alt="image" src="https://github.com/user-attachments/assets/ccb1c864-7ccf-4c28-a008-19638ba1ed96" />

## DESIGN STEPS:

### STEP 1:
Understand the classification task and identify input and output variables.

### STEP 2:
Gather data, clean it, handle missing values, and split it into training and test sets.
### STEP 3:
Normalize/standardize features, encode categorical labels, and reshape data if needed.
### STEP 4:
Choose the number of layers, neurons, and activation functions for your neural network.

### STEP 5:
Select a loss function (e.g., binary cross-entropy), optimizer (e.g., Adam), and metrics (e.g., accuracy).


### STEP 6:
Feed training data into the model, run multiple epochs, and monitor the loss and accuracy.

### STEP 7:
Save the trained model, export it if needed, and deploy it for real-world use.


## PROGRAM

### Name: NARESH.R
### Register Number: 212223240104

```python
class NARESH(nn.Module):
    def __init__(self, input_size):
        super(NARESH, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)  # 4 classes (A, B, C, D)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)   # No softmax here (CrossEntropyLoss handles it)
        return x
```
```python
# Initialize the Model, Loss Function, and Optimizer
input_size = X_train.shape[1]
num_classes = 4

Naresh = NARESH(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Naresh.parameters(), lr=0.001)

```
```python
def train_model(Naresh, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = Naresh(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
```



## Dataset Information

<img width="1149" height="465" alt="image" src="https://github.com/user-attachments/assets/a666efea-0f91-4259-b67e-70dcd7fd2e0c" />


## OUTPUT



### Confusion Matrix

<img width="555" height="533" alt="image" src="https://github.com/user-attachments/assets/71b14626-6165-4dbe-8154-dce6ebc93f9d" />


### Classification Report

<img width="571" height="241" alt="image" src="https://github.com/user-attachments/assets/a6895600-e11b-4762-b606-e84b2e740420" />



### New Sample Data Prediction
<img width="601" height="165" alt="image" src="https://github.com/user-attachments/assets/59f94157-38ce-46b6-9953-c9f4f12ffeca" />

## RESULT

Thus the neural network classification model was successfully developed.
