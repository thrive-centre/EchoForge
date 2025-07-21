from echoforge import load_model

# Load any model by name from the registry
model = load_model("Echo2DClassifier", pretrained=True)

model.summary()