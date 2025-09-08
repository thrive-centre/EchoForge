import numpy as np

from echoforge import load_model

def compare_weights(model1, model2, num_layers=3):
    """
    Compare the weights of the base models in two models.
    """
    base1 = model1.layers[0]
    base2 = model2.layers[0]

    print("\nComparing base model weights:")
    for i in range(num_layers):
        if i >= len(base1.weights) or i >= len(base2.weights):
            break
        w1 = base1.weights[i].numpy()
        w2 = base2.weights[i].numpy()

        if np.allclose(w1, w2):
            print(f"Layer {i}: SAME")
        else:
            print(f"Layer {i}: DIFFERENT")

if __name__ == "__main__":
    print("Loading model with pretrained=True")
    model_pretrained = load_model("EchoView47_classifierr", pretrained=True)
    model_pretrained.summary()

    print("\nLoading model with pretrained=False")
    model_fresh = load_model("EchoView47_classifier", pretrained=False)
    model_fresh.summary()

    compare_weights(model_pretrained, model_fresh)
