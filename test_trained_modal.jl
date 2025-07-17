# Simple Test Script - Direct approach using main.jl
# This includes the main.jl file which has all model definitions

include("main.jl")

println("🚀 Simple Spam Detection Test")
println("=" ^ 40)

# Load the trained model directly using BSON
println("Loading model...")
model_data = BSON.load("trained_model.bson")
trained_model = model_data[:trained_final_model]
vocab = model_data[:vocab]
max_seq_len = model_data[:max_seq_len]

println("Model loaded!")

# Simple prediction function
function simple_predict(text::String)
    # Preprocess text (using functions from main.jl)
    processed = full_preprocessing(text)
    
    # Convert to indices
    indices = text_to_indices(processed, vocab, max_seq_len)
    input_tensor = reshape(indices, length(indices), 1)
    
    # Predict
    logits = trained_model(input_tensor)
    if size(logits, 1) == 1 && size(logits, 2) == 2
        logits = transpose(logits)
    end
    
    probabilities = softmax(logits, dims=1)
    pred_class = Flux.onecold(logits, 1:2)[1]
    prediction = pred_class == 1 ? "Ham" : "Spam"
    confidence = pred_class == 1 ? probabilities[1, 1] : probabilities[2, 1]
    
    println("Text: \"$(text[1:min(50, length(text))])...\"")
    println("Prediction: $prediction ($(round(confidence*100, digits=1))%)")
    println()
    
    return prediction, confidence
end

# Test examples
println("\n Testing")

# Test 1: Spam
simple_predict("URGENT! You have won a million dollars! Click here now!")

# Test 2: Ham  
simple_predict("Hi Sarah, can you send me the project timeline?")

# Test 3: Promotional
simple_predict("Don't miss our special sale! 20% off this weekend only.")

println("Tests completed!") 