# Spam Detection with Lightweight BERT Model
# Required packages (install once with): 
# using Pkg; Pkg.add(["CSV", "DataFrames", "Flux", "TextAnalysis", "Languages", "StatsBase", "Random", "MLUtils", "OneHotArrays", "Plots", "StatsPlots", "ProgressMeter", "WordTokenizers", "BSON"])

using CSV, DataFrames, Flux, TextAnalysis, Languages, StatsBase, Random, MLUtils
using OneHotArrays, Plots, StatsPlots, ProgressMeter, WordTokenizers
using Statistics, LinearAlgebra, BSON

# Set random seed for reproducibility
Random.seed!(42)

# =============================================
# DATA PREPROCESSING PIPELINE
# =============================================

function preprocess_text(text::AbstractString)
    text = lowercase(String(text))
    text = replace(text, r"\[.*?\]" => "")
    text = replace(text, r"<.*?>" => "")
    text = replace(text, r"http\S+|www\S+" => "")
    text = replace(text, r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" => "")
    text = replace(text, r"\n|\r|\t" => " ")
    text = replace(text, r"\s+" => " ")
    text = replace(text, r"[^\w\s]" => " ")
    text = replace(text, r"\b\w*\d+\w*\b" => "")
    text = replace(text, r"\s+" => " ")
    text = strip(text)
    return text
end

"""
Remove stopwords from text
"""
function remove_stopwords(text::AbstractString)
    stopwords = Set([
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", 
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", 
        "was", "were", "will", "with", "the", "this", "but", "they", "have", 
        "had", "what", "said", "each", "which", "she", "do", "how", "their", 
        "if", "up", "out", "many", "then", "them", "these", "so", "some", "her", 
        "would", "make", "like", "into", "him", "time", "two", "more", "go", 
        "no", "way", "could", "my", "than", "first", "water", "been", "call", 
        "who", "oil", "its", "now", "find", "long", "down", "day", "did", 
        "get", "come", "made", "may", "part"
    ])
    
    words = split(text)
    filtered_words = [word for word in words if !(word in stopwords)]
    return join(filtered_words, " ")
end

"""
Remove single character words
"""
function remove_single_chars(text::AbstractString)
    words = split(text)
    filtered_words = [word for word in words if length(word) > 1]
    return join(filtered_words, " ")
end

"""
Complete preprocessing pipeline
"""
function full_preprocessing(text::AbstractString)
    text = String(text)  # Convert to String type
    text = preprocess_text(text)
    text = remove_stopwords(text)
    text = remove_single_chars(text)
    return text
end

# =============================================
# BERT-LIKE MODEL ARCHITECTURE WITH <1000 PARAMETERS
# =============================================

"""
Positional encoding for sequence position information
"""
function create_positional_encoding(max_seq_len::Int, d_model::Int)
    pe = zeros(Float32, max_seq_len, d_model)
    
    for pos in 1:max_seq_len
        for i in 1:2:d_model
            pe[pos, i] = sin(pos / (10000^((i-1)/d_model)))
            if i < d_model
                pe[pos, i+1] = cos(pos / (10000^((i-1)/d_model)))
            end
        end
    end
    
    return pe
end

"""
Lightweight attention mechanism
"""
struct LightweightAttention
    query_proj::Dense
    key_proj::Dense
    value_proj::Dense
    output_proj::Dense
    dropout::Dropout
end

function LightweightAttention(d_model::Int, dropout_rate::Float32=0.1f0)
    # Use smaller projection dimensions to save parameters
    proj_dim = max(1, d_model ÷ 2)
    
    return LightweightAttention(
        Dense(d_model, proj_dim, bias=false), 
        Dense(d_model, proj_dim, bias=false),
        Dense(d_model, proj_dim, bias=false),
        Dense(proj_dim, d_model, bias=false),
        Dropout(dropout_rate)
    )
end

Flux.@functor LightweightAttention

function (attn::LightweightAttention)(x)
    batch_size, seq_len, d_model = size(x)
    
    x_flat = reshape(x, batch_size * seq_len, d_model)
    q = attn.query_proj(x_flat')
    k = attn.key_proj(x_flat')
    v = attn.value_proj(x_flat')
    
    proj_dim = size(q, 1)
    q = reshape(q', batch_size, seq_len, proj_dim)
    k = reshape(k', batch_size, seq_len, proj_dim)
    v = reshape(v', batch_size, seq_len, proj_dim)
    
    # This is a very simplified attention mechanism
    pooled_q = mean(q, dims=2)
    pooled_k = mean(k, dims=2)
    pooled_v = mean(v, dims=2)
    
    attention_score = sum(pooled_q .* pooled_k, dims=3)
    attention_weight = sigmoid.(attention_score)
    
    attended_output = attention_weight .* pooled_v
    
    output = repeat(attended_output, 1, seq_len, 1)
    
    output_flat = reshape(output, batch_size * seq_len, proj_dim)
    final_output = attn.output_proj(output_flat')
    final_output = reshape(final_output', batch_size, seq_len, d_model)
    
    return attn.dropout(final_output)
end

"""
BERT-like model with proper transformer components under 1000 parameters
"""
struct BERTLike
    embedding::Embedding
    pos_encoding::Matrix{Float32}  # Fixed positional encoding (not learned)
    attention::LightweightAttention
    attention_norm::LayerNorm
    feedforward::Dense
    ff_norm::LayerNorm
    classifier::Dense
    dropout::Dropout
    cls_token::Vector{Float32}
    class_bias::Vector{Float32}
end

function BERTLike(vocab_size::Int, d_model::Int=8, max_seq_len::Int=20, num_classes::Int=2)
    ff_dim = 4
    
    # Create positional encoding (not learned parameters)
    pos_enc = create_positional_encoding(max_seq_len + 1, d_model)  # +1 for CLS
    
    # CLS token embedding
    cls_token = randn(Float32, d_model) * 0.1f0
    
    # Strong class bias for imbalanced data
    class_bias = Float32[-5.0, 5.0]
    
    return BERTLike(
        Embedding(vocab_size, d_model),
        pos_enc,
        LightweightAttention(d_model, 0.1f0),
        LayerNorm(d_model),
        Dense(d_model, ff_dim, relu),
        LayerNorm(ff_dim),
        Dense(d_model, num_classes),
        Dropout(0.2f0),
        cls_token,
        class_bias
    )
end

Flux.@functor BERTLike

function (model::BERTLike)(x)
    seq_len, batch_size = size(x)
    d_model = length(model.cls_token)
    
    x = transpose(x)
    batch_size, seq_len = size(x)
    
    embedded = model.embedding(x)
    embedded = permutedims(embedded, (2, 3, 1))
    
    cls_tokens = repeat(reshape(model.cls_token, 1, 1, d_model), batch_size, 1, 1)
    embedded = cat(cls_tokens, embedded, dims=2)
    
    pos_enc = model.pos_encoding[1:seq_len+1, :]
    embedded = embedded .+ reshape(pos_enc, 1, seq_len+1, d_model)
    
    embedded = model.dropout(embedded)
    
    # Self-attention with residual connection
    attn_output = model.attention(embedded)
    embedded = embedded + attn_output
    
    embedded_flat = reshape(embedded, batch_size * (seq_len+1), d_model)
    normed_flat = model.attention_norm(embedded_flat')
    normed_flat = normed_flat'
    embedded = reshape(normed_flat, batch_size, seq_len+1, d_model)
    
    cls_output = embedded[:, 1, :]
    
    ff_output = model.feedforward(cls_output')
    ff_output = ff_output'
    
    ff_normed = model.ff_norm(ff_output')
    ff_normed = ff_normed'
    
    logits = model.classifier(cls_output')
    logits = logits'
    
    logits = logits .+ reshape(model.class_bias, 1, 2)
    
    temperature = 0.7f0
    logits = logits ./ temperature
    
    return logits
end

function count_parameters(model::BERTLike)
    total = 0
    for p in Flux.params(model)
        total += length(p)
    end
    return total
end

"""
Build vocabulary from the full dataset
"""
function build_vocabulary(texts::Vector{String}, max_vocab_size::Int=52)
    word_counts = Dict{String, Int}()
    
    for text in texts
        words = split(text)
        for word in words
            word_counts[word] = get(word_counts, word, 0) + 1
        end
    end
    
    sorted_words = sort(collect(word_counts), by=x->x[2], rev=true)
    
    # Create vocabulary (reserve indices for special tokens)
    vocab = Dict{String, Int}()
    vocab["<PAD>"] = 1
    vocab["<UNK>"] = 2
    
    idx = 3
    for (word, count) in sorted_words
        if idx > max_vocab_size
            break
        end
        vocab[word] = idx
        idx += 1
    end
    
    return vocab
end

function text_to_indices(text::AbstractString, vocab::Dict{String, Int}, max_len::Int=20)
    words = split(text)
    indices = Int[]
    
    for word in words
        if length(indices) >= max_len
            break
        end
        push!(indices, get(vocab, word, vocab["<UNK>"]))
    end
    
    # Pad or truncate to max_len
    while length(indices) < max_len
        push!(indices, vocab["<PAD>"])
    end
    
    return indices[1:max_len]
end

# =============================================
# TRAINING AND EVALUATION WITH IMPROVEMENTS
# =============================================

function accuracy(ŷ, y)
    return mean(Flux.onecold(ŷ, 1:2) .== Flux.onecold(y, 1:2))
end

"""
K-fold cross validation for model evaluation
"""
function k_fold_validation(X, y, vocab_size::Int, max_seq_len::Int, k::Int=10, epochs::Int=20)
    n_samples = size(X, 1)
    fold_size = n_samples ÷ k
    
    fold_results = []
    
    println("Starting $k-fold cross validation...")
    
    for fold in 1:k
        println("\n=== FOLD $fold/$k ===")
        
        start_idx = (fold - 1) * fold_size + 1
        end_idx = fold == k ? n_samples : fold * fold_size
        
        test_indices = start_idx:end_idx
        train_indices = vcat(1:(start_idx-1), (end_idx+1):n_samples)
        
        X_train_fold = X[train_indices, :]
        y_train_fold = y[:, train_indices]
        X_test_fold = X[test_indices, :]
        y_test_fold = y[:, test_indices]
        
        n_train = size(X_train_fold, 1)
        val_size = round(Int, 0.2 * n_train)
        train_size = n_train - val_size
        
        val_indices = 1:val_size
        train_only_indices = (val_size+1):n_train
        
        X_train_only = X_train_fold[train_only_indices, :]
        y_train_only = y_train_fold[:, train_only_indices]
        X_val_fold = X_train_fold[val_indices, :]
        y_val_fold = y_train_fold[:, val_indices]
        
        batch_size = 32
        train_loader = Flux.DataLoader((transpose(X_train_only), y_train_only), batchsize=batch_size, shuffle=true)
        val_loader = Flux.DataLoader((transpose(X_val_fold), y_val_fold), batchsize=batch_size, shuffle=false)
        test_loader = Flux.DataLoader((transpose(X_test_fold), y_test_fold), batchsize=batch_size, shuffle=false)
        
        model = BERTLike(vocab_size, 8, max_seq_len, 2)
        trained_model, _, _, _, _ = train_model(model, train_loader, val_loader, epochs)
        
        test_acc, test_precision, test_recall, test_f1 = evaluate_model(trained_model, test_loader)
        
        fold_result = Dict(
            "fold" => fold,
            "accuracy" => test_acc,
            "precision" => test_precision,
            "recall" => test_recall,
            "f1" => test_f1
        )
        push!(fold_results, fold_result)
    end
    
    avg_accuracy = mean([result["accuracy"] for result in fold_results])
    avg_precision = mean([result["precision"] for result in fold_results])
    avg_recall = mean([result["recall"] for result in fold_results])
    avg_f1 = mean([result["f1"] for result in fold_results])
    
    std_accuracy = std([result["accuracy"] for result in fold_results])
    std_precision = std([result["precision"] for result in fold_results])
    std_recall = std([result["recall"] for result in fold_results])
    std_f1 = std([result["f1"] for result in fold_results])
    
    println("\n" * "="^60)
    println("K-FOLD CROSS VALIDATION RESULTS ($k folds)")
    println("="^60)
    println("Average Accuracy:  $(round(avg_accuracy, digits=3)) ± $(round(std_accuracy, digits=3))")
    println("Average Precision: $(round(avg_precision, digits=3)) ± $(round(std_precision, digits=3))")
    println("Average Recall:    $(round(avg_recall, digits=3)) ± $(round(std_recall, digits=3))")
    println("Average F1 Score:  $(round(avg_f1, digits=3)) ± $(round(std_f1, digits=3))")
    println("="^60)
    
    return fold_results
end

function classification_metrics(ŷ, y)
    y_pred = Flux.onecold(ŷ, 1:2)
    y_true = Flux.onecold(y, 1:2)
    
    # For binary classification (1: Ham, 2: Spam)
    tp = sum((y_pred .== 2) .& (y_true .== 2))
    fp = sum((y_pred .== 2) .& (y_true .== 1))
    fn = sum((y_pred .== 1) .& (y_true .== 2))
    tn = sum((y_pred .== 1) .& (y_true .== 1))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return precision, recall, f1, tp, fp, fn, tn
end

"""
Improved training with class balancing and better monitoring
"""
function train_model(model, train_loader, val_loader, epochs::Int=30)
    # Use higher learning rate to escape local minima
    optimizer = Flux.setup(Adam(0.01), model)
    
    # Calculate class weights for balanced training
    ham_count = 0
    spam_count = 0
    for (x_batch, y_batch) in train_loader
        ham_count += sum(Flux.onecold(y_batch, 1:2) .== 1)
        spam_count += sum(Flux.onecold(y_batch, 1:2) .== 2)
    end
    
    total_samples = ham_count + spam_count
    ham_weight = total_samples / (2 * ham_count)
    spam_weight = total_samples / (2 * spam_count)
    class_weights = [ham_weight, spam_weight]
    
    # Early stopping parameters
    best_val_loss = Inf
    patience = 5
    patience_counter = 0
    best_model_state = nothing
    
    train_losses = Float32[]
    val_losses = Float32[]
    train_accuracies = Float32[]
    val_accuracies = Float32[]
    
    println("Starting training for $(epochs) epochs...")
    
    for epoch in 1:epochs
        train_loss = 0.0f0
        train_acc = 0.0f0
        train_batches = 0
        
        for (x_batch, y_batch) in train_loader
            loss, grads = Flux.withgradient(model) do m
                ŷ = m(x_batch)
                y_classes = Flux.onecold(y_batch, 1:2)
                batch_weights = [class_weights[class_idx] for class_idx in y_classes]
                base_loss = Flux.logitcrossentropy(transpose(ŷ), y_batch)
                weighted_loss = base_loss * mean(batch_weights)
                weighted_loss
            end
            
            Flux.update!(optimizer, model, grads[1])
            
            train_loss += loss
            ŷ = model(x_batch)
            train_acc += accuracy(transpose(ŷ), y_batch)
            train_batches += 1
        end
        
        val_loss = 0.0f0
        for (x_batch, y_batch) in val_loader
            ŷ = model(x_batch)
            val_loss += Flux.logitcrossentropy(transpose(ŷ), y_batch)
        end
        
        avg_val_loss = val_loss / length(val_loader)
        
        # Early stopping
        if avg_val_loss < best_val_loss
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = Flux.state(model)
        else
            patience_counter += 1
            if patience_counter >= patience
                println("Early stopping at epoch $epoch due to validation loss.")
                break
            end
        end
    end
    
    # Load best model
    if best_model_state !== nothing
        Flux.loadmodel!(model, best_model_state)
    end
    
    # Return empty arrays for curves as they are not used
    return model, [], [], [], []
end

function evaluate_model(model, test_loader)
    all_predictions = []
    all_targets = []
    
    for (x_batch, y_batch) in test_loader
        ŷ = model(x_batch)
        append!(all_predictions, Flux.onecold(transpose(ŷ), 1:2))
        append!(all_targets, Flux.onecold(y_batch, 1:2))
    end
    
    precision, recall, f1, tp, fp, fn, tn = classification_metrics(
        OneHotArrays.onehotbatch(all_predictions, 1:2),
        OneHotArrays.onehotbatch(all_targets, 1:2)
    )
    
    acc = accuracy(
        OneHotArrays.onehotbatch(all_predictions, 1:2),
        OneHotArrays.onehotbatch(all_targets, 1:2)
    )
    
    println("\n--- Test Results ---")
    println("Accuracy: $(round(acc, digits=3))")
    println("Precision: $(round(precision, digits=3))")
    println("Recall: $(round(recall, digits=3))")
    println("F1 Score: $(round(f1, digits=3))")
    println("Confusion Matrix:")
    println("                Predicted")
    println("                Ham    Spam")
    println("Actual Ham      $tn     $fp")
    println("Actual Spam     $fn     $tp")
    
    return acc, precision, recall, f1
end

# =============================================
# MAIN EXECUTION
# =============================================

function main(use_kfold::Bool=true, k_folds::Int=10)
    println("--- Spam Detection Model Execution ---")
    
    if use_kfold
        println("Mode: $k_folds-fold Cross Validation")
    else
        println("Mode: Standard Train/Validation/Test Split")
    end
    
    # Load data
    println("\n1. Loading data...")
    df = CSV.read("spam_Emails_data.csv", DataFrame)
    println("Loaded $(nrow(df)) emails.")
    
    # Preprocess data
    println("\n2. Preprocessing data...")
    valid_rows = .!ismissing.(df.text) .& .!ismissing.(df.label)
    df_clean = df[valid_rows, :]
    
    processed_texts = [full_preprocessing(text) for text in df_clean.text]
    labels = [label == "Spam" ? 2 : 1 for label in df_clean.label]
    
    valid_indices = findall(x -> length(x) > 0, processed_texts)
    processed_texts = processed_texts[valid_indices]
    labels = labels[valid_indices]
    println("Data cleaned and preprocessed.")
    
    # Build vocabulary
    println("\n3. Building vocabulary...")
    vocab = build_vocabulary(processed_texts, 52)
    println("Vocabulary built with size $(length(vocab)).")
    
    # Convert texts to indices
    println("\n4. Converting texts to token indices...")
    max_seq_len = 20
    X = hcat([text_to_indices(text, vocab, max_seq_len) for text in processed_texts]...)
    X = permutedims(X, (2, 1))
    
    y = OneHotArrays.onehotbatch(labels, 1:2)
    println("Texts converted to token indices.")
    
    if use_kfold
        println("\n5. Performing k-fold cross validation...")
        n_samples = size(X, 1)
        indices = randperm(n_samples)
        X_shuffled = X[indices, :]
        y_shuffled = y[:, indices]
        
        k_fold_validation(
            X_shuffled, y_shuffled, length(vocab), max_seq_len, k_folds, 20
        )
        
        # Train final model on full dataset after k-fold validation
        println("\n6. Training final model on full dataset...")
        batch_size = 32
        X_full_t = transpose(X_shuffled)
        full_loader = Flux.DataLoader((X_full_t, y_shuffled), batchsize=batch_size, shuffle=true)
        
        final_model = BERTLike(length(vocab), 8, max_seq_len, 2)
        param_count = count_parameters(final_model)
        println("Final model created with $param_count parameters.")
        
        # Train on full dataset
        trained_final_model, _, _, _, _ = train_model(
            final_model, full_loader, full_loader, 20  # Using same data for train/val since we already did k-fold
        )
        
        println("\n7. Saving trained model...")
        BSON.@save "trained_model.bson" trained_final_model vocab max_seq_len
        println("Model saved as 'trained_model.bson'.")
    else
        println("\n5. Splitting data (4:3:3 ratio)...")
        n_samples = size(X, 1)
        indices = randperm(n_samples)
        
        train_size = round(Int, 0.4 * n_samples)
        val_size = round(Int, 0.3 * n_samples)
        
        train_idx = indices[1:train_size]
        val_idx = indices[train_size+1:train_size+val_size]
        test_idx = indices[train_size+val_size+1:end]
        
        X_train, y_train = X[train_idx, :], y[:, train_idx]
        X_val, y_val = X[val_idx, :], y[:, val_idx]
        X_test, y_test = X[test_idx, :], y[:, test_idx]
        
        println("Data split into training, validation, and test sets.")
        
        batch_size = 32
        X_train_t = transpose(X_train)
        X_val_t = transpose(X_val)
        X_test_t = transpose(X_test)
        
        train_loader = Flux.DataLoader((X_train_t, y_train), batchsize=batch_size, shuffle=true)
        val_loader = Flux.DataLoader((X_val_t, y_val), batchsize=batch_size, shuffle=false)
        test_loader = Flux.DataLoader((X_test_t, y_test), batchsize=batch_size, shuffle=false)
    
        println("\n6. Creating and training model...")
        model = BERTLike(length(vocab), 8, max_seq_len, 2)
        param_count = count_parameters(model)
        println("Model created with $param_count parameters.")
        
        trained_model, _, _, _, _ = train_model(
            model, train_loader, val_loader, 30
        )
        
        println("\n7. Evaluating on test set...")
        evaluate_model(trained_model, test_loader)
        
        println("\n8. Saving trained model...")
        BSON.@save "trained_model.bson" trained_model vocab max_seq_len
        println("Model saved as 'trained_model.bson'.")
    end
end

# Run the main function
if abspath(PROGRAM_FILE) == @__FILE__
    # To run with standard train/test split: main(false)
    # To run with k-fold cross validation: main(true, 10)
    main(true, 10)  # Now k-fold also saves the model
    
    println("\n--- Execution Complete ---")
end
