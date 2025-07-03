import torch
import torch.nn.functional as F
import numpy as np
from ripser import ripser
import matplotlib.pyplot as plt

# Already loaded: model_path, model, tokenizer

class LLMTopology:
  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer
    self.model.eval()
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.device = next(self.model.parameters()).device

  def get_embeddings(self, texts, layer=-1):
    if isinstance(texts, str):
        texts = [texts]
    
    with torch.no_grad():
        inputs = self.tokenizer(texts, return_tensors="pt", 
                               padding=True, truncation=True).to(self.device)
        
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer]
        
        
        attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
        embeddings = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
    
    return embeddings

  def get_token_embeddings(self, texts, layer=-1):
        """Get individual token embeddings (not mean-pooled)"""
        if isinstance(texts, str):
            texts = [texts]
        
        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", 
                                   padding=True, truncation=True).to(self.device)
            
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer]
            
            # Get tokens for reference
            tokens = []
            for i, input_ids in enumerate(inputs['input_ids']):
                text_tokens = []
                for j, token_id in enumerate(input_ids):
                    if inputs['attention_mask'][i][j] == 1:  # Only non-padded tokens
                        token_text = self.tokenizer.decode([token_id])
                        text_tokens.append(token_text)
                tokens.extend(text_tokens)
            
            # Flatten embeddings for all valid tokens
            embeddings_list = []
            for i, hidden in enumerate(hidden_states):
                for j, embedding in enumerate(hidden):
                    if inputs['attention_mask'][i][j] == 1:  # Only non-padded tokens
                        embeddings_list.append(embedding)
            
            embeddings = torch.stack(embeddings_list)
        
        return embeddings, tokens


  def compute_distance_matrix(self, embeddings, metric = "cosine"):
    with torch.no_grad():
        embeddings = embeddings.float()
      
        if metric == "cosine":
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            cosine_sim = torch.mm(embeddings_norm, embeddings_norm.t())
            distance_matrix = 1 - cosine_sim
            
        elif metric == "euclidean":
            distance_matrix = torch.cdist(embeddings, embeddings, p=2)

        else:
          raise ValueError(f"Unsupported metric: {metric}")


    return distance_matrix.cpu().numpy()

  def analyze_topology(self, text, layer=-1, persistence_threshold=0.27):
    embeddings = self.get_embeddings(text, layer)

    distance_matrix = self.compute_distance_matrix(embeddings)

    diagrams = ripser(distance_matrix, distance_matrix = True, thresh = persistence_threshold, maxdim = 1)

    h0_features = diagrams['dgms'][0]  # Connected components
    h1_features = diagrams['dgms'][1]  # Loops
    # h2_features = diagrams['dgms'][2]  # Voids

    alive_components = sum(1 for birth, death in h0_features 
                      if birth <= persistence_threshold and 
                      (death > persistence_threshold or np.isinf(death)))

    significant_loops = [(birth, death) for birth, death in h1_features 
                           if death - birth > persistence_threshold]


    print(f"Distance range: {distance_matrix.min():.4f} to {distance_matrix.max():.4f}")
    print(f"Mean distance: {distance_matrix.mean():.4f}")
    print(f"Std distance: {distance_matrix.std():.4f}")

    print(f"\n Topological Analysis:")
    print(f"Connected components: {alive_components}")
    print(f"Total component births: {len(h0_features)}")
    print(f"Total loops: {len(h1_features)}")
    print(f"Significant loops: {len(significant_loops)}")

    
    return diagrams

  def h2_features(self, text, layer=-1, persistence_threshold = 0.27):
    embeddings = self.get_embeddings(text, layer)

    distance_matrix = self.compute_distance_matrix(embeddings)

    diagrams = ripser(distance_matrix, distance_matrix = True, thresh = persistence_threshold, maxdim = 2)

    h2_features = diagrams['dgms'][2]

    significant_voids = [(birth, death) for birth, death in h2_features 
                        if death - birth > persistence_threshold]

    print(f"Total voids: {len(h2_features)}")
    print(f"Significant voids: {len(significant_voids)}")

    return diagrams

  def sig_loops(self, text, layer=-1, persistence_threshold = 0.27):
    embeddings = self.get_embeddings(text, layer)

    distance_matrix = self.compute_distance_matrix(embeddings)

    diagrams = ripser(distance_matrix, distance_matrix = True, thresh = persistence_threshold, maxdim = 1)

    h1_features = diagrams['dgms'][1]

    significant_loops = [(birth, death) for birth, death in h1_features 
                           if death - birth > persistence_threshold]

    return len(significant_loops)

  def sig_voids(self, text, layer=-1, persistence_threshold = 0.27):
    embeddings = self.get_embeddings(text, layer)

    distance_matrix = self.compute_distance_matrix(embeddings)

    diagrams = ripser(distance_matrix, distance_matrix = True, thresh = persistence_threshold, maxdim = 2)

    h2_features = diagrams['dgms'][2]

    significant_voids = [(birth, death) for birth, death in h2_features 
                        if death - birth > persistence_threshold]

    return len(significant_voids)



  def get_output_embeddings(self, text, include_logits=True, include_probabilities=True):
    """
    Get output embeddings from the model's final predictions.
    
    Args:
        text: Input text (string or list of strings)
        include_logits: Whether to return raw logits
        include_probabilities: Whether to return softmax probabilities
    
    Returns:
        Dictionary containing various output representations
    """
    
    if isinstance(text, str):
        text = [text]
    
    with torch.no_grad():
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", 
                               padding=True, truncation=True).to(self.device)
        
        # Get model outputs
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract different types of output embeddings
        results = {}
        
        # 1. Raw logits (model's final layer output before softmax)
        if include_logits:
            logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
            results['logits'] = logits.cpu().numpy()
        
        # 2. Probability distributions (softmax of logits)
        if include_probabilities:
            probabilities = F.softmax(outputs.logits, dim=-1)
            results['probabilities'] = probabilities.cpu().numpy()
        
        # 3. Top-k predictions for each position
        top_k = 5
        top_k_logits, top_k_indices = torch.topk(outputs.logits, top_k, dim=-1)
        results['top_k_logits'] = top_k_logits.cpu().numpy()
        results['top_k_indices'] = top_k_indices.cpu().numpy()
        
        # 4. Predicted tokens (argmax of logits)
        predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        results['predicted_token_ids'] = predicted_token_ids.cpu().numpy()
        
        # Decode predicted tokens
        predicted_tokens = []
        for batch_idx in range(predicted_token_ids.shape[0]):
            batch_tokens = []
            for pos_idx in range(predicted_token_ids.shape[1]):
                if inputs['attention_mask'][batch_idx][pos_idx] == 1:  # Only non-padded positions
                    token_id = predicted_token_ids[batch_idx][pos_idx].item()
                    token_text = self.tokenizer.decode([token_id])
                    batch_tokens.append(token_text)
            predicted_tokens.append(batch_tokens)
        
        results['predicted_tokens'] = predicted_tokens
        results['input_tokens'] = [self.tokenizer.decode(input_id) for input_id in inputs['input_ids'][0]]
        
        return results

  def analyze_output_topology(self, text, use_probabilities=True, top_k_only=True, 
                             persistence_threshold=0.01, top_holes=5):
    """
    Analyze topological structure of output embeddings.
    
    Args:
        text: Input text
        use_probabilities: Use probability distributions instead of raw logits
        top_k_only: Only use top-k predictions to reduce dimensionality
        persistence_threshold: Minimum persistence for significant holes
        top_holes: Number of top holes to analyze
    
    Returns:
        Tuple of (topological_result, output_data, hole_analysis)
    """
    
    print(f"Analyzing output embeddings for: '{text}'")
    print("="*60)
    
    # Get output embeddings
    output_data = self.get_output_embeddings(text)
    
    if use_probabilities:
        embeddings_data = output_data['probabilities'][0]  # First batch
        print(f"Using probability distributions (vocab_size: {embeddings_data.shape[-1]})")
    else:
        embeddings_data = output_data['logits'][0]  # First batch
        print(f"Using raw logits (vocab_size: {embeddings_data.shape[-1]})")
    
    # Handle high dimensionality
    if top_k_only:
        # Use only top-k predictions to reduce dimensionality
        top_k_data = output_data['top_k_logits'][0]  # Shape: [seq_len, k]
        embeddings_tensor = torch.tensor(top_k_data, dtype=torch.float32)
        print(f"Using top-5 predictions only, shape: {embeddings_tensor.shape}")
    else:
        # Use full vocabulary (this might be very slow due to high dimensionality)
        embeddings_tensor = torch.tensor(embeddings_data, dtype=torch.float32)
        print(f"Using full vocabulary, shape: {embeddings_tensor.shape}")
        if embeddings_tensor.shape[-1] > 10000:
            print("WARNING: High dimensionality may make analysis very slow!")
    
    # Compute distance matrix
    print("\nComputing distance matrix...")
    distance_matrix = self.compute_distance_matrix(embeddings_tensor)
    
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Distance range: {distance_matrix.min():.4f} to {distance_matrix.max():.4f}")
    
    # Run topological analysis
    result = ripser(distance_matrix, distance_matrix=True, maxdim=1, do_cocycles=True)
    
    # Analyze holes
    H1 = result['dgms'][1]
    print(f"\nFound {len(H1)} 1-dimensional features")
    
    # Find significant holes
    persistences = []
    for i, (birth, death) in enumerate(H1):
        if not np.isinf(death):
            persistence = death - birth
            if persistence >= persistence_threshold:
                persistences.append((i, birth, death, persistence))
    
    persistences.sort(key=lambda x: x[3], reverse=True)
    significant_holes = persistences[:top_holes]
    
    # Analyze hole details
    hole_analysis = []
    if significant_holes:
        print(f"\nTop {len(significant_holes)} significant holes:")
        cocycles = result['cocycles'][1] if 'cocycles' in result and len(result['cocycles']) > 1 else []
        
        for hole_num, (cocycle_idx, birth, death, persistence) in enumerate(significant_holes):
            print(f"\nHole {hole_num + 1}: persistence {persistence:.4f}")
            
            hole_info = {
                'hole_number': hole_num + 1,
                'persistence': persistence,
                'birth': birth,
                'death': death,
                'token_indices': [],
                'involved_tokens': [],
                'predicted_tokens': []
            }
            
            # Get tokens involved in this hole
            if cocycles and cocycle_idx < len(cocycles):
                cocycle = cocycles[cocycle_idx]
                token_indices = set()
                for edge in cocycle:
                    v1, v2, coeff = edge
                    token_indices.add(v1)
                    token_indices.add(v2)
                
                token_indices = sorted(list(token_indices))
                hole_info['token_indices'] = token_indices
                
                # Get input tokens at these positions
                involved_tokens = []
                predicted_tokens = []
                for idx in token_indices:
                    if idx < len(output_data['input_tokens']):
                        involved_tokens.append(output_data['input_tokens'][idx])
                    if idx < len(output_data['predicted_tokens'][0]):
                        predicted_tokens.append(output_data['predicted_tokens'][0][idx])
                
                hole_info['involved_tokens'] = involved_tokens
                hole_info['predicted_tokens'] = predicted_tokens
                
                print(f"  Input tokens: {involved_tokens}")
                print(f"  Predicted tokens: {predicted_tokens}")
            
            hole_analysis.append(hole_info)
    else:
        print("No significant holes found in output embeddings.")
        print("Try lowering the persistence_threshold.")
    
    return result, output_data, hole_analysis

  def compare_input_vs_output_topology(self, text, persistence_threshold=0.01):
    """
    Compare topological structure between input embeddings and output embeddings.
    
    Args:
        text: Input text
        persistence_threshold: Minimum persistence for significant holes
    
    Returns:
        Dictionary containing comparison results
    """
    # Analyze input embeddings (existing method)
    input_embeddings, input_tokens = self.get_token_embeddings(text)
    input_distance_matrix = self.compute_distance_matrix(input_embeddings)
    input_result = ripser(input_distance_matrix, distance_matrix=True, maxdim=1, do_cocycles=True)
    
    input_H1 = input_result['dgms'][1]
    input_significant = [h for h in input_H1 if not np.isinf(h[1]) and (h[1] - h[0]) >= persistence_threshold]
    print(f"Input significant holes: {len(input_significant)}")
    
    # Analyze output embeddings
    output_result, output_data, output_holes = self.analyze_output_topology(
        text, top_k_only=True, persistence_threshold=persistence_threshold
    )
    
    output_H1 = output_result['dgms'][1]
    output_significant = [h for h in output_H1 if not np.isinf(h[1]) and (h[1] - h[0]) >= persistence_threshold]
    print(f"Output significant holes: {len(output_significant)}")
    
    # Compare persistence distributions
    input_persistences = [h[1] - h[0] for h in input_significant]
    output_persistences = [h[1] - h[0] for h in output_significant]
    
    comparison = {
        'input_holes': len(input_significant),
        'output_holes': len(output_significant),
        'input_persistences': input_persistences,
        'output_persistences': output_persistences,
        'input_avg_persistence': np.mean(input_persistences) if input_persistences else 0,
        'output_avg_persistence': np.mean(output_persistences) if output_persistences else 0,
        'hole_difference': len(output_significant) - len(input_significant),
        'input_tokens': input_tokens,
        'output_hole_analysis': output_holes
    }
    
    # Print comparison summary
    print(f"\nCOMPARISON SUMMARY:")
    print(f"Input topology: {comparison['input_holes']} significant holes")
    print(f"Output topology: {comparison['output_holes']} significant holes")
    print(f"Difference: {comparison['hole_difference']} holes")
    print(f"Input avg persistence: {comparison['input_avg_persistence']:.4f}")
    print(f"Output avg persistence: {comparison['output_avg_persistence']:.4f}")
    
    return comparison

  def analyze_generation_topology(self, prompt, max_new_tokens=50, temperature=0.7):
    """
    Analyze how topology changes during text generation.
    
    Args:
        prompt: Starting prompt text
        max_new_tokens: Maximum tokens to generate
        temperature: Generation temperature
    
    Returns:
        List of topological analyses at each generation step
    """
    
    generation_analysis = []
    current_text = prompt
    
    # Generate tokens one by one and analyze topology at each step
    for step in range(max_new_tokens):
        print(f"\nGeneration step {step + 1}:")
        print(f"Current text: '{current_text}'")
        
        # Get output embeddings for current text
        output_data = self.get_output_embeddings(current_text)
        
        # Analyze output topology
        try:
            result, _, hole_analysis = self.analyze_output_topology(
                current_text, top_k_only=True, persistence_threshold=0.005, top_holes=3
            )
            
            step_analysis = {
                'step': step + 1,
                'text': current_text,
                'num_holes': len(hole_analysis),
                'hole_analysis': hole_analysis,
                'text_length': len(current_text.split())
            }
            
            generation_analysis.append(step_analysis)
            
            # Generate next token (simplified - you might want to use model.generate() instead)
            inputs = self.tokenizer(current_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                next_token_logits = outputs.logits[0, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1)
                next_token = self.tokenizer.decode(next_token_id)
            
            current_text += next_token
            
            # Stop if we hit an end token or the text becomes too long
            if next_token in ['<|endoftext|>', '</s>', '<|end|>'] or len(current_text) > 500:
                break
                
        except Exception as e:
            print(f"Error at step {step + 1}: {e}")
            break
    
    # Summary
    print(f"\nGENERATION TOPOLOGY SUMMARY:")
    print(f"Generated {len(generation_analysis)} steps")
    hole_counts = [step['num_holes'] for step in generation_analysis]
    print(f"Hole count range: {min(hole_counts)} to {max(hole_counts)}")
    print(f"Average holes per step: {np.mean(hole_counts):.2f}")
    
    return generation_analysis

  
