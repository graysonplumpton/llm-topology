import torch
import torch.nn.functional as F
import numpy as np
from ripser import ripser
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr

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

  def get_output_embeddings(self, input_sentence, target_tokens, layer=-1):

    if isinstance(target_tokens, str):
        target_tokens = [target_tokens]
    
    embeddings = []
    
    with torch.no_grad():
        for token in target_tokens:
            # Create context with each target token
            full_text = input_sentence + " " + token
            
            # Tokenize 
            inputs = self.tokenizer(full_text, return_tensors="pt", 
                                   padding=True, truncation=True).to(self.device)
            
            # Get hidden states from specified layer
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer]
            
            # Get embedding for the target token position (last token)
            token_embedding = hidden_states[0, -1, :]  # [hidden_dim]
            embeddings.append(token_embedding)
    
    return torch.stack(embeddings).cpu()  # [num_tokens, hidden_dim]

  def sig_loops_out(self, input_sentence, target_tokens, layer=-1, persistence_threshold=0.25):
    embeddings = self.get_output_embeddings(input_sentence, target_tokens, layer)
    distance_matrix = self.compute_distance_matrix(embeddings)
    diagrams = ripser(distance_matrix, distance_matrix=True, thresh=persistence_threshold, maxdim=1)
    h1_features = diagrams['dgms'][1]
    significant_loops = [(birth, death) for birth, death in h1_features 
                           if death - birth > persistence_threshold]
    return len(significant_loops)

  def topology_out(self, input_sentence, target_tokens, layer=-1, persistence_threshold = 0.27):
    embeddings = self.get_output_embeddings(input_sentence, target_tokens, layer)

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

    print(f"\n Topological Analysis:")
    print(f"Connected components: {alive_components}")
    print(f"Total component births: {len(h0_features)}")
    print(f"Total loops: {len(h1_features)}")
    print(f"Significant loops: {len(significant_loops)}")

  def total_loops_out(self, input_sentence, target_tokens, layer=-1, persistence_threshold=0.27):
    embeddings = self.get_output_embeddings(input_sentence, target_tokens, layer)
    distance_matrix = self.compute_distance_matrix(embeddings)

    diagrams = ripser(distance_matrix, distance_matrix = True, thresh = persistence_threshold, maxdim = 1)

    h1_features = diagrams['dgms'][1]

    return len(h1_features)

  def out_components(self, input_sentence, target_tokens, layer=-1, persistence_threshold=0.27):
    embeddings = self.get_output_embeddings(input_sentence, target_tokens, layer)
    distance_matrix = self.compute_distance_matrix(embeddings)

    diagrams = ripser(distance_matrix, distance_matrix = True, thresh = persistence_threshold, maxdim = 1)

    h0_features = diagrams['dgms'][0]

    alive_components = sum(1 for birth, death in h0_features 
                      if birth <= persistence_threshold and 
                      (death > persistence_threshold or np.isinf(death)))

    return alive_components

  def internal_layer_topology(self, input_sentence, target_tokens, persistence_threshold=0.27):

    layers = list(range(-29, 0))  
    all_embeddings = []
    
    # Get embeddings for all tokens from each layer
    for layer in layers:
        layer_embeddings = self.get_output_embeddings(input_sentence, target_tokens, layer=layer)
        # layer_embeddings shape: [num_tokens, hidden_dim]
        
        # Add each token's embedding from this layer
        for i in range(layer_embeddings.shape[0]):
            all_embeddings.append(layer_embeddings[i])  # [hidden_dim]
    
    # Stack all embeddings: [num_tokens * num_layers, hidden_dim]
    trajectory_embeddings = torch.stack(all_embeddings)
    
    # Compute distance matrix between all representations
    distance_matrix = self.compute_distance_matrix(trajectory_embeddings)
    
    # Compute persistent homology
    diagrams = ripser(distance_matrix, distance_matrix=True, thresh=persistence_threshold, maxdim=2)
    
    h0_features = diagrams['dgms'][0]  # Connected components
    h1_features = diagrams['dgms'][1]  # Loops
    h2_features = diagrams['dgms'][2]  # Voids
    
    alive_components = sum(1 for birth, death in h0_features 
                          if birth <= persistence_threshold and 
                          (death > persistence_threshold or np.isinf(death)))
    
    significant_loops = [(birth, death) for birth, death in h1_features 
                           if death - birth > persistence_threshold]
    
    significant_voids = [(birth, death) for birth, death in h2_features 
                        if death - birth > persistence_threshold]
    
    print(f"\n Topological Analysis:")
    print(f"Connected components: {alive_components}")
    print(f"Total component births: {len(h0_features)}")
    print(f"Total loops: {len(h1_features)}")
    print(f"Significant loops: {len(significant_loops)}")
    print(f"Total voids: {len(h2_features)}")
    print(f"Significant voids: {len(significant_voids)}")
    
    return diagrams

  def apply_isomap_analysis(self, embeddings, n_neighbors=5, n_components=2, standardize=True):
    """
    Apply ISOMAP and return key metrics without plotting
    
    Returns:
        dict with embedding, metrics, and quality measures
    """
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings
        
    print(f"Original embedding shape: {embeddings_np.shape}")
    print(f"Original embedding stats:")
    print(f"  Mean: {embeddings_np.mean():.4f}")
    print(f"  Std: {embeddings_np.std():.4f}")
    print(f"  Min: {embeddings_np.min():.4f}")
    print(f"  Max: {embeddings_np.max():.4f}")
    
    # Optional standardization
    if standardize:
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_np)
        print(f"Applied standardization")
    else:
        embeddings_scaled = embeddings_np
        scaler = None
    
    # Apply ISOMAP
    try:
        isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
        isomap_embedding = isomap.fit_transform(embeddings_scaled)
        
        print(f"\nISOMAP Results:")
        print(f"  Target dimensions: {n_components}")
        print(f"  Neighbors used: {n_neighbors}")
        print(f"  Output shape: {isomap_embedding.shape}")
        
        # Compute quality metrics
        try:
            reconstruction_error = isomap.reconstruction_error()
            print(f"  Reconstruction error: {reconstruction_error:.6f}")
        except:
            reconstruction_error = None
            print(f"  Reconstruction error: Not available")
            
        # Analyze the reduced space
        print(f"\nReduced embedding stats:")
        print(f"  Mean: {isomap_embedding.mean():.4f}")
        print(f"  Std: {isomap_embedding.std():.4f}")
        print(f"  Min: {isomap_embedding.min():.4f}")
        print(f"  Max: {isomap_embedding.max():.4f}")
        
        # Compute distance preservation
        from sklearn.metrics import pairwise_distances
        
        # Original distances (sample if too large)
        if len(embeddings_scaled) > 100:
            indices = np.random.choice(len(embeddings_scaled), 100, replace=False)
            orig_sample = embeddings_scaled[indices]
            isomap_sample = isomap_embedding[indices]
        else:
            orig_sample = embeddings_scaled
            isomap_sample = isomap_embedding
            
        orig_distances = pairwise_distances(orig_sample)
        isomap_distances = pairwise_distances(isomap_sample)
        
        # Correlation between distance matrices
        from scipy.stats import pearsonr
        orig_flat = orig_distances[np.triu_indices_from(orig_distances, k=1)]
        isomap_flat = isomap_distances[np.triu_indices_from(isomap_distances, k=1)]
        
        if len(orig_flat) > 0:
            correlation, p_value = pearsonr(orig_flat, isomap_flat)
            print(f"  Distance correlation: {correlation:.4f} (p={p_value:.4f})")
        else:
            correlation = None
            
        return {
            'isomap_embedding': isomap_embedding,
            'isomap_model': isomap,
            'scaler': scaler,
            'reconstruction_error': reconstruction_error,
            'distance_correlation': correlation,
            'original_shape': embeddings_np.shape,
            'reduced_shape': isomap_embedding.shape,
            'success': True
        }
        
    except Exception as e:
        print(f"ISOMAP failed: {str(e)}")
        print("This often happens when:")
        print("- n_neighbors is too large for the dataset size")
        print("- The neighborhood graph becomes disconnected")
        print("- Not enough data points")
        
        return {
            'success': False,
            'error': str(e),
            'original_shape': embeddings_np.shape
        }

def isomap_text_analysis(self, texts, layer=-1, n_neighbors=5, n_components=2, 
                        persistence_threshold=0.27, compare_topology=True):
    """
    Complete ISOMAP analysis pipeline with detailed output (no plots)
    """
    print(f"=== ISOMAP Analysis (Layer {layer}) ===")
    print(f"Analyzing {len(texts) if isinstance(texts, list) else 1} texts")
    
    # Get embeddings
    embeddings = self.get_embeddings(texts, layer)
    
    # Apply ISOMAP
    isomap_results = self.apply_isomap_analysis(embeddings, n_neighbors, n_components)
    
    if not isomap_results['success']:
        return isomap_results
    
    isomap_embedding = isomap_results['isomap_embedding']
    
    if compare_topology:
        print(f"\n=== Topological Comparison ===")
        
        # Original space topology
        print(f"\n--- Original High-Dimensional Space ---")
        original_diagrams = self._topology_from_embeddings(embeddings, persistence_threshold)
        
        # ISOMAP space topology  
        print(f"\n--- ISOMAP Low-Dimensional Space ---")
        isomap_tensor = torch.tensor(isomap_embedding, dtype=torch.float32)
        isomap_diagrams = self._topology_from_embeddings(isomap_tensor, persistence_threshold)
        
        # Compare topological features
        orig_components = len(original_diagrams['dgms'][0])
        orig_loops = len(original_diagrams['dgms'][1])
        
        isomap_components = len(isomap_diagrams['dgms'][0])
        isomap_loops = len(isomap_diagrams['dgms'][1])
        
        print(f"\n--- Topology Preservation Summary ---")
        print(f"Connected components: {orig_components} → {isomap_components}")
        print(f"Loops: {orig_loops} → {isomap_loops}")
        
        if orig_loops > 0:
            loop_preservation = 1 - abs(orig_loops - isomap_loops) / orig_loops
            print(f"Loop preservation score: {loop_preservation:.3f}")
        
        isomap_results.update({
            'original_topology': original_diagrams,
            'isomap_topology': isomap_diagrams,
            'topology_comparison': {
                'components_change': isomap_components - orig_components,
                'loops_change': isomap_loops - orig_loops
            }
        })
    
    return isomap_results

def find_optimal_isomap_params(self, texts, layer=-1, max_neighbors=None):
    """
    Find optimal ISOMAP parameters automatically
    """
    embeddings = self.get_embeddings(texts, layer)
    n_samples = len(embeddings)
    
    if max_neighbors is None:
        max_neighbors = min(15, n_samples - 1)
    
    print(f"=== Finding Optimal ISOMAP Parameters ===")
    print(f"Testing n_neighbors from 3 to {max_neighbors}")
    
    best_params = None
    best_score = float('inf')
    results = []
    
    for n_neighbors in range(3, max_neighbors + 1):
        try:
            isomap_results = self.apply_isomap_analysis(
                embeddings, n_neighbors=n_neighbors, n_components=2
            )
            
            if isomap_results['success']:
                # Use reconstruction error as quality metric
                score = isomap_results['reconstruction_error']
                if score is not None and score < best_score:
                    best_score = score
                    best_params = n_neighbors
                
                results.append({
                    'n_neighbors': n_neighbors,
                    'reconstruction_error': score,
                    'distance_correlation': isomap_results['distance_correlation']
                })
                
                print(f"n_neighbors={n_neighbors}: recon_error={score:.6f}, "
                      f"dist_corr={isomap_results['distance_correlation']:.3f}")
            else:
                print(f"n_neighbors={n_neighbors}: FAILED - {isomap_results['error']}")
                
        except Exception as e:
            print(f"n_neighbors={n_neighbors}: ERROR - {str(e)}")
    
    if best_params:
        print(f"\n=== Optimal Parameters ===")
        print(f"Best n_neighbors: {best_params}")
        print(f"Best reconstruction error: {best_score:.6f}")
    else:
        print(f"\nNo successful ISOMAP configuration found!")
    
    return {
        'best_n_neighbors': best_params,
        'best_score': best_score,
        'all_results': results
    }

def analyze_manifold_quality(self, texts, layer=-1, n_neighbors=5):
    """
    Assess if the data lies on a meaningful manifold
    """
    print(f"=== Manifold Quality Assessment ===")
    
    embeddings = self.get_embeddings(texts, layer)
    
    # Compare ISOMAP with PCA
    from sklearn.decomposition import PCA
    
    # PCA (linear)
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(embeddings.cpu().numpy())
    pca_var_explained = pca.explained_variance_ratio_.sum()
    
    print(f"PCA results:")
    print(f"  Explained variance (2D): {pca_var_explained:.3f}")
    print(f"  Component 1: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"  Component 2: {pca.explained_variance_ratio_[1]:.3f}")
    
    # ISOMAP (nonlinear)
    isomap_results = self.apply_isomap_analysis(embeddings, n_neighbors, n_components=2)
    
    if isomap_results['success']:
        isomap_correlation = isomap_results['distance_correlation']
        
        print(f"\nManifold Assessment:")
        if isomap_correlation > 0.8:
            print(f"  Strong manifold structure detected (correlation: {isomap_correlation:.3f})")
            manifold_quality = "Strong"
        elif isomap_correlation > 0.6:
            print(f"  Moderate manifold structure (correlation: {isomap_correlation:.3f})")
            manifold_quality = "Moderate"
        else:
            print(f"  Weak manifold structure (correlation: {isomap_correlation:.3f})")
            manifold_quality = "Weak"
            
        # Check if nonlinear structure is meaningful
        if pca_var_explained > 0.9:
            print(f"  Data appears mostly linear (PCA explains {pca_var_explained:.1%})")
            print(f"  ISOMAP may not provide significant advantage")
        else:
            print(f"  Nonlinear structure likely present")
            print(f"  ISOMAP could reveal hidden patterns")
            
        return {
            'manifold_quality': manifold_quality,
            'isomap_correlation': isomap_correlation,
            'pca_variance_explained': pca_var_explained,
            'recommendation': 'use_isomap' if isomap_correlation > 0.6 and pca_var_explained < 0.9 else 'use_pca'
        }
    else:
        print(f"ISOMAP failed - data may not have clear manifold structure")
        return {'manifold_quality': 'Unknown', 'recommendation': 'use_pca'}

def token_manifold_analysis(self, texts, layer=-1, n_neighbors=5):
    """
    Analyze manifold structure of individual tokens
    """
    print(f"=== Token-Level Manifold Analysis ===")
    
    # Get token embeddings
    embeddings, tokens = self.get_token_embeddings(texts, layer)
    print(f"Analyzing {len(tokens)} tokens")
    
    if len(tokens) < n_neighbors + 1:
        print(f"Warning: Only {len(tokens)} tokens, reducing n_neighbors to {len(tokens)-1}")
        n_neighbors = max(2, len(tokens) - 1)
    
    # Apply manifold analysis
    isomap_results = self.apply_isomap_analysis(embeddings, n_neighbors, n_components=2)
    
    if isomap_results['success']:
        # Analyze token clustering in reduced space
        isomap_embedding = isomap_results['isomap_embedding']
        
        # Find most distant token pairs
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(isomap_embedding)
        
        # Most distant pair
        max_dist_idx = np.unravel_index(distances.argmax(), distances.shape)
        most_distant = (tokens[max_dist_idx[0]], tokens[max_dist_idx[1]], distances[max_dist_idx])
        
        # Most similar pair (excluding diagonal)
        np.fill_diagonal(distances, np.inf)
        min_dist_idx = np.unravel_index(distances.argmin(), distances.shape)
        most_similar = (tokens[min_dist_idx[0]], tokens[min_dist_idx[1]], distances[min_dist_idx])
        
        print(f"\nToken Relationships in Manifold Space:")
        print(f"  Most distant: '{most_distant[0]}' ↔ '{most_distant[1]}' (dist: {most_distant[2]:.3f})")
        print(f"  Most similar: '{most_similar[0]}' ↔ '{most_similar[1]}' (dist: {most_similar[2]:.3f})")
        
        # Token density analysis
        mean_dist = distances[distances != np.inf].mean()
        std_dist = distances[distances != np.inf].std()
        
        print(f"  Mean token distance: {mean_dist:.3f}")
        print(f"  Distance std: {std_dist:.3f}")
        print(f"  Density ratio: {std_dist/mean_dist:.3f}")
        
        return {
            'tokens': tokens,
            'manifold_results': isomap_results,
            'most_distant_tokens': most_distant,
            'most_similar_tokens': most_similar,
            'mean_distance': mean_dist,
            'distance_std': std_dist
        }
    else:
        print("Token manifold analysis failed")
        return {'success': False}
  
