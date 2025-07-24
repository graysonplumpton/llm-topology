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

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import scipy.spatial.distance as distance

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

  def hopkins_out(self, input_sentence, target_tokens, layer=-1, n_samples=None):
    """Hopkins statistic for output embeddings - returns float"""
    embeddings = self.get_output_embeddings(input_sentence, target_tokens, layer)
    return self.hopkins_statistic(embeddings, n_samples)

  def hopkins_statistic(self, embeddings, n_samples=None, debug=False):
    """Hopkins statistic for clustering tendency - Simple robust implementation using sklearn"""
    if not torch.is_tensor(embeddings):
        embeddings = torch.tensor(embeddings, device=self.device)
    
    # Convert to numpy for robust calculation using sklearn
    embeddings_np = embeddings.cpu().numpy().astype(np.float32)
    
    if len(embeddings_np) < 2:
        if debug:
            print("Warning: Less than 2 embeddings, returning 0.5")
        return 0.5
    
    if n_samples is None:
        n_samples = min(int(0.1 * len(embeddings_np)), 50)
    
    # Ensure we have enough samples
    n_samples = min(n_samples, len(embeddings_np) - 1)
    
    if debug:
        print(f"Embeddings shape: {embeddings_np.shape}")
        print(f"Using {n_samples} samples")
    
    n_dims = embeddings_np.shape[1]
    
    # Random points in data space
    data_min, data_max = embeddings_np.min(axis=0), embeddings_np.max(axis=0)
    
    if debug:
        print(f"Data range: {data_min.mean():.6f} to {data_max.mean():.6f}")
    
    # Check if all embeddings are identical
    data_range = data_max - data_min
    if np.all(data_range < 1e-6):
        if debug:
            print("Warning: All embeddings are nearly identical, returning 0.5")
        return 0.5
    
    # Generate random points
    random_points = np.random.uniform(data_min, data_max, size=(n_samples, n_dims))
    
    # Sample points from actual data
    sample_indices = np.random.choice(len(embeddings_np), n_samples, replace=False)
    sample_points = embeddings_np[sample_indices]
    
    # Find nearest neighbors using sklearn
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=2).fit(embeddings_np)
    
    # Distances from random points to nearest data point
    u_distances, _ = nbrs.kneighbors(random_points)
    u_distances = u_distances[:, 0]  # First neighbor
    
    # Distances from sample points to nearest other data point
    w_distances, _ = nbrs.kneighbors(sample_points)
    w_distances = w_distances[:, 1]  # Second neighbor (exclude self)
    
    if debug:
        print(f"U distances: mean={u_distances.mean():.6f}, sum={np.sum(u_distances):.6f}")
        print(f"W distances: mean={w_distances.mean():.6f}, sum={np.sum(w_distances):.6f}")
    
    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)
    total_sum = u_sum + w_sum
    
    if debug:
        print(f"U sum: {u_sum:.6f}")
        print(f"W sum: {w_sum:.6f}")
        print(f"Total sum: {total_sum:.6f}")
    
    if total_sum == 0 or np.isnan(total_sum) or np.isinf(total_sum):
        if debug:
            print("Warning: Numerical issue with distance sums, returning 0.5")
        return 0.5
    
    hopkins = u_sum / total_sum
    
    if debug:
        print(f"Hopkins result: {hopkins:.6f}")
    
    # Check for nan/inf result
    if np.isnan(hopkins) or np.isinf(hopkins):
        if debug:
            print("Warning: Hopkins result is nan/inf, returning 0.5")
        return 0.5
    
    return float(hopkins)

  def component_test(self, questions, cot_questions, words):
    comp = []
    cot_comp = []
    i = 1
    print(f"Initial connected components: {self.out_components(questions[0], words, layer = 1, persistence_threshold = 0.3)}")
    print(f"Testing {len(questions)} questions on {len(words)} words")
    for q in questions:
      print(f"Testing question {i}")
      comp.append(self.out_components(q, words, persistence_threshold = 0.3))
      i = i+1
    print(f"Regular components: {comp}")
    i = 1
    for q in cot_questions:
      print(f"Testing chain of thought question {i}")
      cot_comp.append(self.out_components(q, words, persistence_threshold = 0.3))
      i = i+1
    print(f"Chain of thought components: {cot_comp}")
    difference = [a - b for a, b in zip(cot_comp, comp)]
    print(f"Difference between chain of thought and regular: {difference}")

  def prompt_components(self, prompt, persistence_threshold = 0.3):
    #Assume loaded model with audomodelforcausallm
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

    with torch.no_grad():
      outputs = self.model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=self.tokenizer.eos_token_id
      )

    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text = generated_text.split()

    return self.out_components(prompt, output_text, persistence_threshold = 0.3)
