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


  def compute_distance_matrix(self, embeddings, metric = "cosine"):
    with torch.no_grad():
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



  # Made my Claude, review: 
  def print_sig_loops(self, text, layer=-1, persistence_threshold=0.27, top_k=5):
        """Print which tokens contribute to significant loops using cocycles"""
        # Get token-level embeddings and tokens
        embeddings, tokens = self.get_token_embeddings(text, layer)
        distance_matrix = self.compute_distance_matrix(embeddings)
        
        # Compute persistence diagrams WITH cocycles
        diagrams = ripser(distance_matrix, distance_matrix=True, 
                         thresh=persistence_threshold, maxdim=1, do_cocycles=True)
        
        h1_features = diagrams['dgms'][1]  # Loops
        h1_cocycles = diagrams['cocycles'][1] if 'cocycles' in diagrams else []
        
        significant_loops = [(i, birth, death) for i, (birth, death) in enumerate(h1_features) 
                           if death - birth > persistence_threshold]
        
        if not significant_loops:
            print("No significant loops found.")
            return
        
        print(f"\nFound {len(significant_loops)} significant loops:")
        print("=" * 60)
        
        # Sort by persistence (death - birth)
        significant_loops.sort(key=lambda x: x[2] - x[1], reverse=True)
        
        for loop_idx, (orig_idx, birth, death) in enumerate(significant_loops[:top_k]):
            persistence = death - birth
            print(f"\nLoop {loop_idx + 1} (Persistence: {persistence:.4f}):")
            print(f"Birth: {birth:.4f}, Death: {death:.4f}")
            
            # Get the cocycle (representative cycle) for this loop
            if orig_idx < len(h1_cocycles):
                cocycle = h1_cocycles[orig_idx]
                
                print(f"  Loop formed by {len(cocycle)} edges:")
                
                # Extract the tokens involved in this specific loop
                loop_tokens = set()
                edge_details = []
                
                for edge_idx, coeff in cocycle:
                    # Each edge connects two vertices (tokens)
                    # We need to decode which tokens these are
                    # This is a bit tricky - we need to map edge indices back to vertex pairs
                    
                    # For now, let's collect all edges and show the coefficient
                    edge_details.append((edge_idx, coeff))
                    
                    # Try to find which token pair this edge represents
                    # This requires understanding Ripser's internal edge indexing
                    n_vertices = len(tokens)
                    if edge_idx < n_vertices * (n_vertices - 1) // 2:
                        # Convert edge index to vertex pair
                        i, j = self._edge_index_to_vertices(edge_idx, n_vertices)
                        loop_tokens.add(i)
                        loop_tokens.add(j)
                        
                        dist = distance_matrix[i, j]
                        print(f"    Edge {edge_idx}: '{tokens[i]}' ↔ '{tokens[j]}' (coeff: {coeff}, dist: {dist:.4f})")
                
                # Summary of tokens in this loop
                if loop_tokens:
                    print(f"  Tokens involved in this loop:")
                    for token_idx in sorted(loop_tokens):
                        print(f"    '{tokens[token_idx]}'")
            else:
                print("  (Cocycle information not available for this loop)")
    
    def _edge_index_to_vertices(self, edge_idx, n_vertices):
        """Convert Ripser's edge index back to vertex pair indices"""
        # Ripser uses upper triangular indexing
        # Edge index k corresponds to vertices (i,j) where i < j
        k = edge_idx
        i = 0
        while i + 1 < n_vertices:
            max_j_for_i = n_vertices - i - 1
            if k < max_j_for_i:
                j = i + 1 + k
                return i, j
            k -= max_j_for_i
            i += 1
        return 0, 1  # fallback
    

  
