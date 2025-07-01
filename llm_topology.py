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
            
    def print_sig_loops(self, text, layer=-1, persistence_threshold=0.27, top_k=5):
        """Print which tokens contribute to significant loops with complete loop structure"""
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
            
            # Find all edges that exist at the birth scale
            birth_edges = []
            for i in range(len(tokens)):
                for j in range(i + 1, len(tokens)):
                    dist = distance_matrix[i, j]
                    if dist <= birth + 0.001:  # Small tolerance for numerical precision
                        birth_edges.append((i, j, dist))
            
            print(f"  Total edges at birth scale: {len(birth_edges)}")
            
            # Build a graph from these edges
            from collections import defaultdict, deque
            
            graph = defaultdict(list)
            for i, j, dist in birth_edges:
                graph[i].append((j, dist))
                graph[j].append((i, dist))
            
            # Get the birth edge from cocycle (this triggered the loop)
            birth_edge_vertices = set()
            if orig_idx < len(h1_cocycles):
                cocycle = h1_cocycles[orig_idx]
                print(f"  Birth edge(s):")
                for edge_data in cocycle:
                    if len(edge_data) >= 3:
                        vertex1, vertex2, coeff = edge_data[0], edge_data[1], edge_data[2]
                        birth_edge_vertices.update([vertex1, vertex2])
                        if vertex1 < len(tokens) and vertex2 < len(tokens):
                            dist = distance_matrix[vertex1, vertex2]
                            print(f"    '{tokens[vertex1]}' ↔ '{tokens[vertex2]}' (dist: {dist:.4f})")
            
            # Find actual cycles in the graph using DFS
            def find_cycles_containing_vertices(graph, target_vertices, max_cycles=3):
                cycles = []
                visited_global = set()
                
                for start_vertex in target_vertices:
                    if start_vertex in visited_global:
                        continue
                        
                    # DFS to find cycles containing this vertex
                    def dfs_cycles(current, path, visited, start):
                        if len(path) > 10:  # Prevent very long cycles
                            return
                        
                        for neighbor, _ in graph[current]:
                            if neighbor == start and len(path) >= 3:
                                # Found a cycle back to start
                                cycle = path + [current]
                                if len(cycle) >= 3:
                                    cycles.append(cycle[:])
                                    visited_global.update(cycle)
                            elif neighbor not in visited and len(path) < 8:
                                visited.add(neighbor)
                                dfs_cycles(neighbor, path + [current], visited, start)
                                visited.remove(neighbor)
                    
                    if len(cycles) < max_cycles:
                        visited = {start_vertex}
                        dfs_cycles(start_vertex, [], visited, start_vertex)
                
                return cycles[:max_cycles]
            
            # Find cycles containing the birth edge vertices
            if birth_edge_vertices:
                cycles = find_cycles_containing_vertices(graph, birth_edge_vertices)
                
                if cycles:
                    print(f"  Found {len(cycles)} cycle(s) in the loop:")
                    for cycle_idx, cycle in enumerate(cycles):
                        print(f"    Cycle {cycle_idx + 1}: {len(cycle)} tokens")
                        cycle_tokens = [tokens[v] for v in cycle if v < len(tokens)]
                        print(f"      Path: {' → '.join(cycle_tokens[:8])}{'...' if len(cycle_tokens) > 8 else ''}")
                        
                        # Show distances along the cycle
                        print(f"      Distances:")
                        for i in range(len(cycle)):
                            v1, v2 = cycle[i], cycle[(i + 1) % len(cycle)]
                            if v1 < len(tokens) and v2 < len(tokens):
                                dist = distance_matrix[v1, v2]
                                print(f"        '{tokens[v1]}' → '{tokens[v2]}': {dist:.4f}")
                else:
                    print("  Could not trace complete cycle structure")
                    
                    # Fallback: show highly connected vertices
                    print("  Highly connected tokens at birth scale:")
                    vertex_connections = defaultdict(int)
                    for i, j, dist in birth_edges:
                        vertex_connections[i] += 1
                        vertex_connections[j] += 1
                    
                    sorted_vertices = sorted(vertex_connections.items(), 
                                           key=lambda x: x[1], reverse=True)
                    for vertex, conn_count in sorted_vertices[:10]:
                        if vertex < len(tokens):
                            print(f"    '{tokens[vertex]}': {conn_count} connections")
            else:
                print("  (Cocycle information not available for detailed analysis)")
    
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
    

  
