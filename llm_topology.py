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


  def print_components(self, input_sentence, target_tokens, layer=-1, persistence_threshold=0.27):
    """
    Returns lists of words in each connected component based on topological analysis
    """
    embeddings = self.get_output_embeddings(input_sentence, target_tokens, layer)
    distance_matrix = self.compute_distance_matrix(embeddings)

    # Get persistent homology
    diagrams = ripser(distance_matrix, distance_matrix=True, thresh=persistence_threshold, maxdim=1)
    h0_features = diagrams['dgms'][0]
    
    # Find alive components (those that persist beyond threshold)
    alive_components = [(birth, death) for birth, death in h0_features 
                       if birth <= persistence_threshold and 
                       (death > persistence_threshold or np.isinf(death))]
    
    # Use hierarchical clustering to identify components
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    # Clean up distance matrix for scipy (handle negative values and ensure symmetry)
    distance_matrix = np.maximum(distance_matrix, 0)  # Remove negative values
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Ensure symmetry
    np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is zero
    
    # Convert distance matrix to condensed form for scipy
    condensed_distances = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distances, method='single')
    
    # Cut the dendrogram at the persistence threshold to get clusters
    cluster_labels = fcluster(linkage_matrix, persistence_threshold, criterion='distance')
    
    # Group words by cluster
    components = {}
    for i, (word, cluster_id) in enumerate(zip(target_tokens, cluster_labels)):
        if cluster_id not in components:
            components[cluster_id] = []
        components[cluster_id].append(word)
    
    # Sort components by size (largest first)
    sorted_components = sorted(components.values(), key=len, reverse=True)
    
    print(f"\nFound {len(sorted_components)} connected components:")
    print(f"Alive components (persistent): {len(alive_components)}")
    
    for i, component in enumerate(sorted_components, 1):
        print(f"Component {i} ({len(component)} words): {component}")
    
    return sorted_components

  def output_distance_mean(self, prompt, words):
    output_embeddings = self.get_output_embeddings(prompt, words, layer=-1)
    input_embeddings = self.get_output_embeddings(prompt, words, layer=1)
    norm_output_embed = F.normalize(output_embeddings, p = 2, dim = 1)
    norm_input_embed = F.normalize(input_embeddings, p = 2, dim = 1)

    cosine_similarities = torch.sum(norm_output_embed * norm_input_embed, dim=1).tolist()

    distances = [1 - sim for sim in cosine_similarities]

    return np.mean(distances)

  def output_distance_cumulative(self, prompt, words, start_layer=0, end_layer=-1):
    """
    Compute sum of cosine distances between consecutive layers across the entire model
    """
    # Determine the actual layer range
    if end_layer == -1:
        # Assuming your model has around 32 layers (adjust based on your model)
        total_layers = 64  # Change this to match your model's layer count
        end_layer = total_layers - 1
    
    if start_layer < 0:
        start_layer = total_layers + start_layer
    
    total_distance = 0.0
    
    # Iterate through consecutive layer pairs
    for layer_idx in range(start_layer, end_layer):
        current_layer = layer_idx
        next_layer = layer_idx + 1
        
        # Get embeddings from consecutive layers
        current_embeddings = self.get_output_embeddings(prompt, words, layer=current_layer)
        next_embeddings = self.get_output_embeddings(prompt, words, layer=next_layer)
        
        # Normalize embeddings
        norm_current = F.normalize(current_embeddings, p=2, dim=1)
        norm_next = F.normalize(next_embeddings, p=2, dim=1)
        
        # Compute cosine similarities
        cosine_similarities = torch.sum(norm_current * norm_next, dim=1)
        
        # Convert to distances and take mean for this layer transition
        distances = 1 - cosine_similarities
        layer_mean_distance = torch.mean(distances).item()
        
        # Add to total
        total_distance += layer_mean_distance
    
    return total_distance

  def component_layer_test(self, question, words, layers):
    i = 1
    comp = []
    for l in layers:
      print(f"Testing layer {l}")
      comp.append(self.out_components(question, words, layer=l))
      i = i+1
    print(f"Components per layer for question {question}: {comp}")
    return comp

  def layertest(self, questions, words, layers):
    for q in questions:
      self.component_layer_test(q, words, layers)

  def truthtest(self, true_false_statements, truth_list, words, layer=-1):
    difference_list = []
    for i in range(0, len(true_false_statements)):
      question = "Is the following statement true or false? " + true_false_statements[i]
      question_true = question + " True."
      question_false = question + " False."
      truth_value = truth_list[i]
      print(f"Testing question: {question} ({truth_value})")
      
      true_components = self.out_components(question_true, words, layer=layer)
      print(f"Components when 'True' appended: {true_components}")
      
      false_components = self.out_components(question_false, words, layer=layer)
      print(f"Components when 'False' appended: {false_components}")
      
      if truth_value == False:
        correct_comp = false_components
        incorrect_comp = true_components

      if truth_value == True:
        correct_comp = true_components
        incorrect_comp = false_components

      difference = correct_comp - incorrect_comp
      difference_list.append(difference)

      print(f"Components of correct answer - incorrect answer: {difference}")
      if i%25 == 0:
        print(f"Current differences: {difference_list}")


  def clustertest(self, input, tokens, layer=-1):
    print(f"Input sentence: {input}")
    print("Scores:")
    if isinstance(tokens, str):
        tokens = [tokens]
    
    embeddings = []
    
    with torch.no_grad():
        for token in tokens:
            # Create context with each target token
            full_text = input + " " + token
            
            # Tokenize 
            inputs = self.tokenizer(full_text, return_tensors="pt", 
                                   padding=True, truncation=True).to(self.device)
            
            # Get hidden states from specified layer
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer]
            
            # Get embedding for the target token position (last token)
            token_embedding = hidden_states[0, -1, :]  # [hidden_dim]
            embeddings.append(token_embedding)

        embeddings = torch.stack(embeddings).cpu()

    with torch.no_grad():
        embeddings = embeddings.float()

        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        cosine_sim = torch.mm(embeddings_norm, embeddings_norm.t())

    scores = []
    epsilon = 1e-10

    for i in range(len(tokens)):
      score = 0.0
      for j in range(len(tokens)):
        if i != j:
          sim_value = cosine_sim[i, j].item()
          shifted_sim = sim_value + epsilon
          score -= np.log(shifted_sim)

      scores.append(score)

    for token, score in zip(tokens, scores):
      print(f"{token}: {score}")


  def full_layer_test(self):
    # load model already as self.model
    num_layers = self.model.config.num_hidden_layers
    layers = np.arange(-num_layers, 0)
    colors = ["Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Black", "White"]
    questions = [
    "What color is a cherry",
    "What color is the sky",
    "What color is grass", 
    "What color is the sun",
    "What color is a carrot",
    "What color is a grape",
    "What color is coal",
    "What color is snow"
]
    
    for q_idx, q in enumerate(questions):
        cherry_scores = []
        london_scores = []
        tokyo_scores = []
        copenhagen_scores = []
        ottawa_scores = []
        beijing_scores = []
        dublin_scores = []
        berlin_scores = []
        
        for l in layers:
            embeddings = []
            with torch.no_grad():
                for token in cities:
                    # Create context with each target token
                    full_text = q + " " + token  # Changed from 'input' to 'q'
                    
                    # Tokenize 
                    inputs = self.tokenizer(full_text, return_tensors="pt", 
                                           padding=True, truncation=True).to(self.device)
                    
                    # Get hidden states from specified layer
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[l]  # Changed from 'layer' to 'l'
                    
                    # Get embedding for the target token position (last token)
                    token_embedding = hidden_states[0, -1, :]  # [hidden_dim]
                    embeddings.append(token_embedding)
        
                embeddings = torch.stack(embeddings).cpu()
        
            with torch.no_grad():
                embeddings = embeddings.float()
        
                embeddings_norm = F.normalize(embeddings, p=2, dim=1)
                cosine_sim = torch.mm(embeddings_norm, embeddings_norm.t())
        
            scores = []
            epsilon = 1e-10
        
            for i in range(len(cities)):
                score = 0.0
                for j in range(len(cities)):
                    if i != j:
                        sim_value = cosine_sim[i, j].item()
                        shifted_sim = sim_value + epsilon
                        score -= np.log(shifted_sim)
        
                scores.append(score)
        
            paris_scores.append(scores[0])
            london_scores.append(scores[1])
            tokyo_scores.append(scores[2])
            copenhagen_scores.append(scores[3])
            ottawa_scores.append(scores[4])
            beijing_scores.append(scores[5])
            dublin_scores.append(scores[6])
            berlin_scores.append(scores[7])
            
        all_scores = [paris_scores, london_scores, tokyo_scores, copenhagen_scores, 
                     ottawa_scores, beijing_scores, dublin_scores, berlin_scores]
        
        score_dict = {}
        for city, scores in zip(cities, all_scores):
            score_dict[city] = [round(float(score), 3) for score in scores]
        
        # Print in the desired format
        print(f'"{q}": {{')
        for i, (city, scores) in enumerate(score_dict.items()):
            comma = "," if i < len(score_dict) - 1 else ""
            print(f'    "{city}": {scores}{comma}')
        
        # Add comma after closing brace for all questions except the last one
        if q_idx < len(questions) - 1:
            print('},')
        else:
            print('}')

    
    

