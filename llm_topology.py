import torch
import torch.nn.functional as F
import numpy as np
import json
from ripser import ripser
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
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
    
    # Define countries and their cities (1 capital + 7 other major cities)
    country_data = {
        "France": ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes", "Strasbourg", "Bordeaux"],
        "Germany": ["Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt", "Stuttgart", "Dusseldorf", "Leipzig"],
        "Italy": ["Rome", "Milan", "Naples", "Turin", "Florence", "Venice", "Bologna", "Genoa"],
        "Spain": ["Madrid", "Barcelona", "Valencia", "Seville", "Bilbao", "Malaga", "Zaragoza", "Murcia"],
        "United Kingdom": ["London", "Manchester", "Birmingham", "Glasgow", "Liverpool", "Edinburgh", "Bristol", "Leeds"],
        "Japan": ["Tokyo", "Osaka", "Kyoto", "Yokohama", "Nagoya", "Kobe", "Fukuoka", "Sapporo"],
        "Brazil": ["Brasilia", "Rio", "Paulo", "Salvador", "Fortaleza", "Recife", "Curitiba", "Manaus"],
        "Australia": ["Canberra", "Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Darwin", "Hobart"]
    }
    
    # Create questions
    questions = [f"The capital of {country} is (choose between {country_data[country]})" for country in country_data.keys()]
    
    results = {}
    
    for q_idx, (country, q) in enumerate(zip(country_data.keys(), questions)):
        cities = country_data[country]
        
        # Initialize score lists for each city
        city_scores = {city: [] for city in cities}
        
        for l in layers:
            embeddings = []
            with torch.no_grad():
                for city in cities:
                    # Create context with each target city
                    full_text = q + " " + city
                    
                    # Tokenize 
                    inputs = self.tokenizer(full_text, return_tensors="pt", 
                                           padding=True, truncation=True).to(self.device)
                    
                    # Get hidden states from specified layer
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[l]
                    
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
        
            # Assign scores to each city
            for city, score in zip(cities, scores):
                city_scores[city].append(score)
        
        # Format scores for output
        score_dict = {}
        for city, scores in city_scores.items():
            score_dict[city] = [round(float(score), 3) for score in scores]
        
        # Print in the desired format
        print(f'"{q}": {{')
        for i, (city, scores) in enumerate(score_dict.items()):
            comma = "," if i < len(score_dict) - 1 else ""
            # Mark the capital (first city in list) with a comment
            capital_marker = "  # capital" if i == 0 else ""
            print(f'    "{city}": {scores}{comma}{capital_marker}')
        
        # Add comma after closing brace for all questions except the last one
        if q_idx < len(questions) - 1:
            print('},')
        else:
            print('}')


  def promptcluster(self, prompt, score="cosine", print_full=False):
    """
    Compute clustering scores for tokens in a prompt using contextualized embeddings.
    """
    print(f"Input prompt: {prompt}")
    print(f"Scoring method: {score}")
    
    # Tokenize the prompt
    inputs = self.tokenizer(prompt, return_tensors="pt", 
                           padding=True, truncation=True).to(self.device)
    
    # Get token strings for display
    token_ids = inputs['input_ids'][0]
    tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
    
    with torch.no_grad():
        # Check if we need attention weights
        need_attention = (score == "entropy")
        
        # Get model outputs
        outputs = self.model(
            **inputs, 
            output_hidden_states=True, 
            output_attentions=need_attention
        )
        
        # Get embeddings from the last layer
        hidden_states = outputs.hidden_states[-1]
        embeddings = hidden_states[0]
        
        if score == "cosine":
            # [Previous cosine cluster code remains the same]
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            cosine_sim = torch.mm(embeddings_norm, embeddings_norm.t())
            
            epsilon = 1e-10
            individual_scores = []
            total_score = 0.0
            
            for i in range(len(tokens)):
                token_score = 0.0
                for j in range(len(tokens)):
                    if i != j:
                        sim_value = cosine_sim[i, j].item()
                        shifted_sim = (sim_value + 1) / 2 + epsilon
                        pairwise_score = -np.log(shifted_sim)
                        token_score += pairwise_score
                        if i < j:
                            total_score += pairwise_score
                individual_scores.append(token_score)
            
            final_score = total_score
            
        elif score == "entropy":
            if outputs.attentions is None:
                print("Warning: Attention weights not available.")
                return self.promptcluster(prompt, score="cosine", print_full=print_full)
            
            # Get attention weights - average across all layers and heads for richer signal
            # You can also just use the last layer: attentions = outputs.attentions[-1]
            all_attentions = torch.stack(outputs.attentions)  # [num_layers, batch_size, num_heads, seq_len, seq_len]
            avg_attention = all_attentions.mean(dim=(0, 2))[0]  # Average over layers and heads -> [seq_len, seq_len]
            
            individual_scores = []
            total_score = 0.0
            
            # Calculate attention entropy for each token
            for i in range(len(tokens)):
                token_entropy = 0.0
                
                # Entropy from this token attending to others (outgoing attention)
                for j in range(len(tokens)):
                    if avg_attention[i, j] > 0:
                        a_ij = avg_attention[i, j].item()
                        token_entropy += -a_ij * np.log(a_ij + 1e-10)
                
                # Entropy from other tokens attending to this one (incoming attention)
                for j in range(len(tokens)):
                    if i != j and avg_attention[j, i] > 0:
                        a_ji = avg_attention[j, i].item()
                        token_entropy += -a_ji * np.log(a_ji + 1e-10)
                
                individual_scores.append(token_entropy)
                total_score += token_entropy
            
            # Total score is sum of all individual entropies divided by 2 
            # (since we counted each interaction twice)
            final_score = total_score / 2
            
        elif score == "euclidean":
            # [Previous euclidean code remains the same]
            individual_scores = []
            total_score = 0.0
            
            for i in range(len(tokens)):
                token_score = 0.0
                for j in range(len(tokens)):
                    if i != j:
                        dist = torch.norm(embeddings[i] - embeddings[j], p=2).item()
                        token_score += dist
                        if i < j:
                            total_score += dist
                individual_scores.append(token_score)
            
            final_score = total_score
            
        else:
            raise ValueError(f"Unknown scoring method: {score}")
    
    # Print results
    print(f"\nTotal {score} score: {final_score:.4f}")
    
    if print_full:
        print("\nIndividual token scores:")
        for token, token_score in zip(tokens, individual_scores):
            print(f"  {token}: {token_score:.4f}")
    
    return final_score, individual_scores, tokens

  def prompt_score(self, prompt, score="cosine", layer=-1):
    inputs = self.tokenizer(prompt, return_tensors="pt", 
                           padding=True, truncation=True).to(self.device)
    
    # Get token strings for display
    token_ids = inputs['input_ids'][0]
    tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
    
    with torch.no_grad():
        # Check if we need attention weights
        need_attention = (score == "entropy")
        
        # Get model outputs
        outputs = self.model(
            **inputs, 
            output_hidden_states=True, 
            output_attentions=need_attention
        )
        
        # Get embeddings from the last layer
        hidden_states = outputs.hidden_states[-1]
        embeddings = hidden_states[0]
        
        if score == "cosine":
            # [Previous cosine cluster code remains the same]
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            cosine_sim = torch.mm(embeddings_norm, embeddings_norm.t())
            
            epsilon = 1e-10
            individual_scores = []
            total_score = 0.0
            
            for i in range(len(tokens)):
                token_score = 0.0
                for j in range(len(tokens)):
                    if i != j:
                        sim_value = cosine_sim[i, j].item()
                        shifted_sim = (sim_value + 1) / 2 + epsilon
                        pairwise_score = -np.log(shifted_sim)
                        token_score += pairwise_score
                        if i < j:
                            total_score += pairwise_score
                individual_scores.append(token_score)
            
            final_score = total_score
            
        elif score == "entropy":
            if outputs.attentions is None:
                print("Warning: Attention weights not available.")
                return self.promptcluster(prompt, score="cosine", print_full=print_full)
            
            # Get attention weights - average across all layers and heads for richer signal
            # You can also just use the last layer: attentions = outputs.attentions[-1]
            all_attentions = torch.stack(outputs.attentions)  # [num_layers, batch_size, num_heads, seq_len, seq_len]
            avg_attention = all_attentions.mean(dim=(0, 2))[0]  # Average over layers and heads -> [seq_len, seq_len]
            
            individual_scores = []
            total_score = 0.0
            
            # Calculate attention entropy for each token
            for i in range(len(tokens)):
                token_entropy = 0.0
                
                # Entropy from this token attending to others (outgoing attention)
                for j in range(len(tokens)):
                    if avg_attention[i, j] > 0:
                        a_ij = avg_attention[i, j].item()
                        token_entropy += -a_ij * np.log(a_ij + 1e-10)
                
                # Entropy from other tokens attending to this one (incoming attention)
                for j in range(len(tokens)):
                    if i != j and avg_attention[j, i] > 0:
                        a_ji = avg_attention[j, i].item()
                        token_entropy += -a_ji * np.log(a_ji + 1e-10)
                
                individual_scores.append(token_entropy)
                total_score += token_entropy
            
            # Total score is sum of all individual entropies divided by 2 
            # (since we counted each interaction twice)
            final_score = total_score / 2
            
        elif score == "euclidean":
            # [Previous euclidean code remains the same]
            individual_scores = []
            total_score = 0.0
            
            for i in range(len(tokens)):
                token_score = 0.0
                for j in range(len(tokens)):
                    if i != j:
                        dist = torch.norm(embeddings[i] - embeddings[j], p=2).item()
                        token_score += dist
                        if i < j:
                            total_score += dist
                individual_scores.append(token_score)
            
            final_score = total_score
            
        else:
            raise ValueError(f"Unknown scoring method: {score}")

    return final_score


  def analyze_layer_evolution(self, prompt, token_position=-1):
    """Track how a token's embedding evolves through layers"""
    
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    
    with torch.no_grad():
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get embeddings across all layers for the target token
        all_layer_embeddings = []
        for layer_hidden in outputs.hidden_states:
            token_embedding = layer_hidden[0, token_position, :]
            all_layer_embeddings.append(token_embedding)
        
        # Calculate similarity between consecutive layers
        layer_similarities = []
        for i in range(1, len(all_layer_embeddings)):
            cos_sim = F.cosine_similarity(
                all_layer_embeddings[i-1].unsqueeze(0),
                all_layer_embeddings[i].unsqueeze(0)
            ).item()
            layer_similarities.append(cos_sim)
            
        return layer_similarities

  from typing import Literal, Dict, Optional
  
  def analyze_token_distance_evolution(
      self,
      prompt: str,
      target_token: str,
      distance_metric: Literal["cosine", "euclidean"] = "cosine",
      return_details: bool = False
  ) -> Dict:
      """
      Analyze how the distance between a target token and the last token changes
      as they progress through model layers.
      
      Args:
          prompt: Input text
          target_token: Token to track (e.g., "France")
          distance_metric: "cosine" or "euclidean"
          return_details: If True, return additional information
      
      Returns:
          Dictionary with distances across layers and optional details
      """
      
      # Tokenize the prompt
      inputs = self.tokenizer(
          prompt, 
          return_tensors="pt",
          add_special_tokens=True
      ).to(self.device)
      
      # Get token strings and find target token position(s)
      token_ids = inputs['input_ids'][0]
      tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
      
      # Find the target token position(s)
      # Handle potential subword tokenization
      target_positions = []
      for i, token in enumerate(tokens):
          # Clean token for comparison (remove special chars like ▁ for Llama/Qwen)
          clean_token = token.replace('▁', '').replace('Ġ', '')
          if target_token.lower() in clean_token.lower():
              target_positions.append(i)
      
      if not target_positions:
          # Try partial matching for split tokens
          for i in range(len(tokens) - 1):
              combined = ''.join([
                  tokens[j].replace('▁', '').replace('Ġ', '') 
                  for j in range(i, min(i+3, len(tokens)))
              ])
              if target_token.lower() in combined.lower():
                  target_positions.append(i)
                  break
      
      if not target_positions:
          raise ValueError(f"Token '{target_token}' not found in prompt. "
                          f"Available tokens: {[t.replace('▁', '') for t in tokens]}")
      
      # Use the first occurrence if multiple found
      target_pos = target_positions[0]
      
      # Get the actual sequence length (excluding padding)
      seq_len = inputs['attention_mask'][0].sum().item()
      last_token_pos = seq_len - 1
      
      # Get model outputs with all hidden states
      with torch.no_grad():
          outputs = self.model(**inputs, output_hidden_states=True)
      
      # Calculate distances across layers
      distances = []
      layer_details = []
      
      for layer_idx, hidden_state in enumerate(outputs.hidden_states):
          # Get embeddings for this layer
          layer_embeddings = hidden_state[0]  # Remove batch dimension
          
          # Get target token and last token embeddings
          target_embedding = layer_embeddings[target_pos]
          last_token_embedding = layer_embeddings[last_token_pos]
          
          # Calculate distance based on metric
          if distance_metric == "cosine":
              # Cosine distance = 1 - cosine_similarity
              cos_sim = F.cosine_similarity(
                  target_embedding.unsqueeze(0),
                  last_token_embedding.unsqueeze(0)
              )
              distance = (1 - cos_sim).item()
          else:  # euclidean
              distance = torch.norm(
                  target_embedding - last_token_embedding, p=2
              ).item()
          
          distances.append(distance)
          
          if return_details:
              layer_details.append({
                  'layer': layer_idx,
                  'distance': distance,
                  'target_norm': torch.norm(target_embedding).item(),
                  'last_token_norm': torch.norm(last_token_embedding).item()
              })
      
      # Create result dictionary
      result = {
          'distances': distances,
          'target_token': tokens[target_pos],
          'target_position': target_pos,
          'last_token': tokens[last_token_pos],
          'last_token_position': last_token_pos,
          'metric': distance_metric,
          'num_layers': len(distances)
      }
      
      # Add analysis
      result['analysis'] = {
          'initial_distance': distances[0],
          'final_distance': distances[-1],
          'total_change': distances[-1] - distances[0],
          'max_distance': max(distances),
          'min_distance': min(distances),
          'max_distance_layer': distances.index(max(distances)),
          'min_distance_layer': distances.index(min(distances)),
          'monotonic_increase': all(distances[i] <= distances[i+1] 
                                    for i in range(len(distances)-1)),
          'monotonic_decrease': all(distances[i] >= distances[i+1] 
                                    for i in range(len(distances)-1))
      }
      
      if return_details:
          result['layer_details'] = layer_details
      
      return result
  
  # Example usage
  def demo_distance_tracking(self):
      """Demo the distance evolution analysis."""
      
      prompt = "The capital of France is Paris"
      target = "France"
      
      # Get distance evolution
      result = self.analyze_token_distance_evolution(
          prompt=prompt,
          target_token=target,
          distance_metric="cosine",
          return_details=True
      )
      
      # Print summary
      print(f"Tracking: '{result['target_token']}' (pos {result['target_position']}) "
            f"→ '{result['last_token']}' (pos {result['last_token_position']})")
      print(f"Metric: {result['metric']}")
      print(f"Number of layers: {result['num_layers']}")
      print("\nDistance Evolution:")
      print(f"  Initial (layer 0): {result['analysis']['initial_distance']:.4f}")
      print(f"  Final (layer {result['num_layers']-1}): {result['analysis']['final_distance']:.4f}")
      print(f"  Total change: {result['analysis']['total_change']:.4f}")
      print(f"  Min distance: {result['analysis']['min_distance']:.4f} "
            f"(layer {result['analysis']['min_distance_layer']})")
      print(f"  Max distance: {result['analysis']['max_distance']:.4f} "
            f"(layer {result['analysis']['max_distance_layer']})")
      
      # Show first few and last few distances
      print("\nLayer-by-layer distances:")
      for i in range(min(5, len(result['distances']))):
          print(f"  Layer {i}: {result['distances'][i]:.4f}")
      if len(result['distances']) > 10:
          print("  ...")
          for i in range(-3, 0):
              layer_num = len(result['distances']) + i
              print(f"  Layer {layer_num}: {result['distances'][layer_num]:.4f}")
      
      return result
  
  def logit_lens_analysis(self, prompt, layer_indices=None):
    """
    See what tokens the model is 'thinking about' at each layer.
    """
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    
    with torch.no_grad():
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get the unembedding matrix and find its device
        unembed = self.model.lm_head.weight
        unembed_device = unembed.device
        
        # Get the final layer norm - this is crucial for correct predictions
        # For Llama models, it's usually model.model.norm
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
            final_norm = self.model.model.norm
        else:
            final_norm = None
            print("Warning: No final layer norm found - results may be incorrect")
        
        # Analyze specific layers
        if layer_indices is None:
            layer_indices = [10, 15, 20, 25, -1]  # Sample layers
        
        results = {}
        for layer_idx in layer_indices:
            hidden_state = outputs.hidden_states[layer_idx][0]  # [seq_len, hidden_dim]
            
            # Focus on the last token position (where answer forms)
            last_token_hidden = hidden_state[-1]  # [hidden_dim]
            
            # Apply layer norm for all non-final layers (critical for correct predictions!)
            if layer_idx != -1 and final_norm is not None:
                last_token_hidden = final_norm(last_token_hidden)
            
            # Move hidden state to same device as unembed matrix
            last_token_hidden = last_token_hidden.to(unembed_device)
            
            # Project to vocabulary space
            logits = torch.matmul(last_token_hidden, unembed.T)  # [vocab_size]
            
            # Apply temperature scaling for better probability distribution
            logits = logits / 0.8  # Slightly lower temperature for sharper distribution
            
            probs = torch.softmax(logits, dim=-1)
            
            # Get top predicted tokens
            top_probs, top_indices = torch.topk(probs, k=10)
            top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]
            
            results[f"layer_{layer_idx}"] = list(zip(top_tokens, top_probs.tolist()))
    
    return results


  def print_results(self, results):
    for layer, predictions in results.items():
      print(f"\n{layer}:")
      for token, prob in predictions[:10]:
        print(f"  {token}: {prob:.3f}")

  def analyze_prompt_entropy(self, prompt, return_per_position=False):
    """
    Calculate entropy for next-token predictions.
    
    Low entropy = model is confident
    High entropy = model is uncertain
    """
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    
    with torch.no_grad():
        outputs = self.model(**inputs)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
        
        # Calculate probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate entropy for each position
        entropies = []
        for pos in range(len(logits)):
            pos_probs = probs[pos]
            # Filter near-zero probabilities to avoid log(0)
            pos_probs = pos_probs[pos_probs > 1e-8]
            entropy = -torch.sum(pos_probs * torch.log2(pos_probs)).item()
            entropies.append(entropy)
    
    if return_per_position:
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        return list(zip(tokens, entropies))
    else:
        # Return entropy for last position (next token prediction)
        return entropies[-1]


  def multi_choice(self, question, answers, score="cosine"):
    """
    Analyze multiple choice answers by computing clustering scores at each layer.
    
    Args:
        question: The question string
        answers: List of answer choices
        score: Scoring method ("cosine", "entropy", or "euclidean")
    
    Returns:
        Dictionary mapping each answer to its scores across all layers
    """
    num_layers = self.model.config.num_hidden_layers
    
    # Format the prompt
    answers_str = ", ".join(answers)
    prompt = f"{question} (choose between {answers_str})"
    
    # Tokenize the prompt
    inputs = self.tokenizer(prompt, return_tensors="pt", 
                           padding=True, truncation=True).to(self.device)
    
    # Get token strings and find answer positions
    token_ids = inputs['input_ids'][0]
    tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
    
    # Find the positions of each answer in the token sequence
    answer_positions = {}
    
    print(f"\nTokenized prompt: {tokens}")
    
    for answer in answers:
        # Tokenize each answer separately to handle multi-token answers
        answer_tokens = self.tokenizer.tokenize(answer)
        answer_token_ids = self.tokenizer.convert_tokens_to_ids(answer_tokens)
        
        found = False
        # Find where this answer appears in the full sequence
        for i in range(len(token_ids) - len(answer_token_ids) + 1):
            if all(token_ids[i+j] == answer_token_ids[j] for j in range(len(answer_token_ids))):
                # Store the range of positions for this answer
                answer_positions[answer] = list(range(i, i + len(answer_token_ids)))
                found = True
                break
        
        if not found:
            # Try a fallback: look for partial matches or similar tokens
            for i, token in enumerate(tokens):
                if answer.lower() in token.lower() or token.lower() in answer.lower():
                    answer_positions[answer] = [i]
                    break
    
    if len(answer_positions) < 2:
        return {answer: [0.0] * num_layers for answer in answers}
    
    # Initialize results dictionary
    results = {answer: [] for answer in answers}
    
    with torch.no_grad():
        # Check if we need attention weights
        need_attention = (score == "entropy")
        
        # Get model outputs with all hidden states
        outputs = self.model(
            **inputs, 
            output_hidden_states=True, 
            output_attentions=need_attention
        )
        
        # Process each layer
        for layer_idx in range(num_layers):
            layer_scores = {}
            
            # Get embeddings from this layer
            hidden_states = outputs.hidden_states[layer_idx + 1]  # +1 because first is embeddings
            embeddings = hidden_states[0]
            
            # Calculate scores for each answer
            for answer in answers:
                if answer not in answer_positions:
                    layer_scores[answer] = 0.0
                    continue
                
                positions = answer_positions[answer]
                answer_score = 0.0
                comparisons_made = 0
                
                if score == "cosine":
                    # Calculate cosine similarity-based clustering score
                    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
                    
                    # Average embedding for multi-token answers
                    if len(positions) > 1:
                        answer_embedding = embeddings_norm[positions].mean(dim=0, keepdim=True)
                    else:
                        answer_embedding = embeddings_norm[positions[0]].unsqueeze(0)
                    
                    # Calculate similarity with all other answer embeddings
                    for other_answer in answers:
                        if other_answer != answer and other_answer in answer_positions:
                            other_positions = answer_positions[other_answer]
                            if len(other_positions) > 1:
                                other_embedding = embeddings_norm[other_positions].mean(dim=0, keepdim=True)
                            else:
                                other_embedding = embeddings_norm[other_positions[0]].unsqueeze(0)
                            
                            sim = torch.cosine_similarity(answer_embedding, other_embedding).item()
                            # Convert similarity to distance-like score
                            shifted_sim = (sim + 1) / 2 + 1e-10
                            answer_score += -np.log(shifted_sim)
                            comparisons_made += 1
                    
                    if comparisons_made == 0:
                        answer_score = 0.0
                    
                elif score == "entropy":
                    if outputs.attentions is None:
                        print(f"Warning: Attention weights not available for layer {layer_idx}")
                        answer_score = 0.0
                    else:
                        # Get attention weights for this layer
                        layer_attention = outputs.attentions[layer_idx]  # [batch_size, num_heads, seq_len, seq_len]
                        avg_attention = layer_attention.mean(dim=1)[0]  # Average over heads -> [seq_len, seq_len]
                        
                        # Calculate attention entropy for answer tokens
                        for pos in positions:
                            # Outgoing attention entropy
                            for j in range(len(tokens)):
                                if avg_attention[pos, j] > 0:
                                    a_ij = avg_attention[pos, j].item()
                                    answer_score += -a_ij * np.log(a_ij + 1e-10)
                            
                            # Incoming attention entropy
                            for j in range(len(tokens)):
                                if j != pos and avg_attention[j, pos] > 0:
                                    a_ji = avg_attention[j, pos].item()
                                    answer_score += -a_ji * np.log(a_ji + 1e-10)
                        
                        # Normalize by number of tokens in answer
                        answer_score /= len(positions)
                
                elif score == "euclidean":
                    # Calculate Euclidean distance-based clustering score
                    # Average embedding for multi-token answers
                    if len(positions) > 1:
                        answer_embedding = embeddings[positions].mean(dim=0, keepdim=True)
                    else:
                        answer_embedding = embeddings[positions[0]].unsqueeze(0)
                    
                    # Calculate distance to all other answer embeddings
                    for other_answer in answers:
                        if other_answer != answer and other_answer in answer_positions:
                            other_positions = answer_positions[other_answer]
                            if len(other_positions) > 1:
                                other_embedding = embeddings[other_positions].mean(dim=0, keepdim=True)
                            else:
                                other_embedding = embeddings[other_positions[0]].unsqueeze(0)
                            
                            dist = torch.norm(answer_embedding - other_embedding, p=2).item()
                            answer_score += dist
                
                else:
                    raise ValueError(f"Unknown scoring method: {score}")
                
                # Round to 3 decimal places
                layer_scores[answer] = round(float(answer_score), 3)
            
            # Append scores for this layer to results
            for answer in answers:
                results[answer].append(layer_scores.get(answer, 0.0))
    
    # Print results with each answer on a different line
    print("{")
    for i, (answer, scores) in enumerate(results.items()):
        if i < len(results) - 1:
            print(f'    "{answer}": {scores},')
        else:
            print(f'    "{answer}": {scores}')
    print("}")


  def print_pca_data(self, tokens, layer=-1):
    """
    Extract PCA data and print for copy-pasting (no file saving)
    
    Args:
        tokens: List of tokens/words to analyze
        layer: Which layer to extract embeddings from (-1 = last layer)
    """
    
    # Ensure model outputs hidden states
    original_output_hidden_states = getattr(self.model.config, 'output_hidden_states', False)
    self.model.config.output_hidden_states = True
    
    # Get embeddings for each token
    embeddings = []
    valid_tokens = []
    
    with torch.no_grad():
        for token in tokens:
            # Tokenize 
            inputs = self.tokenizer(token, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states
            
            # Extract embedding from specified layer
            token_embedding = hidden_states[layer][0].mean(dim=0)
            
            embeddings.append(token_embedding.cpu().numpy())
            valid_tokens.append(token)
    
    # Restore original config
    self.model.config.output_hidden_states = original_output_hidden_states
    
    # Convert to numpy array
    embeddings_matrix = np.array(embeddings)
    
    # Perform PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_matrix)
    
    # Prepare data dictionary (fix JSON serialization)
    pca_data = {
        'tokens': [str(token) for token in valid_tokens],  # Ensure strings
        'embeddings_2d': [[float(x) for x in row] for row in embeddings_2d],  # Ensure floats
        'explained_variance_ratio': [float(x) for x in pca.explained_variance_ratio_],  # Ensure floats
        'layer': int(layer),  # Ensure int
        'total_explained_variance': float(pca.explained_variance_ratio_.sum())
    }
    
    # Print summary first
    print(f"\n=== PCA Analysis Summary ===")
    print(f"Tokens: {valid_tokens}")
    print(f"Layer: {layer}")
    print(f"Explained variance: PC1={pca_data['explained_variance_ratio'][0]:.3f}, PC2={pca_data['explained_variance_ratio'][1]:.3f}")
    print(f"Total explained variance: {pca_data['total_explained_variance']:.3f}")
    
    # Print the data for copy-pasting
    try:
        print(f"\n{'='*60}")
        print("COPY THE DATA BELOW (including the curly braces):")
        print(f"{'='*60}")
        print(json.dumps(pca_data, indent=2))
        print(f"{'='*60}")
    except Exception as e:
        print(f"JSON serialization error: {e}")
        print("Raw data types:")
        for key, value in pca_data.items():
            print(f"{key}: {type(value)}")
    
    return pca_data

  def print_contextualized_pca_data(self, prompt, layer=-1, distance_metric='euclidean'):
    """
    Perform PCA analysis on contextualized token embeddings from a prompt
    
    Args:
        prompt: String prompt to analyze (e.g., "The bank by the river is peaceful")
        layer: Which layer to extract embeddings from (-1 = last layer)
        distance_metric: 'euclidean' or 'cosine' - distance metric for PCA preprocessing
    """
    
    # Ensure model outputs hidden states
    original_output_hidden_states = getattr(self.model.config, 'output_hidden_states', False)
    self.model.config.output_hidden_states = True
    
    # Tokenize the entire prompt
    inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    
    # Get model outputs
    with torch.no_grad():
        outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
    
    # Restore original config
    self.model.config.output_hidden_states = original_output_hidden_states
    
    # Extract embeddings from specified layer
    # Shape: [batch_size, sequence_length, hidden_size]
    layer_embeddings = hidden_states[layer][0]  # Remove batch dimension
    
    # Get token strings (decode each token)
    input_ids = inputs['input_ids'][0]  # Remove batch dimension
    tokens = [self.tokenizer.decode([token_id]) for token_id in input_ids]
    
    # Convert to numpy
    embeddings_matrix = layer_embeddings.cpu().numpy()
    
    # Apply distance metric transformation if cosine
    if distance_metric.lower() == 'cosine':
        # For cosine distance, we need to transform the data
        # Cosine distance focuses on direction, not magnitude
        # We'll normalize the vectors (unit vectors)
        from sklearn.preprocessing import normalize
        embeddings_matrix = normalize(embeddings_matrix, norm='l2')
        distance_name = "Cosine Distance (normalized)"
    elif distance_metric.lower() == 'euclidean':
        # No transformation needed for Euclidean
        distance_name = "Euclidean Distance"
    else:
        raise ValueError("distance_metric must be 'euclidean' or 'cosine'")
    
    # Perform PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_matrix)
    
    # Prepare data dictionary
    pca_data = {
        'prompt': prompt,
        'tokens': [str(token).strip() for token in tokens],  # Clean up token strings
        'embeddings_2d': [[float(x) for x in row] for row in embeddings_2d],
        'explained_variance_ratio': [float(x) for x in pca.explained_variance_ratio_],
        'layer': int(layer),
        'distance_metric': distance_metric,
        'total_explained_variance': float(pca.explained_variance_ratio_.sum()),
        'sequence_length': len(tokens)
    }
    
    # Print summary first
    print(f"\n=== Contextualized PCA Analysis Summary ===")
    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {pca_data['tokens']}")
    print(f"Layer: {layer}")
    print(f"Distance metric: {distance_name}")
    print(f"Sequence length: {len(tokens)}")
    print(f"Explained variance: PC1={pca_data['explained_variance_ratio'][0]:.3f}, PC2={pca_data['explained_variance_ratio'][1]:.3f}")
    print(f"Total explained variance: {pca_data['total_explained_variance']:.3f}")
    
    # Print the data for copy-pasting
    try:
        print(f"\n{'='*60}")
        print("COPY THE DATA BELOW (including the curly braces):")
        print(f"{'='*60}")
        print(json.dumps(pca_data, indent=2))
        print(f"{'='*60}")
    except Exception as e:
        print(f"JSON serialization error: {e}")
        print("Raw data types:")
        for key, value in pca_data.items():
            print(f"{key}: {type(value)}")
    
    return pca_data

def compare_distance_metrics_pca(self, prompt, layer=-1):
    """
    Compare PCA results using both Euclidean and Cosine distance metrics
    """
    print(f"\n=== Comparing Distance Metrics ===")
    print(f"Prompt: '{prompt}'")
    print(f"Layer: {layer}")
    
    # Analyze with both metrics
    euclidean_data = self.print_contextualized_pca_data(prompt, layer, 'euclidean')
    cosine_data = self.print_contextualized_pca_data(prompt, layer, 'cosine')
    
    # Combined data
    comparison_data = {
        'prompt': prompt,
        'layer': layer,
        'euclidean': euclidean_data,
        'cosine': cosine_data
    }
    
    print(f"\n{'='*60}")
    print("COPY THE COMPARISON DATA BELOW:")
    print(f"{'='*60}")
    print(json.dumps(comparison_data, indent=2))
    print(f"{'='*60}")
    
    return comparison_data

  def analyze_multiple_prompts_pca(self, prompts, layer=-1, distance_metric='euclidean'):
    """
    Analyze multiple prompts to see how context affects the same words
    
    Args:
        prompts: List of prompts (e.g., ["The bank by the river", "The bank gave a loan"])
        layer: Which layer to extract from
        distance_metric: 'euclidean' or 'cosine'
    """
    
    all_prompt_data = {}
    
    print(f"\n=== Multiple Prompt Analysis ===")
    print(f"Distance metric: {distance_metric}")
    print(f"Layer: {layer}")
    print(f"Analyzing {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1}: '{prompt}' ---")
        prompt_data = self.print_contextualized_pca_data(prompt, layer, distance_metric)
        all_prompt_data[f'prompt_{i+1}'] = prompt_data
    
    # Print combined data
    print(f"\n{'='*60}")
    print("COPY THE MULTI-PROMPT DATA BELOW:")
    print(f"{'='*60}")
    print(json.dumps(all_prompt_data, indent=2))
    print(f"{'='*60}")
    
    return all_prompt_data
  


    












    


  
