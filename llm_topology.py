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

  def internal_layer_topology(self, input, target_tokens, persistence_threshold=0.27):
    layers = np.arange(-29, 0)
    embeddings = []
    for l in layers:
      embeddings = embeddings + self.get_output_embeddings(input, target_tokens, layer=l)

    distance_matrix = self.compute_distance_matrix(embeddings)

    diagrams = ripser(distance_matrix, distance_matrix = True, thresh = persistence_threshold, maxdim = 2)

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
    print(f"Significant voids: {len(significant_voids)})
  
