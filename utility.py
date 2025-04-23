from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

def identify_novel_sentences(target_abstract, predecessor_abstracts, n_gram_threshold=0.3, semantic_threshold=0.7):
    """
    Identify sentences in target abstract that are novel compared to predecessor abstracts
    using both n-gram overlap and semantic similarity.
    """
    # Initialize sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Tokenize abstracts into sentences
    target_sentences = sent_tokenize(target_abstract)
    all_predecessor_sentences = []
    for abstract in predecessor_abstracts:
        all_predecessor_sentences.extend(sent_tokenize(abstract))
    
    novel_sentences = []
    
    # Calculate novelty for each target sentence
    for target_sent in target_sentences:
        # Calculate n-gram overlap
        n_gram_novel = calculate_ngram_novelty(target_sent, all_predecessor_sentences)
        
        # Calculate semantic similarity
        target_embedding = model.encode([target_sent])[0]
        predecessor_embeddings = model.encode(all_predecessor_sentences)
        
        # If no predecessor sentences, consider sentence novel
        if len(predecessor_embeddings) == 0:
            novel_sentences.append(target_sent)
            continue
            
        # Compute semantic similarity scores
        similarities = cosine_similarity([target_embedding], predecessor_embeddings)[0]
        max_similarity = max(similarities)
        
        # Consider a sentence novel if it has low n-gram overlap AND not too high semantic similarity
        if n_gram_novel > n_gram_threshold and max_similarity < semantic_threshold:
            novel_sentences.append(target_sent)
    
    return novel_sentences

def calculate_ngram_novelty(sentence, predecessor_sentences, n=2):
    """Calculate percentage of n-grams in the sentence that don't appear in predecessor sentences"""
    from nltk.util import ngrams
    from nltk.tokenize import word_tokenize
    
    # Extract n-grams from the target sentence
    sentence_tokens = word_tokenize(sentence.lower())
    sentence_ngrams = set(ngrams(sentence_tokens, n))
    
    if not sentence_ngrams:
        return 0.0
    
    # Extract n-grams from all predecessor sentences
    predecessor_ngrams = set()
    for pred_sent in predecessor_sentences:
        pred_tokens = word_tokenize(pred_sent.lower())
        predecessor_ngrams.update(ngrams(pred_tokens, n))
    
    # Calculate novelty as the proportion of sentence n-grams not in predecessor n-grams
    novel_ngrams = sentence_ngrams - predecessor_ngrams
    novelty_score = len(novel_ngrams) / len(sentence_ngrams)
    
    return novelty_score

def assign_structural_weights(sentences, position_weight=0.2):
    """
    Assign weights to sentences based on their position in the abstract
    Middle sentences often contain main contributions in scientific abstracts
    """
    n_sentences = len(sentences)
    if n_sentences <= 1:
        return [1.0] * n_sentences
        
    # Implement a simple weighting scheme:
    # - First 25% of sentences: likely background (lower weight)
    # - Middle 50% of sentences: likely contributions (higher weight)
    # - Last 25% of sentences: likely conclusions (medium weight)
    weights = []
    
    for i in range(n_sentences):
        relative_pos = i / n_sentences
        
        if relative_pos < 0.25:  # Background
            weight = 0.7
        elif relative_pos < 0.75:  # Core/contributions
            weight = 1.0
        else:  # Conclusions
            weight = 0.8
            
        # Apply position adjustment
        position_factor = 1.0 + position_weight * (relative_pos - 0.5)
        weights.append(weight * position_factor)
        
    return weights

def identify_novel_sentences_with_structure(target_abstract, predecessor_abstracts, 
                                           n_gram_threshold=0.3, semantic_threshold=0.7,
                                           structure_weight=0.2):
    """Enhanced novelty detection with structural context"""
    # Get basic novel sentences
    target_sentences = sent_tokenize(target_abstract)
    novel_sentences = identify_novel_sentences(target_abstract, predecessor_abstracts, 
                                             n_gram_threshold, semantic_threshold)
    
    # Apply structural weights
    structural_weights = assign_structural_weights(target_sentences, structure_weight)
    
    # Create a weighted list of sentences
    weighted_sentences = []
    for i, sentence in enumerate(target_sentences):
        if sentence in novel_sentences:
            weighted_sentences.append((sentence, structural_weights[i]))
    
    # Sort by structural weight
    weighted_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Return sentences in original order but filtered by weight threshold
    weight_threshold = 0.75
    selected_sentences = [s[0] for s in weighted_sentences if s[1] >= weight_threshold]
    
    # Preserve original order for readability
    ordered_novel_sentences = [s for s in target_sentences if s in selected_sentences]
    
    return ordered_novel_sentences

def enhance_with_citation_context(target_paper, predecessor_papers, novel_sentences):
    """
    Enhance novelty detection using citation contexts if available
    """
    # Check if we have citation contexts available
    if not hasattr(target_paper, 'citation_contexts') or not target_paper.citation_contexts:
        return novel_sentences
    
    enhanced_sentences = list(novel_sentences)  # Create a copy
    
    # Extract cited paper IDs
    cited_paper_ids = [paper.id for paper in predecessor_papers]
    
    # Look for citation contexts that mention the predecessors
    for context in target_paper.citation_contexts:
        if context.cited_paper_id in cited_paper_ids:
            # Extract sentences from the citation context
            context_sentences = sent_tokenize(context.text)
            
            # Focus on sentences that explain differences or improvements
            for sent in context_sentences:
                # Look for comparative language
                if any(phrase in sent.lower() for phrase in 
                      ["unlike", "improve", "extend", "advance", "better than", 
                       "compared to", "in contrast", "different from", "contribution"]):
                    if sent not in enhanced_sentences:
                        enhanced_sentences.append(sent)
    
    return enhanced_sentences

def filter_contribution_sentences(sentences):
    """
    Filter sentences to focus on actual contributions
    rather than background, limitations, or future work
    """
    contribution_sentences = []
    
    # Define phrase patterns for different sentence types
    background_phrases = ["has been", "have been", "previous work", "traditionally", 
                         "commonly", "typically", "in the past", "literature"]
    
    limitation_phrases = ["limitation", "limited by", "constraint", "future work", 
                         "further research", "remains to be", "not addressed"]
    
    contribution_phrases = ["we propose", "we present", "we introduce", "we develop", 
                           "we demonstrate", "this paper", "our approach", "our method",
                           "our work", "our model", "results show", "we find", "we show"]
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Check if the sentence indicates a contribution
        is_contribution = any(phrase in sentence_lower for phrase in contribution_phrases)
        
        # Check if it's a background or limitation sentence
        is_background = any(phrase in sentence_lower for phrase in background_phrases)
        is_limitation = any(phrase in sentence_lower for phrase in limitation_phrases)
        
        # Only keep sentences that seem to be about contributions
        if (is_contribution and not is_limitation) or (not is_background and not is_limitation):
            contribution_sentences.append(sentence)
    
    return contribution_sentences

def enhance_coherence(sentences, target_abstract):
    """
    Enhance the coherence of selected novel sentences
    by potentially adding connective sentences
    """
    if not sentences:
        return []
        
    # If we only have 1-2 sentences, we might need connecting sentences
    if len(sentences) <= 2:
        all_sentences = sent_tokenize(target_abstract)
        
        # Find indices of our selected sentences
        indices = [all_sentences.index(s) for s in sentences if s in all_sentences]
        
        # If sentences are far apart, add a connecting sentence
        if len(indices) > 1 and max(indices) - min(indices) > 1:
            # Add one connecting sentence between them
            middle_idx = (indices[0] + indices[-1]) // 2
            if middle_idx not in indices:
                sentences = list(sentences)  # Convert to list if it's a tuple
                sentences.append(all_sentences[middle_idx])
    
    # Sort sentences to maintain original order
    all_sentences = sent_tokenize(target_abstract)
    ordered_sentences = [s for s in all_sentences if s in sentences]
    
    # Add simple connectives if needed
    coherent_summary = " ".join(ordered_sentences)
    
    return coherent_summary

def generate_contrastive_summary(target_paper, predecessor_papers):
    """
    Generate a contrastive summary highlighting novel contributions
    """
    # Extract abstracts
    target_abstract = target_paper.get("abstract", "")
    predecessor_abstracts = [p.get("abstract", "") for p in predecessor_papers if p.get("abstract")]
    
    # Skip if no meaningful data
    if not target_abstract or not predecessor_abstracts:
        return target_abstract
    
    # 1. Identify novel sentences using semantic similarity
    novel_sentences = identify_novel_sentences(target_abstract, predecessor_abstracts)
    
    # 2. Apply structural weighting
    novel_sentences = identify_novel_sentences_with_structure(target_abstract, predecessor_abstracts)
    
    # 3. Enhance with citation context if available
    novel_sentences = enhance_with_citation_context(target_paper, predecessor_papers, novel_sentences)
    
    # 4. Filter to focus on contributions
    novel_sentences = filter_contribution_sentences(novel_sentences)
    
    # 5. Enhance coherence
    coherent_summary = enhance_coherence(novel_sentences, target_abstract)
    
    return coherent_summary