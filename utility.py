from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

def identify_novel_sentences(target_abstract, predecessor_abstracts, n_gram_threshold=0.3, semantic_threshold=0.7):
    # Initialize sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Tokenize abstracts into sentences
    target_sentences = sent_tokenize(target_abstract)
    all_predecessor_sentences = []
    for abstract in predecessor_abstracts:
        all_predecessor_sentences.extend(sent_tokenize(abstract))
    
    novel_sentences = []
    
    for target_sent in target_sentences:
        n_gram_novel = calculate_ngram_novelty(target_sent, all_predecessor_sentences)
        target_embedding = model.encode([target_sent])[0]
        predecessor_embeddings = model.encode(all_predecessor_sentences)
        if len(predecessor_embeddings) == 0:
            novel_sentences.append(target_sent)
            continue
            
        similarities = cosine_similarity([target_embedding], predecessor_embeddings)[0]
        max_similarity = max(similarities)
        
        if n_gram_novel > n_gram_threshold and max_similarity < semantic_threshold:
            novel_sentences.append(target_sent)
    
    return novel_sentences

def calculate_ngram_novelty(sentence, predecessor_sentences, n=2):
    """Calculate percentage of n-grams in the sentence that don't appear in predecessor sentences"""
    from nltk.util import ngrams
    from nltk.tokenize import word_tokenize
    
    sentence_tokens = word_tokenize(sentence.lower())
    sentence_ngrams = set(ngrams(sentence_tokens, n))
    
    if not sentence_ngrams:
        return 0.0
    
    predecessor_ngrams = set()
    for pred_sent in predecessor_sentences:
        pred_tokens = word_tokenize(pred_sent.lower())
        predecessor_ngrams.update(ngrams(pred_tokens, n))
    
    novel_ngrams = sentence_ngrams - predecessor_ngrams
    novelty_score = len(novel_ngrams) / len(sentence_ngrams)
    
    return novelty_score

def assign_structural_weights(sentences, position_weight=0.2):
    n_sentences = len(sentences)
    if n_sentences <= 1:
        return [1.0] * n_sentences
        
    weights = []
    
    for i in range(n_sentences):
        relative_pos = i / n_sentences
        
        if relative_pos < 0.25: 
            weight = 0.7
        elif relative_pos < 0.75: 
            weight = 1.0
        else: 
            weight = 0.8
            
        position_factor = 1.0 + position_weight * (relative_pos - 0.5)
        weights.append(weight * position_factor)
        
    return weights

def identify_novel_sentences_with_structure(target_abstract, predecessor_abstracts, 
                                           n_gram_threshold=0.3, semantic_threshold=0.7,
                                           structure_weight=0.2):
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
    
    weighted_sentences.sort(key=lambda x: x[1], reverse=True)
    
    weight_threshold = 0.75
    selected_sentences = [s[0] for s in weighted_sentences if s[1] >= weight_threshold]
    
    ordered_novel_sentences = [s for s in target_sentences if s in selected_sentences]
    
    return ordered_novel_sentences

def enhance_with_citation_context(target_paper, predecessor_papers, novel_sentences):
    if not hasattr(target_paper, 'citation_contexts') or not target_paper.citation_contexts:
        return novel_sentences
    
    enhanced_sentences = list(novel_sentences)  # Create a copy
    
    cited_paper_ids = [paper.id for paper in predecessor_papers]
    
    for context in target_paper.citation_contexts:
        if context.cited_paper_id in cited_paper_ids:
            context_sentences = sent_tokenize(context.text)
            
            for sent in context_sentences:
                # Look for comparative language
                if any(phrase in sent.lower() for phrase in 
                      ["unlike", "improve", "extend", "advance", "better than", 
                       "compared to", "in contrast", "different from", "contribution"]):
                    if sent not in enhanced_sentences:
                        enhanced_sentences.append(sent)
    
    return enhanced_sentences

def filter_contribution_sentences(sentences):
    contribution_sentences = []
    
    background_phrases = ["has been", "have been", "previous work", "traditionally", 
                         "commonly", "typically", "in the past", "literature"]
    
    limitation_phrases = ["limitation", "limited by", "constraint", "future work", 
                         "further research", "remains to be", "not addressed"]
    
    contribution_phrases = ["we propose", "we present", "we introduce", "we develop", 
                           "we demonstrate", "this paper", "our approach", "our method",
                           "our work", "our model", "results show", "we find", "we show"]
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        is_contribution = any(phrase in sentence_lower for phrase in contribution_phrases)
        is_background = any(phrase in sentence_lower for phrase in background_phrases)
        is_limitation = any(phrase in sentence_lower for phrase in limitation_phrases)
        
        if (is_contribution and not is_limitation) or (not is_background and not is_limitation):
            contribution_sentences.append(sentence)
    
    return contribution_sentences
