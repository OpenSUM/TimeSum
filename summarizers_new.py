import networkx as nx
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
import collections
import numpy as np
from nltk import FreqDist
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import markov_clustering as mc
from sklearn.cluster import AffinityPropagation

class Summarizer:

    def summarize(self, sents, k, vectorizer, filters=None):
        raise NotImplementedError


class TextRank(Summarizer):

    def __init__(self, max_sim=0.9999):
        self.name = 'TextRank Summarizer'
        self.max_sim = max_sim

    def score_sentences(self, X):
        S = cosine_similarity(X)
        nodes = list(range(S.shape[0]))
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for i in range(S.shape[0]):
            for j in range(S.shape[0]):
                graph.add_edge(nodes[i], nodes[j], weight=S[i, j])
        pagerank = nx.pagerank(graph, weight='weight')
        scores = [pagerank[i] for i in nodes]
        return scores

    def summarize(self, sents, k, vectorizer, filter=None):
        raw_sents = [s.raw for s in sents]
        try:
            X = vectorizer.transform(raw_sents)
        except:
            return None

        scores = self.score_sentences(X)

        indices = list(range(len(sents)))
        ranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)

        summary_sents = []
        summary_vectors = []
        for i, _ in ranked:
            if len(summary_sents) >= k:
                break
            new_x = X[i]
            s = sents[i]
            is_redundant = False
            for x in summary_vectors:
                if cosine_similarity(new_x, x)[0, 0] > self.max_sim:
                    is_redundant = True
                    break
            if filter and not filter(s):
                continue
            elif is_redundant:
                continue
            else:
                summary_sents.append(sents[i])
                summary_vectors.append(new_x)

        summary = [s.raw for s in summary_sents]
        return summary


class CentroidRank(Summarizer):

    def __init__(self,  max_sim=0.9999):
        self.name = 'Sentence-Centroid Summarizer'
        self.max_sim = max_sim

    def score_sentences(self, X):
        Xsum = sparse.csr_matrix(X.sum(0))
        centroid = normalize(Xsum)
        scores = cosine_similarity(X, centroid)
        return scores

    def summarize(self, sents, k, vectorizer, filter=None):
        raw_sents = [s.raw for s in sents]
        try:
            X = vectorizer.transform(raw_sents)
            for i, s in enumerate(sents):
                s.vector = X[i]
        except:
            return None

        scores = self.score_sentences(X)
        indices = list(range(len(sents)))
        ranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)

        summary_sents = []
        summary_vectors = []
        for i, _ in ranked:
            if len(summary_sents) >= k:
                break
            new_x = X[i]
            s = sents[i]
            is_redundant = False
            for x in summary_vectors:
                if cosine_similarity(new_x, x)[0, 0] > self.max_sim:
                    is_redundant = True
                    break
            if filter and not filter(s):
                continue
            elif is_redundant:
                continue
            else:
                summary_sents.append(sents[i])
                summary_vectors.append(new_x)

        summary = [s.raw for s in summary_sents]
        return summary


class CentroidOpt(Summarizer):

    def __init__(self, max_sim=0.9999):
        self.name = 'Summary-Centroid Summarizer'
        self.max_sim = max_sim

    def optimise(self, centroid, X, sents, k, filter):
        remaining = set(range(len(sents)))
        selected = []
        while len(remaining) > 0 and len(selected) < k:
            if len(selected) > 0:
                summary_vector = sparse.vstack([X[i] for i in selected])
                summary_vector = sparse.csr_matrix(summary_vector.sum(0))
            i_to_score = {}
            for i in remaining:
                if len(selected) > 0:
                    new_x = X[i]
                    new_summary_vector = sparse.vstack([new_x, summary_vector])
                    new_summary_vector = normalize(new_summary_vector.sum(0))
                else:
                    new_summary_vector = X[i]
                score = cosine_similarity(new_summary_vector, centroid)[0, 0]
                i_to_score[i] = score

            ranked = sorted(i_to_score.items(), key=lambda x: x[1], reverse=True)
            for i, score in ranked:
                s = sents[i]
                remaining.remove(i)
                if filter and not filter(s):
                    continue
                elif self.is_redundant(i, selected, X):
                    continue
                else:
                    selected.append(i)
                    break
        return selected

    def is_redundant(self, new_i, selected, X):
        summary_vectors = [X[i] for i in selected]
        new_x = X[new_i]
        for x in summary_vectors:
            if cosine_similarity(new_x, x)[0] > self.max_sim:
                return True
        return False

    def summarize(self, sents, k, vectorizer, filter=None):
        raw_sents = [s.raw for s in sents]
        try:
            X = vectorizer.transform(raw_sents)
        except:
            return None
        X = sparse.csr_matrix(X)
        Xsum = sparse.csr_matrix(X.sum(0))
        centroid = normalize(Xsum)
        selected = self.optimise(centroid, X, sents, k, filter)
        summary = [sents[i].raw for i in selected]
        return summary


class SubmodularSummarizer(Summarizer):
    """
    Selects a combination of sentences as a summary by greedily optimising
    a submodular function.
    The function models the coverage and diversity of the sentence combination.
    """
    def __init__(self, a=5, div_weight=6,con_weight=1,imp_weight=15, cluster_factor=0.2):
        self.name = 'Submodular Summarizer'
        self.a = a
        self.div_weight = div_weight
        self.con_weight = con_weight
        self.imp_weight = imp_weight
        self.cluster_factor = cluster_factor

    def cluster_sentences(self, X,pairwise_sims):
        n = X.shape[0]
        n_clusters = round(self.cluster_factor * n)
        if n_clusters <= 1 or n <= 2:
            return dict((i, 1) for i in range(n))  
        clusterer = MiniBatchKMeans(n_clusters=n_clusters)
        labels = clusterer.fit_predict(X)
        i_to_label = dict((i, l) for i, l in enumerate(labels))
        return i_to_label

    def compute_summary_coverage(self,
                                 alpha,
                                 summary_indices,
                                 sent_coverages,
                                 pairwise_sims):
        cov = 0
        for i, i_generic_cov in enumerate(sent_coverages):
            i_summary_cov = sum([pairwise_sims[i, j] for j in summary_indices])
            i_cov = min(i_summary_cov, alpha * i_generic_cov)
            cov += i_cov
        return cov

    def compute_summary_diversity(self,
                                  summary_indices,
                                  ix_to_label,
                                  avg_sent_sims):

        cluster_to_ixs = collections.defaultdict(list)
        for i in summary_indices:
            l = ix_to_label[i]
            cluster_to_ixs[l].append(i)
        div = 0
        for l, l_indices in cluster_to_ixs.items():
            cluster_score = sum([avg_sent_sims[i] for i in l_indices])
            cluster_score = np.sqrt(cluster_score)
            div += cluster_score
        return div
    def compute_summary_coherence(self,
                                 alpha,
                                 summary_indices,
                                 sent_coverages,
                                 pairwise_sims):
        cov = 0
        for i,j in enumerate(list(summary_indices)):
            if(i!=0):
                i_summary_cov = pairwise_sims[list(summary_indices)[i-1],j]
                cov += i_summary_cov
        cov=cov/(len(summary_indices))
        return cov
    def compute_summary_importance(self,summary_indices,x1,sents,ngram_vectorizer,ngram_freq):
        tsum=0;
        for i in summary_indices:
            ngram_vectorizer1 = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",token_pattern = r'\b\w+\b',min_df=1)
            try:
                temp=ngram_vectorizer1.fit_transform([sents[i].raw])
                ngram1_freq=temp.toarray().sum(axis=0)
                for each in ngram_vectorizer1.vocabulary_:
                    index=ngram_vectorizer.vocabulary_.get(each)
                    index1=ngram_vectorizer1.vocabulary_.get(each)
                    tsum+=(ngram_freq[index] if index else 0 ) *(ngram1_freq[index1] if index1 else 0 )/(ngram1_freq.sum(axis=0))
            except ValueError:
                pass
        return tsum
    def optimise(self,
                 sents,
                 k,
                 filter,
                 ix_to_label,
                 pairwise_sims,
                 sent_coverages,
                 avg_sent_sims,
                 x1,ngram_vectorizer,ngram_freq):

        alpha = self.a / len(sents)
        remaining = set(range(len(sents)))
        selected = []
        while len(remaining) > 0 and len(selected) < k:

            i_to_score = {}
            for i in remaining:
                summary_indices = selected + [i]
                cov = self.compute_summary_coverage(
                    alpha, summary_indices, sent_coverages, pairwise_sims)
                div = self.compute_summary_diversity(
                    summary_indices, ix_to_label, avg_sent_sims)
                con = self.compute_summary_coherence(
                    alpha, summary_indices, sent_coverages, pairwise_sims)
                imp = self.compute_summary_importance(summary_indices,x1,sents,ngram_vectorizer,ngram_freq)
                score = cov + self.div_weight * div+ self.imp_weight * imp#+self.con_weight*con
                print(score,imp_weight)
                i_to_score[i] = score
                
            ranked = sorted(i_to_score.items(), key=lambda x: x[1], reverse=True)
            for i, score in ranked:
                s = sents[i]
                remaining.remove(i)
                if filter and not filter(s):
                    continue
                else:
                    selected.append(i)
                    break

        return selected

    def summarize(self, sents, k, vectorizer, filter=None):
        raw_sents = [s.raw for s in sents]
        bi_gram_raw=''
        for each in raw_sents:
            bi_gram_raw="".join([bi_gram_raw,each])
        ngram_vectorizer = TfidfVectorizer(stop_words='english',decode_error="ignore",token_pattern = r'\b\w+\b',min_df=1)
        try:
            x1 = ngram_vectorizer.fit_transform([bi_gram_raw])
        except:
            return None
        ngram_freq=x1.toarray().sum(axis=0)
    
        try:
            X = vectorizer.transform(raw_sents)
        except:
            return None
        pairwise_sims = cosine_similarity(X)
        ix_to_label = self.cluster_sentences(X,pairwise_sims)
        sent_coverages = pairwise_sims.sum(0)
        avg_sent_sims = sent_coverages / len(sents)

        selected = self.optimise(
            sents, k, filter, ix_to_label,
            pairwise_sims, sent_coverages, avg_sent_sims,x1,ngram_vectorizer,ngram_freq
        )

        summary = [sents[i].raw for i in selected]
        return summary
