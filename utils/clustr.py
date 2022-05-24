"""
@author: noobrainers
@contact: Bryan T. Kim bryantaekim at gmail dot com
          Daniel Lee dslee47 at gmail dot com
          Juswaldy Jusman juswaldy at gmail dot com
@description: Utility functions for Hierarchical Clustering.
@content:
    def reduce_dim(cos_sim, random_state):
    def clustering(tfidf_matrix, _n_clusters):
    def _plot_distance(Z, _top):
    def _plot_clusters(df_2d, save=False):
    def _plot_dendrogram(cos_sim, target, _p=30, _trunc_mode=None, fw=15, fh=10, zoom_in=True, zoom_xlim = 2500, threshold=0, save_pic=False):
    def _tokenize_and_stem(text):
    def tfidf_vectorizer(_raw_text, max_df=.5, min_df=10, gram=3, max_features=20*10000, n_show=20):
    def _tagPos(text):
    def prepCorpus(corpus, subset_cat, cols, label, feature, time_period, year):
    def get_tfidf(corpus, subset_cat, cols, label, feature, time_period, year, times):
    def plot_dendrogram(cos_sim, target, _p=30, _trunc_mode=None, fw=15, fh=10, zoom_in=True, zoom_xlim = 2500, threshold=0, save_pic=False):
    def plot_distance(Z, _top):
    def plot_clusters(df_2d, save=False):
"""

## Import required packages
import pandas as pd, numpy as np, sys, mpld3, matplotlib.pyplot as plt, seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram

import nltk
from nltk.stem.snowball import SnowballStemmer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def reduce_dim(cos_sim, random_state):
    ## Dimension reduced to 2D for visualization - xs,ys
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state)
    pos = mds.fit_transform(cos_sim)
    return pos[:, 0], pos[:, 1]

def clustering(tfidf_matrix, _n_clusters):
    cluster = AgglomerativeClustering(n_clusters=_n_clusters, affinity='euclidean', linkage='ward')
    cluster.fit_predict(tfidf_matrix.toarray())
    return cluster.labels_

def _plot_distance(Z, _top):
    idx = np.argsort(Z[:,-1])
    Z = Z[idx][::-1][:_top]
    
    df = pd.DataFrame(
        columns=Z[:,0].astype(int), 
        index=Z[:,1].astype(int)
    )
    
    for i, d in enumerate(Z[:,2]):
        df.iloc[i, i] = d
    
    df.fillna(0, inplace=True)
    # mask everything but diagonal
    mask = np.ones(df.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    
    # plot the heatmap
    plt.figure(figsize=(20,20))
    sns.heatmap(df, 
                annot=True, fmt='.0f', cmap="YlGnBu", xticklabels=False, yticklabels=False,
                mask=mask)
    plt.title("Top " + str(_top) + " Distances")
    plt.show()
    plt.close()

class custom_css:
    #define custom css to format the font and to remove the axis labeling
    css = """
        text.mpld3-text, div.mpld3-tooltip {
          font-family:Arial, Helvetica, sans-serif;
        }

        g.mpld3-xaxis, g.mpld3-yaxis {
        display: none; }

        svg.mpld3-figure {
        margin-left: -200px;}
     """
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}
        
def _plot_clusters(df_2d, save=False):
    fig, ax = plt.subplots(figsize=(14,6))
    ax.margins(0.05)

    for c, group in df_2d.groupby('segment'):
        points = ax.plot(group.x, group.y, marker='o',label=c, linestyle='', ms=15)
        ax.set_aspect('auto')
        clusters = list(group.label)
        
        #set tooltip using points, labels and the already defined 'css'
        tooltip = mpld3.plugins.PointHTMLTooltip(points[0], clusters, voffset=10, hoffset=10, css=custom_css.css)
        #connect tooltip to fig
        mpld3.plugins.connect(fig, tooltip, TopToolbar())    
        
        #set tick marks as blank
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        
        #set axis as blank
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    ax.legend(numpoints=1)

    # mpld3.display()
    plt.show()

    if save:
        html = mpld3.fig_to_html(fig)
        print(html)

    plt.close(fig)
    
def _plot_dendrogram(cos_sim, target, _p=30, _trunc_mode=None, fw=15, fh=10, zoom_in=True, zoom_xlim = 2500, threshold=0, save_pic=False):

    linkage_matrix = ward(cos_sim)

    plt.figure(figsize=(fw, fh))
    dendrogram(linkage_matrix
              ,labels=target
              ,above_threshold_color='y'
              ,p=_p
              ,truncate_mode=_trunc_mode
               )
    if zoom_in:
        plt.xlim(0, zoom_xlim)
        plt.title('Dendrogram - zoomed in up to '+ str(zoom_xlim))
    else:
        plt.title('Dendrogram - All data points')
    
    if threshold > 0:
        plt.axhline(y=threshold, color='r', linestyle='--')
        
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    if save_pic:
        plt.savefig('hierarchical_clusters_dendrogram.png')
        
    plt.close()
    return linkage_matrix.astype(int)

def _tokenize_and_stem(text):
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer('english')
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]    
    stems = [stemmer.stem(t) for t in tokens if t not in stopwords]
    return stems

def _tfidf_vectorizer(_raw_text, max_df=.5, min_df=10, gram=3, max_features=20*10000, n_show=20):
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features,\
                                       tokenizer=_tokenize_and_stem, ngram_range=(1,gram))
    tfidf_matrix = tfidf_vectorizer.fit_transform(_raw_text)
    print(tfidf_matrix.shape)

    terms = tfidf_vectorizer.get_feature_names_out()
    print('Feature names up to ' + str(n_show) + ' : ', terms[:n_show])

    tfidf_matrix = tfidf_matrix.astype(np.uint8)
    cos_sim = cosine_similarity(tfidf_matrix)

    print('cosine_similarity dim : ', cos_sim.shape)
    print('cosine_similarity matrix : ', cos_sim)

    return cos_sim, tfidf_matrix

def _tagPos(text):
    sentence = nltk.sent_tokenize(text)
    for sent in sentence:
        postags = nltk.pos_tag(nltk.word_tokenize(sent))
        return ' '.join([p[0] for p in postags if p[1] in ('NN','NNS','JJ','JJR','JJS','NNP','NNPS'
#                                                            ,'VB','VBG','VBD','VBP','VBN','VBZ'\
                                                          )])

def prepCorpus(corpus, subset_cat, cols, label, feature, start_date, end_date):
    sub_corpus = corpus[(corpus[label].astype(str).str.contains('|'.join(subset_cat))) \
        & ((corpus['date'].str[:10] >= start_date) \
        & (corpus['date'].str[:10] <= end_date))][cols]
    sub_corpus.drop_duplicates(subset=[feature], inplace=True)
    sub_corpus.drop(sub_corpus[sub_corpus[feature] == 'none'].index, inplace=True)
    
    sub_corpus[feature + '_pos_tagged'] = sub_corpus[feature].apply(lambda x: _tagPos(x))
    
    return sub_corpus

def get_tfidf(corpus, subset_cat, cols, label, feature, start_date, end_date):
    sub_corpus = prepCorpus(corpus, subset_cat, cols, label, feature, start_date, end_date)

    _raw_text = sub_corpus[feature]
    _labels = sub_corpus[label].values

    # Calculate similiarity and transform the corpus into a tf-idf matrix
    cos_sim, tfidf_matrix = _tfidf_vectorizer(_raw_text, max_df=.5, min_df=.025)

    return cos_sim, tfidf_matrix, _labels, _raw_text

def plot_dendrogram(cos_sim, target, _p=30, _trunc_mode=None, fw=15, fh=10, zoom_in=True, zoom_xlim = 2500, threshold=0, save_pic=False):
    
    linkage_matrix = ward(cos_sim)

    fig = plt.figure(figsize=(fw, fh))
    dendrogram(linkage_matrix
              ,labels=target
              ,above_threshold_color='y'
              ,p=_p
              ,truncate_mode=_trunc_mode
               )
    if zoom_in:
        plt.xlim(0, zoom_xlim)
        plt.title('Dendrogram - zoomed in up to '+ str(zoom_xlim))
    else:
        plt.title('Dendrogram - All data points')
    
    if threshold > 0:
        plt.axhline(y=threshold, color='r', linestyle='--')
        
    plt.xticks(rotation=0)
    plt.tight_layout()
    # plt.show()
    
    if save_pic:
        plt.savefig('hierarchical_clusters_dendrogram.png')
        
    # plt.close()
    return fig, linkage_matrix.astype(int)

def plot_distance(Z, _top):
    idx = np.argsort(Z[:,-1])
    Z = Z[idx][::-1][:_top]
    
    df = pd.DataFrame(
        columns=Z[:,0].astype(int), 
        index=Z[:,1].astype(int)
    )
    
    for i, d in enumerate(Z[:,2]):
        df.iloc[i, i] = d
    
    df.fillna(0, inplace=True)
    # mask everything but diagonal
    mask = np.ones(df.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    
    # plot the heatmap
    fig = plt.figure(figsize=(20,20))
    sns.heatmap(df, 
                annot=True, fmt='.0f', cmap="YlGnBu", xticklabels=False, yticklabels=False,
                mask=mask)
    plt.title("Top " + str(_top) + " Distances")
    #plt.show()
    #plt.close()
    return fig

def plot_clusters(df_2d, save=False):
    fig, ax = plt.subplots(figsize=(14,6))
    ax.margins(0.05)

    for c, group in df_2d.groupby('segment'):
        points = ax.plot(group.x, group.y, marker='o',label=c, linestyle='', ms=15)
        ax.set_aspect('auto')
        clusters = list(group.label)
        
        #set tooltip using points, labels and the already defined 'css'
        tooltip = mpld3.plugins.PointHTMLTooltip(points[0], clusters, voffset=10, hoffset=10, css=custom_css.css)
        #connect tooltip to fig
        mpld3.plugins.connect(fig, tooltip, TopToolbar())    
        
        #set tick marks as blank
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        
        #set axis as blank
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    ax.legend(numpoints=1)

    # mpld3.display()
    #plt.show()

    if save:
        html = mpld3.fig_to_html(fig)
        print(html)

    #plt.close(fig)
    
    return fig
