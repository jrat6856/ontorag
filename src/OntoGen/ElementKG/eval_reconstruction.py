import sys
import argparse
sys.path.append("..")
from owlready2 import *
from tree import Tree
import pickle
import itertools
import os
from collections import defaultdict
# disable warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# CLI arguments
parser = argparse.ArgumentParser(description="Run tree evaluation with configurable paths.")
parser.add_argument(
    "--eval_tree",
    type=str,
    #default="taxonomy_0/tree_0.pkl",
    default="../SACs/sacs_llama3.1:70b/tree_4.pkl",
    help="Path to the evaluation tree pickle file."
)
args = parser.parse_args()

GT_TREE = 'docs/groud_truth/truth_tree.pkl'
EVAL_TREE = args.eval_tree
OUT_FILE = 'delete.csv'


def output(list_):
    # write each element of list_ in a new column in a csv file
    with open(OUT_FILE, 'a') as f:
        for item in list_:
            f.write("%s," % item)
        f.write("\n")


def accuracy(truth_vocabulary, predicted_vocabulary):
    truth_vocabulary = set(truth_vocabulary)
    predicted_vocabulary = set(predicted_vocabulary)
    return len(truth_vocabulary.intersection(predicted_vocabulary)) / len(truth_vocabulary)

def missing_terms(truth_vocabulary, predicted_vocabulary):
    truth_vocabulary = set(truth_vocabulary)
    predicted_vocabulary = set(predicted_vocabulary)
    return truth_vocabulary - predicted_vocabulary

from owlready2 import *
from tree import Tree
import pickle
import itertools


wordmap = {}
onto = get_ontology("https://raw.githubusercontent.com/HICAI-ZJU/KANO/main/KGembedding/elementkg.owl").load()

# Create tree node for all classes and instances
for c in onto.classes():
    t = Tree(c.name)
    wordmap[c.name] = t
    t.synonyms = [c.name]
    for ins in c.instances():
        t = Tree(ins.name)
        wordmap[ins.name] = t
        t.synonyms = [ins.name]

# add edges to create tree structure
def add_edges(wordmap, tree, owlclass):
    for sub in owlclass.subclasses():
        tree.add_child(wordmap[sub.name])
        add_edges(wordmap, wordmap[sub.name], sub)
    if len([x for x in owlclass.subclasses()]) == 0:
        for ins in owlclass.instances():
            tree.add_child(wordmap[ins.name])


def remove_self_loops_tree(tree):
    for c in tree.children:
        c.children = [cc for cc in c.children if cc != tree]
        c = remove_self_loops_tree(c)
    return tree


roots = []
for c in onto.classes():
    add_edges(wordmap, wordmap[c.name], c)
    if len([x for x in c.ancestors()]) == 2:
        roots.append(wordmap[c.name])

tree = Tree('Thing')
for root in roots:
    tree.add_child(root)
tree = remove_self_loops_tree(tree)

with open(GT_TREE, 'wb') as f:
    pickle.dump(tree, f)


from pathlib import Path
from utils import read_tuples_list_from_csv

path = 'docs/plain_text'

# get all txt files
files = []
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(Path(r) / file)

terms = set()        
for file in files:
    csv = file.with_suffix('.csv')
    list_of_tuples = read_tuples_list_from_csv(csv)
    for t in list_of_tuples:
        terms.add(t[0])
    csv = file.with_name(file.stem + '.acronyms.csv')
    list_of_tuples = read_tuples_list_from_csv(csv)
    for t in list_of_tuples:
        terms.add(t[0])
        terms.add(t[1])

detected_vocabulary = list(terms)
detected_vocabulary

from pathlib import Path
from utils import read_tuples_list_from_csv

path = 'docs/plain_text'

# get all txt files
files = []
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(Path(r) / file)

rels = []
for file in files:
    try:
        csv = file.with_name(file.stem + '.relationships.csv')
        list_of_tuples = read_tuples_list_from_csv(csv)
        rels += list_of_tuples
    except FileNotFoundError:
        pass

print("len", len(rels))

detected_vocabulary = list(set(detected_vocabulary))
detected_vocabulary = [x.replace(' ', '') for x in detected_vocabulary]
detected_vocabulary = [x.lower() for x in detected_vocabulary]

elementkg_vocabulary = list(wordmap.keys())
elementkg_vocabulary = [x.lower() for x in elementkg_vocabulary]

# fix typos and common terminology
elementkg_vocabulary.remove('noblegasses')
elementkg_vocabulary.append('noblegases')

print("Term extraction accuracy: ", accuracy(elementkg_vocabulary, detected_vocabulary))
output(['Term extraction accuracy', accuracy(elementkg_vocabulary, detected_vocabulary)])

############################################################
# Evaluate taxonomy reconstruction
############################################################

from tree import Tree, lemma
import pickle

with open(EVAL_TREE, 'rb') as f:
    reconstructed_tree = pickle.load(f)
with open(GT_TREE, 'rb') as f:
    truth_tree = pickle.load(f)

def construct_wordmap(wordmap, tree, visited):
    if tree in visited:
        return
    visited.add(tree)
    for c in tree.children:
        construct_wordmap(wordmap, c, visited)
    for syn in tree.synonyms:
        wordmap[syn] = tree

# create lower case words too
def reduced(word):
    res = lemma(word.lower().replace(' ', ''))
    if type(res) == tuple:
        joined = ''.join(list(res))
    if type(res) == list:
        joined = ''.join(res)
    return joined
        


def _merge_two_nodes(source, destination):

    # check it does not create self loop
    if source == destination:
        raise Exception("Cannot merge a node with itself")

    # check that it does not create a loop
    destination_children = destination.list_all_childs()
    if source in destination_children:
        raise Exception("Cannot merge a node with a child of itself")

    for syn in source.synonyms:
        if syn not in destination.synonyms:
            destination.synonyms.append(syn)
    for child in source.children:
        if child not in destination.children:
            destination.add_child(child)

wordmap = {}
construct_wordmap(wordmap, reconstructed_tree, set())

keys_list = list(wordmap.keys())
for word in keys_list:
    if reduced(word) in wordmap:
        try:
            _merge_two_nodes(wordmap[word], wordmap[reduced(word)])
        except Exception as e:
            pass
    wordmap[reduced(word)] = wordmap[word]
    wordmap[reduced(word)].synonyms.append(reduced(word))


def _quick_fix_tree(tree, visited, level, path):
    if tree in visited:
        print("Loop detected")
        print(path)
        print(tree.synonyms)
        raise Exception("Loop detected")
    #print("vsiit", level, tree.synonyms)
    visited.add(tree)
    to_remove = []
    for c in tree.children:
        try:
            _quick_fix_tree(c, visited, level + 1, path + [tree.synonyms[0]])
        except:
            to_remove.append(c)
    for c in to_remove:
        tree.children.remove(c)
    visited.remove(tree)
    for syn in tree.synonyms:
        wordmap[syn] = tree

visited = set()
_quick_fix_tree(reconstructed_tree, visited, 0, [])


truth_wordmap = {}
construct_wordmap(truth_wordmap, truth_tree, set())

# fix some common terminology
if 'Element' in wordmap and wordmap['Element of the periodic table'] != wordmap['Element']:
    reconstructed_tree.remove_node(wordmap['Element'])
wordmap['Element of the periodic table'].synonyms += ['Element', 'element']
wordmap['Element'] = wordmap['Element of the periodic table']
wordmap['element'] = wordmap['Element of the periodic table']

if 'noblegas' in wordmap:
    wordmap['noblegas'].synonyms += ['noblegasses']

if 'hydrocarbon' in wordmap:
    wordmap['hydrocarbon'].synonyms += ['hydrocarbons']
    #_merge_two_nodes(wordmap['hydrocarbon'], wordmap['hydrocarbons'])

if 'element' in truth_wordmap:
    truth_wordmap['element'].synonyms += ['chemicalelement']
    #_merge_two_nodes(wordmap['element'], wordmap['chemicalelement'])

if 'chemicalelement' in truth_wordmap:
    truth_wordmap['element'].synonyms += ['chemicalelement']
    #_merge_two_nodes(wordmap['element'], wordmap['chemicalelement'])

if 'elementoftheperiodictable' in wordmap:
    wordmap['elementoftheperiodictable'].synonyms += ['element']
    #_merge_two_nodes(wordmap['elementoftheperiodictable'], wordmap['element'])



def merge_nodes(tree, visited):
    if tree in visited:
        return
    visited.add(tree)
    for syn in tree.synonyms:
        if ' (' in syn:
            #tree.synonyms.remove(syn)
            reduced_name = syn.split(' (')[0]
            if reduced_name in wordmap:
                if wordmap[reduced_name] == tree:
                    continue
                print("Merging", syn, "with", reduced_name)
                # merge the two nodes
                _merge_two_nodes(tree, wordmap[reduced_name])
    
    for child in tree.children:
        merge_nodes(child, visited)


merge_nodes(reconstructed_tree, set())                    

def _get_all_appearing_terms(tree, ctx, visited):
    res = []
    if tree in visited:
        return []
    visited.add(tree)
    for c in tree.children:
        res += _get_all_appearing_terms(c, ctx, visited)
    if ctx is None:
        res += [tree.synonyms[0]]
    else:
        found = []
        for syn in tree.synonyms:
            if syn in ctx:
                found.append(syn)
        if len(found) == 0:
            res += [tree.synonyms[0]]
        else:
            res += found
    visited.remove(tree)
    return res

def get_all_appearing_terms(tree, ctx):
    return _get_all_appearing_terms(tree, ctx, set())

#truth_vocab = truth_tree.get_terms()
truth_vocab = get_all_appearing_terms(truth_tree, "")
truth_vocab = [reduced(x) for x in truth_vocab]
truth_vocab_ctx = ' '.join(truth_vocab)

reconstructed_vocab = get_all_appearing_terms(reconstructed_tree, truth_vocab_ctx)
reconstructed_vocab = [reduced(x) for x in reconstructed_vocab]

print("Term extraction accuracy: ", accuracy(truth_vocab, reconstructed_vocab))
output(['Term extraction accuracy', accuracy(truth_vocab, reconstructed_vocab)])
#missing_terms(truth_vocab, reconstructed_vocab)

truth_vocab = truth_wordmap['element'].get_terms()
truth_vocab = [reduced(x) for x in truth_vocab]
truth_vocab_ctx = ' '.join(truth_vocab)

reconstructed_vocab = get_all_appearing_terms(wordmap['element'], truth_vocab_ctx)
reconstructed_vocab = [reduced(x) for x in reconstructed_vocab]

print("Term extraction elements accuracy: ", accuracy(truth_vocab, reconstructed_vocab))
output(['Term extraction elements accuracy', accuracy(truth_vocab, reconstructed_vocab)])
#missing_terms(truth_vocab, reconstructed_vocab)

truth_vocab = truth_wordmap['functionalGroup'].get_terms()
truth_vocab = [reduced(x) for x in truth_vocab]
truth_vocab_ctx = ' '.join(truth_vocab)

reconstructed_vocab = get_all_appearing_terms(wordmap['functionalgroup'], truth_vocab_ctx)
reconstructed_vocab = [reduced(x) for x in reconstructed_vocab]

print("Term extraction funcional group accuracy: ", accuracy(truth_vocab, reconstructed_vocab))
output(['Term extraction funcional group accuracy', accuracy(truth_vocab, reconstructed_vocab)])
#missing_terms(truth_vocab, reconstructed_vocab)


ancestors = {}

ans = truth_tree.get_all_ancestors(truth_wordmap['Fe'])
ans = set(ans)
# remove thing
ans.remove(truth_wordmap['Thing'])
ans = [x._get_ctx_term(x.synonyms, truth_vocab_ctx) for x in ans]
ans

ctx = ' '.join(truth_vocab)

#target_tree = wordmap['nonferrousmetal']
#ancestors = reconstructed_tree.get_all_ancestors(target_tree)
#ancestors = [x._get_ctx_term(x.synonyms, ctx) for x in ancestors]
#ancestors

truth_vocab = truth_tree.get_terms()
truth_vocab = [reduced(x) for x in truth_vocab]
ctx = ' '.join(truth_vocab)
ctx

def get_all_paths_tree(self, visited, ctx=None):
    res = []
    path = []

    for c in self.children:
        terms, p = get_all_paths_tree(c, visited, ctx)
        res += terms
        for p_ in p:
            path.append([reduced(self._get_ctx_term(self.synonyms, ctx))] + p_)

    term = reduced(self._get_ctx_term(self.synonyms, ctx))
    res.append(term)
    path.append([term])
    return res, path

def _get_set_reconstructed_ancestors(tree, term, wordmap, ctx):
    if term not in wordmap:
        return None
    target_tree = wordmap[term]
    ancestors = tree.get_all_ancestors(target_tree)
    ancestors = [x._get_ctx_term(x.synonyms, ctx) for x in ancestors]
    ancestors = set(ancestors)
    if 'thing' in ancestors:
        ancestors.remove('thing')
    return ancestors

def get_all_ancestors_tree(self, tree_root, visited, ctx=None):
    res = {}
    #if self in visited:
    #    return res
    #visited.add(self)
    for c in self.children:
        res.update(get_all_ancestors_tree(c, tree_root, visited, ctx))
    term = reduced(self._get_ctx_term(self.synonyms, ctx))
    ancestors = tree_root.get_all_ancestors(self)
    ancestors = [reduced(x._get_ctx_term(x.synonyms, ctx)) for x in ancestors]
    ancestors = set(ancestors)
    if 'thing' in ancestors:
        ancestors.remove('thing')
    if 'Thing' in ancestors:
        ancestors.remove('Thing')
    res[term] = ancestors
    #visited.remove(self)
    return res

word_to_paths = defaultdict(list)
gt_word_to_paths = defaultdict(list)
word_to_anc = {}
gt_word_to_anc = {}

pr_words, ll = get_all_paths_tree(wordmap['Thing'], [], ctx)
for i in range(len(pr_words)):
    word_to_paths[pr_words[i]].append(ll[i])

gt_words, ll = get_all_paths_tree(truth_wordmap['Thing'], [], ctx)
for i in range(len(gt_words)):
    gt_word_to_paths[gt_words[i]].append(ll[i])

word_to_anc = get_all_ancestors_tree(
    wordmap['Thing'], 
    wordmap['Thing'], 
    [], ctx)
gt_word_to_anc = get_all_ancestors_tree(
    truth_wordmap['Thing'],
    truth_wordmap['Thing'], 
    [], ctx)

def _max_intersection_paths(gt_path, pr_paths):
    max_intersection = 0
    best_pr = None
    for pr in pr_paths:
        intersection = len(set(gt_path).intersection(set(pr)))
        if intersection > max_intersection:
            max_intersection = intersection
            best_pr = pr
    if len(gt_path) == 0 or len(pr_paths) == 0:
        return []
    return best_pr

def hierarchical_precision(gt_paths, pred_paths):
    sum = 0.0
    count = 0.0
    for word, gt_word_paths in gt_paths.items():
        for gt_word_path in gt_word_paths:
            pr_word_paths = pred_paths.get(word, [])
            pr_word_path = _max_intersection_paths(gt_word_path, pr_word_paths)
            intersection_len = len(set(gt_word_path).intersection(set(pr_word_path)))
            prec = float(intersection_len) / len(pr_word_path) if len(pr_word_path) > 0 else 0
            sum += prec
            count += 1
    return sum / count if count > 0 else 0

def hierarchical_recall(gt_paths, pred_paths):
    sum = 0.0
    count = 0.0
    for word, gt_word_paths in gt_paths.items():
        for gt_word_path in gt_word_paths:
            pr_word_paths = pred_paths.get(word, [])
            pr_word_path = _max_intersection_paths(gt_word_path, pr_word_paths)
            intersection_len = len(set(gt_word_path).intersection(set(pr_word_path)))
            prec = float(intersection_len) / len(gt_word_path) if len(pr_word_path) > 0 else 0
            sum += prec
            count += 1
    return sum / count if count > 0 else 0

def hierarchical_accuracy(words, gt_anc, pred_anc):
    sum = 0.0
    count = 0.0
    for word in words:
        gt_word_anc = gt_anc[word]
        pr_word_anc = pred_anc.get(word, set())
        # filter pr_word_anc to only include those terms present in gold taxonomy
        pr_word_anc = set([x for x in pr_word_anc if x in gt_anc])
        intersection_len = len(gt_word_anc.intersection(pr_word_anc))
        if intersection_len > 0:
            sum += 1.0
        count += 1
    return sum / count if count > 0 else 0

def f1(hp, hr):
    return 2 * hp * hr / (hp + hr) if (hp + hr) > 0 else 0

def ancestor_error_count(gt_words, gt_anc, pred_anc):
    sum = 0.0
    count = 0.0
    for word in gt_words:
        gt_word_anc = gt_anc[word]
        pr_word_anc = pred_anc.get(word, set())
        # filter pr_word_anc to only include those terms present in gold taxonomy
        pr_word_anc = set([x for x in pr_word_anc if x in gt_anc])
        intersection_len = len(gt_word_anc.intersection(pr_word_anc))
        err = len(gt_word_anc) + len(pr_word_anc) - 2 * intersection_len
        sum += err
        count += 1
    return sum / count if count > 0 else 0

def lowest_common_ancestor(gt_paths, pred_paths):
    sum = 0.0
    count = 0.0
    for word1, gt_word_paths1 in gt_paths.items():
        for gt_word_path1 in gt_word_paths1:

            pr_word_paths1 = pred_paths.get(word1, [])
            pr_word_path1 = _max_intersection_paths(gt_word_path1, pr_word_paths1)

            for word2, gt_word_paths2 in gt_paths.items():
                for gt_word_path2 in gt_word_paths2:

                    if word1 == word2:
                        continue

                    pr_word_paths2 = pred_paths.get(word2, [])
                    pr_word_path2 = _max_intersection_paths(gt_word_path2, pr_word_paths2)

                    # find common anc in gt (list is ordered left->right from top to bottom)
                    gt_common_anc = None
                    # reversed list
                    for idx in range(len(gt_word_path1) - 1, -1, -1):
                        if gt_word_path1[idx] in gt_word_path2:
                            gt_common_anc = gt_word_path1[idx]
                            break

                    # find common anc in pred
                    pr_common_anc = None
                    # reversed list
                    for idx in range(len(pr_word_path1) - 1, -1, -1):
                        if pr_word_path1[idx] in pr_word_path2:
                            pr_common_anc = pr_word_path1[idx]
                            break
                    
                    if gt_common_anc == pr_common_anc:
                        sum += 1
                    count += 1
    return sum / count if count > 0 else 0

hp = hierarchical_precision(gt_word_to_paths, word_to_paths)
hr = hierarchical_recall(gt_word_to_paths, word_to_paths)
f1s = f1(hp, hr)
ha = hierarchical_accuracy(gt_words, gt_word_to_anc, word_to_anc)
aec = ancestor_error_count(gt_words, gt_word_to_anc, word_to_anc)
lca = lowest_common_ancestor(gt_word_to_paths, word_to_paths)

print("Hierarchical Precision: ", hp)
print("Hierarchical Recall: ", hr)
print("Hierarchical F1: ", f1s)
print("Hierarchical Accuracy: ", ha)
print("Ancestor error count: ", aec)
print("Lowest common ancesto: ", lca)
output(['Hierarchical Precision', hp])
output(['Hierarchical Recall', hr])
output(['Hierarchical F1', f1s])    
output(['Hierarchical Accuracy', ha])
output(['Ancestor error count', aec])
output(['Lowest common ancesto', lca])

word_to_paths = defaultdict(list)
gt_word_to_paths = defaultdict(list)
word_to_anc = {}
gt_word_to_anc = {}

pr_words, ll = get_all_paths_tree(wordmap['Element'], [], ctx)
for i in range(len(pr_words)):
    word_to_paths[pr_words[i]].append(ll[i])

gt_words, ll = get_all_paths_tree(truth_wordmap['element'], [], ctx)
for i in range(len(gt_words)):
    gt_word_to_paths[gt_words[i]].append(ll[i])

word_to_anc = get_all_ancestors_tree(
    wordmap['Element'], 
    wordmap['Thing'], 
    [], ctx)
gt_word_to_anc = get_all_ancestors_tree(
    truth_wordmap['element'],
    truth_wordmap['Thing'], 
    [], ctx)

hp = hierarchical_precision(gt_word_to_paths, word_to_paths)
hr = hierarchical_recall(gt_word_to_paths, word_to_paths)
f1s = f1(hp, hr)
ha = hierarchical_accuracy(gt_words, gt_word_to_anc, word_to_anc)
aec = ancestor_error_count(gt_words, gt_word_to_anc, word_to_anc)
lca = lowest_common_ancestor(gt_word_to_paths, word_to_paths)

print("(E)Hierarchical Precision: ", hp)
print("(E)Hierarchical Recall: ", hr)
print("(E)Hierarchical F1: ", f1s)
print("(E)Hierarchical Accuracy: ", ha)
print("(E)Ancestor error count: ", aec)
print("(E)Lowest common ancesto: ", lca)
output(['(E)Hierarchical Precision', hp])
output(['(E)Hierarchical Recall', hr])
output(['(E)Hierarchical F1', f1s])
output(['(E)Hierarchical Accuracy', ha])
output(['(E)Ancestor error count', aec])
output(['(E)Lowest common ancesto', lca])

word_to_paths = defaultdict(list)
gt_word_to_paths = defaultdict(list)
word_to_anc = {}
gt_word_to_anc = {}

pr_words, ll = get_all_paths_tree(wordmap['Functional group'], [], ctx)
for i in range(len(pr_words)):
    word_to_paths[pr_words[i]].append(ll[i])

gt_words, ll = get_all_paths_tree(truth_wordmap['functionalGroup'], [], ctx)
for i in range(len(gt_words)):
    gt_word_to_paths[gt_words[i]].append(ll[i])

word_to_anc = get_all_ancestors_tree(
    wordmap['Functional group'], 
    wordmap['Thing'], 
    [], ctx)
gt_word_to_anc = get_all_ancestors_tree(
    truth_wordmap['functionalGroup'],
    truth_wordmap['Thing'], 
    [], ctx)

hp = hierarchical_precision(gt_word_to_paths, word_to_paths)
hr = hierarchical_recall(gt_word_to_paths, word_to_paths)
f1s = f1(hp, hr)
ha = hierarchical_accuracy(gt_words, gt_word_to_anc, word_to_anc)
aec = ancestor_error_count(gt_words, gt_word_to_anc, word_to_anc)
lca = lowest_common_ancestor(gt_word_to_paths, word_to_paths)

print("(FG)Hierarchical Precision: ", hp)
print("(FG)Hierarchical Recall: ", hr)
print("(FG)Hierarchical F1: ", f1s)
print("(FG)Hierarchical Accuracy: ", ha)
print("(FG)Ancestor error count: ", aec)
print("(FG)Lowest common ancesto: ", lca)
output(['(FG)Hierarchical Precision', hp])
output(['(FG)Hierarchical Recall', hr])
output(['(FG)Hierarchical F1', f1s])
output(['(FG)Hierarchical Accuracy', ha])
output(['(FG)Ancestor error count', aec])
output(['(FG)Lowest common ancesto', lca])

# KG acc
truth_leaves = truth_tree.get_leaf_nodes()
reconstructed_leaves = reconstructed_tree.get_leaf_nodes()
truth_leaves = set([reduced(x.synonyms[0]) for x in truth_leaves])
reconstructed_leaves = set([reduced(x._get_ctx_term(x.synonyms, truth_vocab_ctx)) for x in reconstructed_leaves])
print("Leaf nodes accuracy: ", accuracy(truth_leaves, reconstructed_leaves))
output(['Leaf nodes accuracy', accuracy(truth_leaves, reconstructed_leaves)])

# KG acc
truth_leaves = truth_wordmap['element'].get_leaf_nodes()
reconstructed_leaves = wordmap['element'].get_leaf_nodes()
truth_leaves = set([reduced(x.synonyms[0]) for x in truth_leaves])
reconstructed_leaves = set([reduced(x._get_ctx_term(x.synonyms, truth_vocab_ctx)) for x in reconstructed_leaves])
print("Leaf nodes elements accuracy: ", accuracy(truth_leaves, reconstructed_leaves))
output(['Leaf nodes elements accuracy', accuracy(truth_leaves, reconstructed_leaves)])

# KG acc
truth_leaves = truth_wordmap['functionalGroup'].get_leaf_nodes()
reconstructed_leaves = wordmap['functionalgroup'].get_leaf_nodes()
truth_leaves = set([reduced(x.synonyms[0]) for x in truth_leaves])
reconstructed_leaves = set([reduced(x._get_ctx_term(x.synonyms, truth_vocab_ctx)) for x in reconstructed_leaves])
print("Leaf nodes funcitonal groups accuracy: ", accuracy(truth_leaves, reconstructed_leaves))
output(['Leaf nodes funcitonal groups accuracy', accuracy(truth_leaves, reconstructed_leaves)])

############################################################

############################################################

def _get_min_number_steps(source_tree, target_tree):
    if source_tree == target_tree:
        return 0
    min_steps = 1000000
    for c in source_tree.children:
        min_steps = min(min_steps, _get_min_number_steps(c, target_tree))
    return 1 + min_steps

def _get_num_edges_visited(tree, visited):
    if tree in visited:
        return 0
    visited.add(tree)
    if len(tree.children) == 0:
        return 0
    num_edges = 0
    for c in tree.children:
        num_edges += 1 + _get_num_edges_visited(c, visited)
    return num_edges

def _get_num_edges(tree):
    visited_edges = set()
    
    def dfs(node):
        for child in node.children:
            edge = (node, child)  # or use IDs if nodes aren't hashable
            if edge not in visited_edges:
                visited_edges.add(edge)
                dfs(child)
    
    dfs(tree)
    return len(visited_edges)
    

def compute_general_metrics(tree):
    nodes = set(tree.get_nodes_list())
    sum_num_children = 0
    for n in nodes:
        sum_num_children += len(set(n.children))
    avg_children = float(sum_num_children) / len(nodes)

    max_num_children = 0
    min_num_children = 1000000
    for n in nodes:
        max_num_children = max(max_num_children, len(set(n.children)))
        min_num_children = min(min_num_children, len(set(n.children)))

    depth = tree.get_depth()

    avg_steps_to_leaf = 0
    for n in nodes:
        avg_steps_to_leaf += _get_min_number_steps(tree, n)
    avg_steps_to_leaf = float(avg_steps_to_leaf) / len(nodes)

    num_nodes = len(nodes)
    num_leafs = len(set(tree.get_leaf_nodes()))
    num_edges = _get_num_edges(tree)

    print("Average number of children: ", avg_children)
    print("Max number of children: ", max_num_children)
    print("Min number of children: ", min_num_children)
    print("Depth: ", depth)
    print("Average steps to leaf: ", avg_steps_to_leaf)
    print("Number of nodes: ", num_nodes)
    print("Number of leafs: ", num_leafs)
    print("Number of edges: ", num_edges)
    output(['Average number of children', avg_children])
    output(['Max number of children', max_num_children])
    output(['Min number of children', min_num_children])
    output(['Depth', depth])
    output(['Average steps to leaf', avg_steps_to_leaf])
    output(['Number of nodes', num_nodes])
    output(['Number of leafs', num_leafs])
    output(['Number of edges', num_edges])

compute_general_metrics(reconstructed_tree)