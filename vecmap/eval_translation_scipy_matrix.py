# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import embeddings
from cupy_utils import *

import argparse
import collections
import numpy as np
import sys
import os
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.append('/data/xyyf/EVA')
import model_info
from model_info import models

import pickle

BATCH_SIZE = 500



def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('src_embeddings', help='the source language embeddings')
    parser.add_argument('trg_embeddings', help='the target language embeddings')
    parser.add_argument('-d', '--dictionary', default=sys.stdin.fileno(), help='the test dictionary file (defaults to stdin)')
    parser.add_argument('--retrieval', default='nn', choices=['nn', 'invnn', 'invsoftmax', 'csls'], help='the retrieval method (nn: standard nearest neighbor; invnn: inverted nearest neighbor; invsoftmax: inverted softmax; csls: cross-domain similarity local scaling)')
    parser.add_argument('--inv_temperature', default=1, type=float, help='the inverse temperature (only compatible with inverted softmax)')
    parser.add_argument('--inv_sample', default=None, type=int, help='use a random subset of the source vocabulary for the inverse computations (only compatible with inverted softmax)')
    parser.add_argument('-k', '--neighborhood', default=10, type=int, help='the neighborhood size (only compatible with csls)')
    parser.add_argument('--dot', action='store_true', help='use the dot product in the similarity computations instead of the cosine')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--seed', type=int, default=0, help='the random seed')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    args = parser.parse_args()

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # Read input embeddings
    srcfile = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)

    # NumPy/CuPy management
    if args.cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        x = xp.asarray(x)
        z = xp.asarray(z)
    else:
        xp = np
    xp.random.seed(args.seed)

    # Length normalize embeddings so their dot product effectively computes the cosine similarity
    if not args.dot:
        embeddings.length_normalize(x)
        embeddings.length_normalize(z)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # Read dictionary and compute coverage
    f = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')
    src2trg = collections.defaultdict(set)
    oov = set()
    vocab = set()
    for line in f:
        src, trg = line.split()
        try:
            src_ind = src_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src2trg[src_ind].add(trg_ind)
            vocab.add(src)
        except KeyError:
            oov.add(src)
    src = list(src2trg.keys())
    oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
    coverage = len(src2trg) / (len(src2trg) + len(oov))

    # change by xyyf
    all_similarities = lil_matrix((x.shape[0],z.shape[0]) ,dtype=dtype)
    k = 10
    threshold = 0.1
    translation = dict()#collections.defaultdict(list)
    similarity_score = dict()#collections.defaultdict(list)
    
    if args.retrieval == 'nn':  # Standard nearest neighbor
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = x[src[i:j]].dot(z.T) # [batch*4096] * [4096*32000] = [batch*32000]
            nn = similarities.argmax(axis=1).tolist() #[batch*1]
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
    elif args.retrieval == 'invnn':  # Inverted nearest neighbor
        best_rank = np.full(len(src), x.shape[0], dtype=int)
        best_sim = np.full(len(src), -100, dtype=dtype)
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            similarities = z[i:j].dot(x.T)
            ind = (-similarities).argsort(axis=1)
            ranks = asnumpy(ind.argsort(axis=1)[:, src])
            sims = asnumpy(similarities[:, src])
            for k in range(i, j):
                for l in range(len(src)):
                    rank = ranks[k-i, l]
                    sim = sims[k-i, l]
                    if rank < best_rank[l] or (rank == best_rank[l] and sim > best_sim[l]):
                        best_rank[l] = rank
                        best_sim[l] = sim
                        translation[src[l]] = k
    elif args.retrieval == 'invsoftmax':  # Inverted softmax
        sample = xp.arange(x.shape[0]) if args.inv_sample is None else xp.random.randint(0, x.shape[0], args.inv_sample)
        partition = xp.zeros(z.shape[0])
        for i in range(0, len(sample), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(sample))
            partition += xp.exp(args.inv_temperature*z.dot(x[sample[i:j]].T)).sum(axis=1)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            p = xp.exp(args.inv_temperature*x[src[i:j]].dot(z.T)) / partition
            nn = p.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
    elif args.retrieval == 'csls':  # Cross-domain similarity local scaling
        knn_sim_bwd = xp.zeros(z.shape[0], dtype=dtype) #[32000]
        #aux模型的special token（012）特殊处理，从3开始
        for i in range(3, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=args.neighborhood, inplace=True)
        for i in range(3, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
            # change by xyyf
            similarities = similarities.get()#cupy.ndarray -> numpy.ndarray
            for idx in range(j-i):
                row = similarities[idx]#numpy.ndarray
                #取top-k个相似度分数和对应indices
                top_k_indices_before = np.argpartition(row, -k)[-k:]#numpy.ndarray
                top_k_values_before = row[top_k_indices_before]#numpy.ndarray
                #main模型的special token（012）特殊处理，映射到的置零
                values_to_remove = [0, 1, 2]
                keep_indices = np.isin(top_k_indices_before, values_to_remove, invert=True)
                top_k_indices = top_k_indices_before[keep_indices]
                top_k_values = top_k_values_before[keep_indices]
                #阈值截断
                keep_indices = top_k_values >= threshold
                top_k_indices = top_k_indices[keep_indices]
                top_k_values = top_k_values[keep_indices]
                #方差截断(这部分是无意义的token)
                std = np.std(top_k_values)
                keep_num = top_k_indices.size
                if keep_num != 0:
                    if std <= 0.01 and keep_num > 5:
                        pass
                    else:
                        translation[src[i+idx]] = top_k_indices.tolist()
                        similarity_score[src[i+idx]] = top_k_values.tolist()
                        for ii, jj in enumerate(top_k_indices):
                            all_similarities[idx+i, jj] = top_k_values[ii]
        #special token映射补充
        for i in range(3):
            translation[i] = [i]
            similarity_score[i] = [1.0]
            all_similarities[i, i] = 1.0
    #保存映射矩阵
    all_similarities = all_similarities.tocsr()
    scipy.sparse.save_npz('/data/xyyf/EVA/sparse_matrix_filter/'+args.dictionary.split('/')[-1].split('-test')[0]+'_top-'+str(k)+'.npz', all_similarities)
    print('Finish similarity matrix')


if __name__ == '__main__':
    main()
