#!/usr/bin/python2.7

import numpy as np
import multiprocessing as mp
import queue
from utils.dataset_didt import Dataset
from utils.network import Forwarder
from utils.grammar import PathGrammar
from utils.length_model import PoissonModel
from utils.viterbi import Viterbi
import click
import os

NUM_ITERS = 10000

### helper function for parallelized Viterbi decoding ##########################
def decode(queue, log_probs, decoder, index2label, result_root):
    while not queue.empty():
        try:
            video = queue.get(timeout = 3)
            score, labels, segments = decoder.decode( log_probs[video] )
            # save result
            with open('%s/' % result_root + video, 'w') as f:
                f.write( '### Recognized sequence: ###\n' )
                f.write( ' '.join( [index2label[s.label] for s in segments] ) + '\n' )
                f.write( '### Score: ###\n' + str(score) + '\n')
                f.write( '### Frame level recognition: ###\n')
                f.write( ' '.join( [index2label[l] for l in labels] ) + '\n' )
        except queue.Empty:
            pass


@click.command()
@click.argument('data-root', type=str)
@click.argument('result-root', type=str)
@click.argument('split', type=int)
@click.option('--seed', type=int, default=0)
def main(data_root, result_root, split, seed):
    result_root += "-s-%d-%d" % (split, seed)

    ### read label2index mapping and index2label mapping ###########################
    label2index = dict()
    index2label = dict()
    with open(os.path.join(data_root, 'mapping.txt'), 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            label2index[line.split()[1]] = int(line.split()[0])
            index2label[int(line.split()[0])] = line.split()[1]

    ### read test data #############################################################
    #with open('data/split1.test', 'r') as f:
    with open(os.path.join(data_root, 'split%d.test' % split), 'r') as f:
        video_list = f.read().split('\n')[0:-1]
    dataset = Dataset(data_root, video_list, label2index, shuffle = False)

    # load prior, length model, grammar, and network
    load_iteration = NUM_ITERS
    log_prior = np.log( np.loadtxt('%s/prior.iter-' % result_root + str(load_iteration) + '.txt') )
    grammar = PathGrammar('%s/grammar.txt' % result_root, label2index)
    length_model = PoissonModel('%s/lengths.iter-' % result_root + str(load_iteration) + '.txt', max_length = 2000)
    forwarder = Forwarder(dataset.input_dimension, dataset.n_classes)
    forwarder.load_model('%s/network.iter-' % result_root + str(load_iteration) + '.net')

    # parallelization
    n_threads = 4

    # Viterbi decoder
    viterbi_decoder = Viterbi(grammar, length_model, frame_sampling = 30, max_hypotheses = np.inf)
    # forward each video
    log_probs = dict()
    queue = mp.Queue()
    for i, data in enumerate(dataset):
        sequence, _ = data
        video = list(dataset.features.keys())[i]
        queue.put(video)
        log_probs[video] = forwarder.forward(sequence) - log_prior
        log_probs[video] = log_probs[video] - np.max(log_probs[video])
    # Viterbi decoding
    procs = []
    for i in range(n_threads):
        p = mp.Process(target = decode, args = (queue, log_probs, viterbi_decoder, index2label, result_root))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()


if __name__ == '__main__':
    main()
