import os
import errno
import pandas as pd
import numpy as np
import math
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class BissectionAnnotator:
    ''' Active Learning annotation tool that relies on Probabilistic Bisection 
        for data sampling. Parameters:
        - dataframe (pandas.DataFrame): pandas dataframe containing the processed
        data from which to sample datapoints to annotate.
        - start (float): start of the interval within which the threshold should be 
        located.
        - stop (float): end of the interval within which the threshold should be 
        located.
        - step (float): width for the discretization of the probability distribution.
        - p (float): probability associated with the direction of the threshold provided
        by the "oracle" (i.e. the annotator)
        - early_termination_weight (float): probability mass that needs to be concentrated
        with the interval "early_termination_width" for the annotation to early stop.
        - early_termination_width (float): width of the interval within which the probability
        mass "early_termination_weight" needs to be concentrated for the annotation to early stop.
        - verbose (int): verbosity of display
        - graphs_out_path (str): path to output folder where the graphs of the probability
        distribution at each annotation timestep are stored.
        - level (str): must be one of "class" or "claim". If "class", the annotation will be 
        done with respect to the class, and will be shared for all the claims relating to the 
        same class. If "claim", the annotation will be done with respect to the claim specifically,
        and will not be shared across claims relating to the same class.
        - sleep_time (int): number of seconds the interface should pause for between displaying cached
        annotations and moving on to the next annotation.
    '''
    def __init__(self, dataframe, start=0.0, stop=1.0, step=0.001, p=0.7, 
                 early_termination_width=0, early_termination_weight=0.95, 
                 verbose=2, graphs_out_path=None, level='claim', sleep_time=0
                 ):
        
        if p <= 0.5:
            raise (ValueError('the probability "p" must be strictly greater than 0.5'))
            
        if not level in ['claim', 'class']:
            raise (ValueError('parameter "level" should be one of "claim" or "class"'))
        
        # Initialising attributes of annotator
        self.df = dataframe
        self.start = start
        self.stop = stop
        self.step = step
        self.p = p
        self.q = 1 - p
        self.early_termination_width = early_termination_width
        self.early_termination_weight = early_termination_weight
        self.verbose = verbose
        self.graphs_out_path = graphs_out_path
        self.level = level
        self.sleep_time = sleep_time

        if not self.graphs_out_path is None:
            if not os.path.isdir(self.graphs_out_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.graphs_out_path)
        
        # Initialize a x-axis of (discretized) probability distribution
        self.x = np.arange(start, stop, step)
        self.x_labels = [round(e, 1) for e in np.arange(start, stop, 0.10)]
        ratio = int(0.10 / step)
        self.x_labels_loc = range(0, len(self.x), ratio)

        
    def round_down(self, score):
        multiplier = 1.0 / self.step
        return math.floor(score * multiplier) / multiplier
    
    
    def get_probability_masses(self, threshold):
        exp_f = np.exp(self.f)
        left_mass = np.sum(exp_f[self.x < threshold])
        right_mass = np.sum(exp_f[self.x >= threshold])
        return left_mass, right_mass


    def get_belief_interval(self):
        ''' Returns the interval from the probability distribution containing 
            the probability mass given by self.early_termination_weight.
        '''
        exp_f = np.exp(self.f)
        eps = 0.5 * (1 - self.early_termination_weight)
        eps = exp_f.sum() * eps

        try:
            left = self.x[exp_f.cumsum() < eps][-1]
        except IndexError:
            left = self.x[0]
        try:
            right = self.x[exp_f.cumsum() > (exp_f.sum() - eps)][0]
        except IndexError:
            right = self.x[-1]

        return left, right
    
    
    def get_median(self):
        ''' Returns the median from the probability distribution.
        '''
        exp_f = np.exp(self.f)
        alpha = exp_f.sum() * 0.5
        try:
            median_low = self.x[exp_f.cumsum() <= alpha][-1]
        except IndexError:
            return self.start
        try:
            median_high = self.x[::-1][exp_f[::-1].cumsum() < alpha][-1]
        except IndexError:
            return self.stop
        median_avg = (median_low + median_high) / 2
        return median_avg
    

    def find_best_match(self, median):
        ''' Returns: score and text of the datapoint in self.df closest 
        to the median (float) given as input.
        '''
        texts = self.df['text'].to_list()
        scores = np.array(self.df['scores_for_annot'].to_list())
        scores_diff = np.abs(scores - median)
        sorted_indices = np.argsort(scores_diff)

        for i in sorted_indices:
            if not texts[i] in self.done:
                return scores[i], texts[i]
        
        print('MUST STOP, all the data has been annotated')
        return None, None
    
    
    def get_next_datapoint(self):
        ''' Finds the new median of the probability distribution self.f
        and returns this median (median), the score of the datapoint in 
        self.df closest to the median (best_match_score), its text 
        (best_match_text) and the probability mass on the left and right 
        of the sampled datapoint respectively (left_mass, right_mass).
        '''
        median = self.get_median()
        if median == self.start or median == self.stop:
            return median, None, None, None, None
        best_match_score, best_match_text = self.find_best_match(median)
        if best_match_score is None:
            return median, None, None, None, None
        left_mass, right_mass = self.get_probability_masses(best_match_score)

        return median, best_match_score, best_match_text, left_mass, right_mass       

    
    def split_text(self, text, width=100):
        ''' Splits text over several lines for display, with each line at most
            as wide as the provided parameter "width"
        '''
        tokens = text.split()
        line = list()
        result = list()
        count = 0
        for t in tokens:
            if count > width:
                result.append(' '.join(line))
                line = list()
                count = 0
            count += len(t)
            line.append(t)
        
        result.append(' '.join(line))
        return '\n'.join(result)
    
    
    def annotation_function(self, median, score, text):
        ''' Queries user for an annotation for datapoint with text "text" 
            and score "score"
        '''
        # Display relevant information about the datapoint
        print()
        print('SCORE:\t %.3f' % score, '(diff. = %.3f)' % abs(median - score))
        print()
        print('TEXT:')
        print()
        print(self.split_text(text))
        print()
        if self.level == 'claim':
            print('\tCLASS:\t\t', self.class_text)
        print('\tCLAIM:\t', self.claim_text)
        print()
        
        # If datapoint is in cache, automatically annotate
        if text in self.cache:
            annot = self.cache[text]
            self.done.append(text)
            if annot == 1:
                if self.verbose >= 1:
                    print('TRUE')
                    time.sleep(self.sleep_time)
                return True
            elif annot == 0:
                if self.verbose >= 1:
                    print('FALSE')
                    time.sleep(self.sleep_time)
                return False

        print('\t"t" + ENTER for True')
        print('\t"f" + ENTER for False')
        print('\t"r" + ENTER to roll back to the previous annotation')
        print('\t"s" + ENTER to skip this claim and move on to the next one')
        print('\t"q" + ENTER to exit and save the annotation')
        print()
        
        # Require user input
        while True:
            try:
                annot = input()
            except KeyboardInterrupt:
                annot = 'q'
            if annot == 't':
                self.cache[text] = 1
                self.done.append(text)
                return True
            elif annot == 'f':
                self.cache[text] = 0
                self.done.append(text)
                return False
            elif annot == 'r':
                if self.verbose >= 1:
                    print('ROLLING BACK TO PREVIOUS ANNOT')
                return annot
            elif annot == 's':
                if self.verbose >= 1:
                    print('SKIPPING THIS CLAIM')
                return annot
            elif annot == 'q':
                if self.verbose >= 1:
                    print('PAUSING AND SAVING ANNOTATION')
                return annot
            else:
                print('Sorry, this is not a valid option. Please enter one of "t", "f", "r", "s" or "c".')
                
                
    def probabilistic_bisection(self):
        ''' Main active learning annotation loop, which uses Probabilistic 
        Bisection to sample the next datapoint for annotation.
        '''
        if not self.graphs_out_path is None:
            graph_path = os.path.join(self.graphs_out_path, 'bissection_graphs')
            if not os.path.exists(graph_path):
                os.mkdir(graph_path)

        round_nb = 0
        tbc = False
        early_stop = False

        while True:
            
            # Remove previous prints
            clear_output()
            
            round_nb += 1

            # Assess the width of the interval concentrating a probability equal to self.early_termination_weight,
            # and early stop if narrower than self.early_termination_width
            belief_interval = self.get_belief_interval()
            if (belief_interval[1] - belief_interval[0]) <= self.early_termination_width:
                break
            
            # Get median of the probability distribution at current timestep, the closest datapoint to the median
            # (both NLI score and text of datapoint) and resulting probability mass on the left and right of the 
            # sampled datapoint.
            median, actual_median, text, left_mass, right_mass = self.get_next_datapoint()
            
            # If no datapoint was sampled due to data sparsity, early stop.
            if actual_median is None:
                if median == self.start:
                    if self.verbose >= 1:
                        print('NEED TO STOP ANNOTATING! Reached lower bound of the threshold range.')
                        time.sleep(2)
                else:
                    if self.verbose >= 1:
                        print('NEED TO STOP ANNOTATING! Reached higher bound of the threshold range.')
                        time.sleep(2)
                break
            
            if self.verbose >= 2:
                # Print relevant information about current state of probability distribution self.f
                print('===============================')
                print()
                print('ANNOT NB:', round_nb)
                print('MEDIAN: %.3f' % median)
                print()
                interval = belief_interval[1] - belief_interval[0]
                print('INTERVAL: %.3f' % belief_interval[0], '- %.3f' % belief_interval[1], '\t==> WIDTH: %.3f' % interval)
                print('TOTAL =', sum(np.exp(self.f)))
                print('LEFT MASS = %.3f' % left_mass, '\tRIGHT MASS = %.3f' % right_mass, '\tTOTAL = %.3f' % (left_mass + right_mass))
                print()
            
            if self.verbose >= 2:
                # Display bar plot of current state of probability distribution self.f
                sns.barplot(x=self.x, y=np.exp(self.f), color="cornflowerblue")
                plt.xticks(self.x_labels_loc, self.x_labels)
                plt.ylabel("Probability")
                plt.xlabel("Score")
                plt.title(self.claim_idx + ': ' + self.claim_text)
                if not self.graphs_out_path is None:
                    # Save plot to target graph folder
                    fig = plt.gcf()
                    fig.savefig(graph_path + '/round_' + str(round_nb) + '.png')
                plt.show()
            
            if max(left_mass, right_mass) > self.p:
                # If no datapoint was sampled due to data sparsity, early stop
                if self.verbose >= 1:
                    print('NEED TO STOP ANNOTATING! Data is too sparse')
                    time.sleep(2)
                break

            else:
                # Get annotation from "oracle" for the sampled datapoint
                z = self.annotation_function(median, actual_median, text)
            
            if z == 'r':
                # Rollback to previous annotation
                self.fs.pop()
                self.f = self.fs[-1].copy()
                last_item = self.done.pop()
                del self.cache[last_item]
                continue
            elif z == 's':
                # End annotation for current claim and move on to the next claim
                tbc = True
                break
            elif z == 'q':
                # Quit the annotation altogether
                tbc = True
                early_stop = True
                break
            
            # Add information of newly annotated datapoint to annotation record
            self.texts_list.append(text)
            self.medians_list.append(median)
            self.scores_list.append(actual_median)
            self.annot_list.append(z)

            # Update probability distribution self.f according to the annotation 
            # provided by the oracle
            if z == False:
                self.f[self.x >= actual_median] += (np.log(self.p) - np.log(right_mass))
                self.f[self.x < actual_median] += (np.log(self.q) - np.log(left_mass))
            else:
                self.f[self.x >= actual_median] += (np.log(self.q) - np.log(right_mass))
                self.f[self.x < actual_median] += (np.log(self.p) - np.log(left_mass))
            
            # Add current state of probability distribution to record
            self.fs.append(self.f.copy())

        
        # Print current best estimate of threshold, i.e. median of current probability distribution
        if self.verbose >= 1:
            print()
            print('*********************************')
            print('***  THRESHOLD FOUND: %.3f'% median, '  ***')
            print('*********************************')
            print()
        
        time.sleep(3)
        
        return tbc, early_stop
    
    
    def __call__(self, claim_idx, claim_text, class_idx, class_text, column='ZSL_scores'):
        ''' Main function called by user to annotate a claim. Input:
        - claim_idx (str): ID of the claim to annotate
        - claim_text (str): text of claim to annotate
        - class_idx (str): ID of the class the claim relates to 
        - class_text (str): name of the class the claim relates to
        '''
        print('====== ANNOTATING:', claim_text, '(' + claim_idx + ') ======')
        
        # Initialise relevant attributes
        self.claim_idx = claim_idx
        self.claim_text = claim_text
        self.class_text = class_text
        self.class_idx = class_idx
        
        # Initialise an empty record for the annotation
        self.texts_list = list()
        self.medians_list = list()
        self.scores_list = list()
        self.annot_list = list()
        
        # Single out the scores for the claim to annotate in a new column of the dataframe self.df
        self.df['scores_for_annot'] = self.df[column].apply(lambda x: x[claim_text])
        
        if self.level == 'class':
            annot_idx = class_idx
        elif self.level == 'claim':
            annot_idx = claim_idx
        
        # Create cache from dataframe
        if annot_idx + '_annot' in self.df.columns:
            rel_df = self.df[self.df[annot_idx + '_annot'].apply(lambda x: x == 0 or x == 1)]
            if len(rel_df) == 0:
                self.cache = dict()
            else:
                list_annot = list(rel_df.apply(lambda x: (x['text'], x[annot_idx + '_annot']), axis=1))
                self.cache = dict(list_annot)
        else:
            self.cache = dict()
        
        self.done = list()
        
        # Initialise uniform probability distribution for the threshold of current claim
        f = np.ones(len(self.x))
        f /= np.sum(f)
        self.f = np.log(f)
        self.fs = [self.f.copy()]
        
        # Execute Probabilistic Bisection
        tbc, early_stop = self.probabilistic_bisection()
        
        # Save cache into dataframe
        self.df[class_idx + '_annot'] = self.df['text'].apply(lambda x: '' if not x in self.cache else (1 if self.cache[x] == 1 else (0 if self.cache[x] == 0 else '')))
        new_cache = pd.DataFrame()
        text_list = list(self.cache.keys())
        annot_list = list(self.cache.values())
        new_cache['text'] = text_list
        new_cache[annot_idx + '_annot'] = annot_list
        
        # Save annotation record as dictionary
        distr_list = [list(f) for f in self.fs]
        out_dict = {
            'texts': self.texts_list,
            'medians': self.medians_list, 
            'scores': self.scores_list,
            'annot': self.annot_list,
            'distributions': distr_list,
            'continue': tbc
                    }
        
        return new_cache, out_dict, early_stop