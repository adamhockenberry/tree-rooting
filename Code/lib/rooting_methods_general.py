import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from Bio import SeqIO

from statsmodels.stats.weightstats import DescrStatsW
import sys
sys.path.append('../../Tree_weighting/Code/')
import weighting_methods
import pairwise_weighting
from collections import defaultdict








###########################################################################################
#Mid-point rooting
###########################################################################################
def mp_root_adhock(tree):
    """
    This function implements a clever mid-point rooting algorithm that is linear with tree
    size as opposed to the behavior of many mid-point rooting algorithms that are polynomial
    by requiring all pair-wise comparisons of terminal-terminal distances. 

    The only really strange behavior that I'm aware of right now are the obvious "ties" in terms of 
    what to do when the algorithm encounters ties. For now I'm just choosing one at random, 
    where random is actually whoever pops up first in the list (so maybe not fully random).

    Input(s):
    tree - a Bio.Phylo tree object

    Output(s):
    rooted-tree - the initial tree object re-rooted
    """

    initial_bl = tree.total_branch_length()
    initial_term_names = [i.name for i in tree.get_terminals()]
    #Root randomly with an outgroup at a terminal
    tree.root_with_outgroup(tree.get_terminals()[0], outgroup_branch_length=0.0)
    
    #Through some type of bug in Bio.Phylo, new terminals can pop into existence and
    #this is a hack to remove them. I think they come from non-zero root branch lengths
    #but this should be investigated and the code can be cleaned up to flag these cases
    pruned_bls = []
    for terminal in tree.get_terminals():
        if terminal.name not in initial_term_names:
            pruned_bls.append(terminal.branch_length)
            tree.prune(terminal)
            print('Pruned a strange terminal that popped into existence during midpoint rooting')
    
    #I'm not entirely sure how this algorithm works on non-bifurcating trees. Thus, this
    #initial assertion statement. It might be fine, but I'd have to think about it.
    assert tree.is_bifurcating()
    
    #Find which terminal/s is farthest away from my randomly selected root terminal
    depths = tree.depths()
    max_depth = max(list(depths.values()))
    clade1 = [i for i,j in depths.items() if j == max_depth]
    #Idealy this would be a list of length 1. If not, there are multiple terminals
    #that are maximally distant (could be due to polytomies or I suppose random chance?)
    if len(clade1) > 1:
        print('Potential for multiple midpoints. Choosing the first that I encounter')

    #Re-root at that farthest terminal (NOTE: just choosing the first clade in the list
    #that may be longer than length 1)
    tree.root_with_outgroup(clade1[0], outgroup_branch_length=0.0)
    
    #And finally find which terminal is farthest from THAT one to identify the farthest pair
    depths = tree.depths()
    max_depth = max(list(depths.values()))
    clade2 = [i for i,j in depths.items() if j == max_depth]
    #Same constraint/caveat applies as above with regard to "ties"
    if len(clade2) > 1:
        print('Potential for multiple midpoints. Choosing the first that I encounter')
    
    #Given the clade pairs, re-root the tree
    rooted_tree = final_root(clade1[0], clade2[0], depths[clade2[0]], tree)
    
    #Ensuring that I've fully conserved branch length after all these manipulations
    #because I've had problems with gaining / losing owing to what I think are issues
    #in Bio.Phylo that I think I've fully figured out.
    assert np.isclose(initial_bl-np.sum(pruned_bls), rooted_tree.total_branch_length())
    
    return rooted_tree

def final_root(clade1, clade2, max_distance, tree):
    """
    Given the target clades, this actually re-roots the tree between them at the midpoint.

    Input(s):
    clade1 - Bio.Phylo clade object that belongs to tree
    clade2 - Bio.Phylo clade object that belongs to tree
    max_distance - the numerical distance between the two clades
    tree - Bio.Phylo tree object to be re-rooted

    Output(s):
    tree - the tree object, re-rooted
    """
    
    #Depth to go from the ingroup tip toward the outgroup tip 
    root_remainder = 0.5 * (max_distance - (tree.root.branch_length or 0)) 
    #This better be the case
    assert root_remainder >= 0 
    #Crawl between the nodes to find the middle branch
    for node in tree.get_path(clade2): 
        root_remainder -= node.branch_length 
        if root_remainder < 0: 
            outgroup_node = node 
            outgroup_branch_length = -root_remainder
            break 
    else: 
        raise ValueError("Somehow, failed to find the midpoint!")
    #Specifying the outgroup_branch_length directly with this flag lead to some
    #error-prone behavior so I'm doing it in two steps. Must be a bug and/or mis-understanding
    #in Bio.Phylo
    tree.root_with_outgroup(outgroup_node, outgroup_branch_length=0.0)
    assert outgroup_node == tree.root.clades[1]
    tree.root.clades[0].branch_length = tree.root.clades[0].branch_length + root_remainder
    tree.root.clades[1].branch_length = tree.root.clades[1].branch_length - root_remainder
    return tree


###########################################################################################
#Rooting via the minimum variance of root-to-tip distances
###########################################################################################
def MinVar_root_adhock_general(tree, weights_type=None, **kwargs):
    """ 
    This implements a rooting scheme to minimize the (weighted) variance of the root-to-tip
    distances for all leaves in the tree. It is unclear whether minimizing the variance or
    some other property of a non-normal distribution would be more appropriate but this could 
    easily be accommodated within this framework.
    
    The function is complicated by the fact that it is written to accommodate several potential
    weighting schemes. Further, when using certain weighting schemes, those weights are 
    root-dependent such that for each putative root both the depths and the weights need to be 
    updated. 
    
    Input/s:
    tree - a Bio.Phylo tree object (in practice I like to root the tree with my basic mid-point 
                algorithm first just to ensure that the tree structure is normal-ish, i.e. bifurcating)
    weights_type - There are several valid options here.
                None - no weighting (or rather, uniform weights) will be applied
                'GSC' - Gerstain-Sonnhammer-Chothia weights that need to be constantly updated
                'GSCn' - My normalized version of GSC weights, also requires updating
                'HH' - Henikoff-Henikoff weights. Calculated once (and only once) at the beginning
    kwargs - These are only necessary for Henikoff and Henikoff weights which require sequence
                information. Should pass a dictionary with the key:value being:
                    'fasta_loc': location_of_my_multiple_sequence_alignment
    
    Output/s:
    tree - the now rooted tree object
    function_optima - a list of all the function optimization output for each putative root tested
    depths_dict - a dictionary of the depths for each terminal for each putative root where: 
                clade(key): array of weights for all terminals (value)
    weights_dict - a dictionary of the weights for each terminal for each putative root where:
                clade(key): matrix of weights for all terminals (value), final column are 
                the relevant weights but the entire matrix is necessary for rapid updating

    """
    initial_depths = tree.root.depths()
    #Instantiate a depths dictionary with the root node
    depths_dict = {}
    depths_dict[tree.root] = np.array([initial_depths[i] for i in tree.get_terminals()])
    ###
    if weights_type in ['GSC', 'GSCn']:
        initial_weights = weighting_methods.GSC_adhock(tree)
        weights_array_dict = {}
        weights_array_dict[tree.root] = np.array([initial_weights[i] for i in tree.get_terminals()])
        weights_update_fxn = update_GSC_weights_dict

    elif weights_type == 'HH':
        fasta_records = list(SeqIO.parse(kwargs['fasta_loc'], 'fasta'))
        initial_weights = weighting_methods.HH_adhock(fasta_records)
        #Instantiate a weights dictionary with the root node
        weights_array = np.array([initial_weights[i.name] for i in tree.get_terminals()])
        weights_array_dict = defaultdict(lambda: weights_array)
        weights_update_fxn = no_updating
    
    elif not weights_type:
        weights_array = np.array([1. for i in tree.get_terminals()])
        weights_array_dict = defaultdict(lambda: weights_array)
        weights_update_fxn = no_updating
    
    else:
        print('Problem interpreting weights_type. See docs for valid options. Returning')
        return
    ##############################################################
    #Recursively crawl the tree visiting each possible root branch
    ##############################################################
    function_optima, depths_dict, weights_dict, finished =\
            recursive_crawl_MinVar_general(tree.root, None, [], depths_dict, weights_array_dict, [], weights_update_fxn, weights_type) 
    ##############################################################
    #And use the output to re-root the tree
    ##############################################################
    #Sort the output by the function value to find the minimum    
    function_optima = sorted(function_optima, key=lambda x: x[1].fun)
    winner = next(obj for obj in function_optima if obj[1].success==True)
    #Re-root at this point 
    tree.root_with_outgroup(winner[0], outgroup_branch_length=0.)
    assert tree.root.clades[1].branch_length == 0.
    assert tree.root.clades[1] == winner[0]
    #And adjust the branch lengths around this root
    tree.root.clades[0].branch_length -= winner[1].x[0]
    tree.root.clades[1].branch_length += winner[1].x[0]
    return tree, function_optima, depths_dict, weights_dict

def recursive_crawl_MinVar_general(my_clade, parent_clade, function_optima, depths_dict,\
                           weights_dict, finished, weights_update_fxn, weights_type):
    """
    This is a recursive function that performs the depth first search / tree crawling. It has, regrettably
    grown to require lots of inputs as I made the code more and more general to accommodate various
    weighting schemes.
    
    Input/s:
    my_clade - Bio.Phylo clade object
    parent_clade - the parent of my_clade
    function_optima - list of all the function_optima that I'm calculating
    depths_dict - dictionary of the terminal depths that will be updated and eventually include entries for 
            each putative root
    weights_dict - dictionary of the terminal weights that will be added to and will eventually include entries
            for each putative root
    finished - list of all terminals that the depth first search has completed
    weights_update_fxn - function to update weights
    weights_type - passed unchanged from MinVar_root_adhock_general. Used in updating weights when necessary

    Output/s (updated versions of each of):
    function_optima
    depths_dict
    weights_dict
    finished
    
    """
    #If the node has a parent, do some calculations (this is here just to skip calculations on the root
    #which is a special case
    if parent_clade:
        #Update the depths dictionary
        depths_dict, ds_count = update_depth_array_dict(my_clade, parent_clade, depths_dict, finished)
        #Update the weights dictionary
        weights_update_fxn(my_clade, parent_clade, weights_dict, finished)
        #Minimize the variance with these values
        if weights_type in ['GSC', 'GSCn']:
            res = optimize_root_loc_on_branch_MinVar_GSC(my_clade, depths_dict[my_clade],\
                                                 weights_dict[my_clade], ds_count, finished, weights_type)
        else:
            res = optimize_root_loc_on_branch_MinVar_general(my_clade, depths_dict[my_clade],\
                                                 weights_dict[my_clade], ds_count, finished)
        #Append results
        function_optima.append((my_clade, res))
    ###Recurse (i.e. depth first search)
    if len(my_clade.clades) == 2:
        l_clade, r_clade = my_clade.clades
        function_optima, depths_dict, weights_dict, finished =\
                recursive_crawl_MinVar_general(l_clade, my_clade, function_optima,\
                                      depths_dict, weights_dict, finished, weights_update_fxn, weights_type)
        function_optima, depths_dict, weights_dict, finished =\
                recursive_crawl_MinVar_general(r_clade, my_clade, function_optima,\
                                      depths_dict, weights_dict, finished, weights_update_fxn, weights_type)
    elif len(my_clade.clades) == 0:
        #This is quite critical, once I reach a node with no clades I'm finished so begin to 
        #backtrack and keep track of how many I've completed
        finished.append(my_clade)
        return function_optima, depths_dict, weights_dict, finished
    else:
        print('Some big error here with the number of clades stemming from this root')
    return function_optima, depths_dict, weights_dict, finished

def optimize_root_loc_on_branch_MinVar_general(my_clade, depths_array, weights_array, ds_count, finished):
    """
    For a given branch, this will take the depths and weights and optimize the exact location
    of the root for that particular branch. This function is ONLY valid for weighting schemes
    that do not change as the root location changes (currently: no weights and Henikoff-Henikoff (HH)
    weights. If this is not the case, special functions will need to be written to accommodate this.
    Example functions follow for GSC weights.
    
    Input/s:
    my_clade - Bio.Phylo clade object
    depths_array - 1D numpy array of all the depths for each terminal
    weights_array - 1D numpy array of all the weights for each terminal
    ds_count - number of downstream terminals emanating from this clade
    finished - list of all completed terminals during the depth first search
    
    Output/s:
    res - the function optima (scipy.optimize object)
    
    """
    ###################################################
    #Root-to-tip distances for all downstream terminals
    ###################################################
    downstream_dists = np.array(depths_array[len(finished):len(finished)+ds_count])
    #And all upstream terminals
    upstream_dists = np.concatenate((depths_array[:len(finished)],\
                                     depths_array[len(finished)+ds_count:]))
    
    ###################################################
    #Weights for all downstream terminals
    ###################################################
    downstream_weights = np.array(weights_array[len(finished):len(finished)+ds_count])
    #And all upstream terminals
    upstream_weights = np.concatenate((weights_array[:len(finished)],\
                                       weights_array[len(finished)+ds_count:]))
    all_weights = np.concatenate((downstream_weights, upstream_weights))
    
    ###################################################
    #Set the bounds and optimize
    ###################################################
    bl_bounds = np.array([[0., my_clade.branch_length]])
    #Valid options for method are L-BFGS-B, SLSQP and TNC
    res = minimize(branch_scan_MinVar_general, np.array(np.mean(bl_bounds)),\
                          args=(downstream_dists, upstream_dists,\
                                all_weights),\
                          bounds=bl_bounds, method='L-BFGS-B')
    return res  


    
def branch_scan_MinVar_general(modifier, ds_dists, us_dists, all_weights):
    """
    This is the function to minimize in order to optimaly situate the root on the putative
    branch. Note that this function is only valid for minimizing the variance of schemes 
    where the weights do not change with regard to changing the root.

    Input/s:
    modifier - This is the parameter to be optimized! Essentially a float of how much to shift the
                root left or right so as to minimize the root-to-tip variance
    ds_dists - array of downstream root-to-tip distances
    us_dists - array of upstream root-to-tip distances
    all_weights - array of downstream and upstream terminal weights
    
    Output/s:
    dsw.var - weighted variance
    
    """
    #Adjust the downstream and upstream root-to-tip distances with the modifier
    temp_ds_dists = ds_dists + modifier
    temp_us_dists = us_dists - modifier
    all_dists = np.concatenate((temp_ds_dists, temp_us_dists))
   
    #Calculate weighted variance and return
    dsw = DescrStatsW(all_dists, all_weights)
    return dsw.var

def optimize_root_loc_on_branch_MinVar_GSC(my_clade, depths_array, weights_array, ds_count, finished, weights_type):
    """
    See docs for "optimize_root_loc_on_branch_MinVar_general". For a given branch, this will take the 
    depths and weights and optimize the exact location of the root for that particular branch. This function 
    is SPECIFIC to GSC weighting schemes (both the initial and my normalized version).    
    
    Input/s:
    my_clade - Bio.Phylo clade object
    depths_array - 1D numpy array of all the depths for each terminal
    weights_array - 1D numpy array of all the weights for each terminal 
    ds_count - number of downstream terminals emanating from this clade
    finished - list of all completed terminals during the depth first search
    weights_type - important for decideing whether or not to normalize weights at the end
    
    Output/s:
    res - the function optima (scipy.optimize object)
    
    """
    ###################################################
    #Root-to-tip distances for all downstream terminals
    ###################################################
    downstream_dists = np.array(depths_array[len(finished):len(finished)+ds_count])
    #And all upstream terminals
    upstream_dists = np.concatenate((depths_array[:len(finished)],\
                                     depths_array[len(finished)+ds_count:]))
    
    ###################################################
    #Weights for all downstream terminals
    ###################################################
    downstream_weights = np.array(weights_array[len(finished):len(finished)+ds_count])
    #And all upstream terminals
    upstream_weights = np.concatenate((weights_array[:len(finished)],\
                                       weights_array[len(finished)+ds_count:]))
    ###################################################
    #Set the bounds and optimize the GSC specific function
    ###################################################
    bl_bounds = np.array([[0., my_clade.branch_length]])
    #Valid options for method are L-BFGS-B, SLSQP and TNC
    res = minimize(branch_scan_MinVar_GSC, np.array(np.mean(bl_bounds)),\
                          args=(downstream_dists, upstream_dists,\
                                downstream_weights, upstream_weights, weights_type),\
                          bounds=bl_bounds, method='L-BFGS-B')
    return res 


    
def branch_scan_MinVar_GSC(modifier, ds_dists, us_dists, ds_weights, us_weights, weights_type):
    """ 
    See docs for "branch_scan_MinVar_general". This is the function to minimize in order to optimaly 
    situate the root on the putative branch and is SPECIFIC to GSC weighting schemes. I should really    
    try to make this a bit quicker/simpler if possible, there might a redundant array operation or two
    but at present it works. 

    Input/s:
    modifier - This is the parameter to be optimized! Essentially a float of how much to shift the
                root left or right so as to minimize the root-to-tip variance
    ds_dists - array of downstream root-to-tip distances
    us_dists - array of upstream root-to-tip distances
    ds_weights - array of downstream terminal weights
    us_weights - array of upstream terminal weights
    weights_type - passed unchanged to determine whether or not to normalize

    Output/s:
    dsw.var - weighted variance
    
    """
    ###########################################################################
    #Adjust the downstream and upstream root-to-tip distances with the modifier
    ###########################################################################
    temp_ds_dists = ds_dists + modifier
    temp_us_dists = us_dists - modifier
    all_dists = np.concatenate((temp_ds_dists, temp_us_dists))
    
    ###########################################################################
    #Now adjust the downstream and upstream weights
    ###########################################################################
    #First get the total downstream weights
    total_ds = np.sum(ds_weights)
    #Divide up the added branch length (modifier) across the downstream weights
    if total_ds != 0:
        temp_ds_weights = ds_weights + (ds_weights/total_ds*modifier)
    #Special case if nothing is downstream (for terminal branches)
    else:
        temp_ds_weights = ds_weights + modifier
    #Get the total old upstream weights
    total_us = np.sum(us_weights)
    #Reclaim the branch length (modifier) from all the upstream weights
    if total_us != 0:
        temp_us_weights = us_weights - (us_weights/total_us*modifier)
    #Special case for terminal branches
    else:
        print('This condition should perhaps never occur and should be investigated')
        temp_us_weights = us_weights - modifier
    #Put all the weights together
    all_weights = np.concatenate((temp_ds_weights, temp_us_weights))
    #In GSC weighting, the weights can't be less than the distance! Minor numerical rounding errors can
    #cause this to happen
    all_weights = np.minimum(all_weights, all_dists) 
    #Finally putting that boolean I've been passing around to use. Basically this is a re-scaling
    #of the GSC weights that I came up with that expresses each GSC weight for a given terminal
    #as a fraction of its total possible weight (its depth). In practice, it is a less dramatic 
    #weighting scheme than the non-normalized counterpart.
    if weights_type=='GSCn':
        all_weights = np.divide(all_weights, all_dists, out=np.zeros_like(all_weights), where=all_dists!=0)
    
    ###########################################################################
    #Calculate weighted variance and return
    ###########################################################################
    dsw = DescrStatsW(all_dists, all_weights)
    return dsw.var

#def optimize_root_loc_on_branch_MinVar_GSC(my_clade, depths_array, weights_array, ds_count, finished, weights_type):
#    """
#    See docs for "optimize_root_loc_on_branch_MinVar_general". For a given branch, this will take the 
#    depths and weights and optimize the exact location of the root for that particular branch. This function 
#    is SPECIFIC to GSC weighting schemes (both the initial and my normalized version).    
#    
#    Input/s:
#    my_clade - Bio.Phylo clade object
#    depths_array - 1D numpy array of all the depths for each terminal
#    weights_array - 2D numpy array of all the weights for each terminal (last column counts)
#    ds_count - number of downstream terminals emanating from this clade
#    finished - list of all completed terminals during the depth first search
#    weights_type - important for decideing whether or not to normalize weights at the end
#    
#    Output/s:
#    res - the function optima (scipy.optimize object)
#    
#    """
#    ###################################################
#    #Root-to-tip distances for all downstream terminals
#    ###################################################
#    downstream_dists = np.array(depths_array[len(finished):len(finished)+ds_count])
#    #And all upstream terminals
#    upstream_dists = np.concatenate((depths_array[:len(finished)],\
#                                     depths_array[len(finished)+ds_count:]))
#    
#    ###################################################
#    #Weights for all downstream terminals
#    ###################################################
#    downstream_weights = np.array(weights_array[len(finished):len(finished)+ds_count, -1])
#    #And all upstream terminals
#    upstream_weights = np.concatenate((weights_array[:len(finished), -1],\
#                                       weights_array[len(finished)+ds_count:, -1]))
#    #Also will need to know the old weights for upstream folks which should be the second to last column
#    old_upstream_weights = np.concatenate((weights_array[:len(finished), -2],\
#                                           weights_array[len(finished)+ds_count:, -2]))
#    ###################################################
#    #Set the bounds and optimize the GSC specific function
#    ###################################################
#    bl_bounds = np.array([[0., my_clade.branch_length]])
#    #Valid options for method are L-BFGS-B, SLSQP and TNC
#    res = minimize(branch_scan_MinVar_GSC, np.array(np.mean(bl_bounds)),\
#                          args=(downstream_dists, upstream_dists,\
#                                downstream_weights, upstream_weights, old_upstream_weights, weights_type),\
#                          bounds=bl_bounds, method='L-BFGS-B')
#    return res 
#
#
#    
#def branch_scan_MinVar_GSC(modifier, ds_dists, us_dists, ds_weights, us_weights, old_us_weights, weights_type):
#    """ 
#    See docs for "branch_scan_MinVar_general". This is the function to minimize in order to optimaly 
#    situate the root on the putative branch and is SPECIFIC to GSC weighting schemes. I should really    
#    try to make this a bit quicker/simpler if possible, there might a redundant array operation or two
#    but at present it works. 
#
#    Input/s:
#    modifier - This is the parameter to be optimized! Essentially a float of how much to shift the
#                root left or right so as to minimize the root-to-tip variance
#    ds_dists - array of downstream root-to-tip distances
#    us_dists - array of upstream root-to-tip distances
#    ds_weights - array of downstream terminal weights
#    us_weights - array of upstream terminal weights
#    old_us_weights - array of upstream terminal weights at the last step
#    weights_type - passed unchanged to determine whether or not to normalize
#
#    Output/s:
#    dsw.var - weighted variance
#    
#    """
#    ###########################################################################
#    #Adjust the downstream and upstream root-to-tip distances with the modifier
#    ###########################################################################
#    temp_ds_dists = ds_dists + modifier
#    temp_us_dists = us_dists - modifier
#    all_dists = np.concatenate((temp_ds_dists, temp_us_dists))
#    
#    ###########################################################################
#    #Now adjust the downstream and upstream weights
#    ###########################################################################
#    #First get the total downstream weights
#    total_ds = np.sum(ds_weights)
#    #Divide up the added branch length (modifier) across the downstream weights
#    if total_ds != 0:
#        temp_ds_weights = ds_weights + (ds_weights/total_ds*modifier)
#    #Special case if nothing is downstream (for terminal branches)
#    else:
#        temp_ds_weights = ds_weights + modifier
#    #Get the total old upstream weights
#    total_us = np.sum(old_us_weights)
#    #Reclaim the branch length (modifier) from all the upstream weights
#    if total_us != 0:
#        temp_us_weights = us_weights - (old_us_weights/total_us*modifier)
#    #Special case for terminal branches
#    else:
#        temp_us_weights = us_weights - modifier
#    #Put all the weights together
#    all_weights = np.concatenate((temp_ds_weights, temp_us_weights))
#    #In GSC weighting, the weights can't be less than the distance! Minor numerical rounding errors can
#    #cause this to happen
#    all_weights = np.minimum(all_weights, all_dists) 
#    #Finally putting that boolean I've been passing around to use. Basically this is a re-scaling
#    #of the GSC weights that I came up with that expresses each GSC weight for a given terminal
#    #as a fraction of its total possible weight (its depth). In practice, it is a less dramatic 
#    #weighting scheme than the non-normalized counterpart.
#    if weights_type=='GSCn':
#        all_weights = np.divide(all_weights, all_dists, out=np.zeros_like(all_weights), where=all_dists!=0)
#    
#    ###########################################################################
#    #Calculate weighted variance and return
#    ###########################################################################
#    dsw = DescrStatsW(all_dists, all_weights)
#    return dsw.var

def update_GSC_weights_dict(my_clade, parent_clade, weights_dict, finished):
    """    
    This allows for rapid re-calculation of GSC weights for all possible root positions on 
    a tree. Specifically, this function takes the weights for a parent clade and re-calcs
    for a daughter clade
    
    Input/s:
    my_clade - just a Bio.Phylo clade object
    parent_clade - the parent of the relevant clade
    weights_dict - the existing dictionary of clade(key):weights array(value) pairs
    finished - a list of the terminals that have been completed (used for rapidly accessing
                the downstream and upstream terminals)
                
    Output/s:
    weights_dict - the updated weights_dict object with a new key:val pair added to it
    
    """
    #Get number of downstream terminals
    ds_count = len(my_clade.get_terminals())
    #Copy matrix from parent
    new_array = np.array(weights_dict[parent_clade])
    #This is the total "weight" to reclaim from the downstream terms and distribute to the upstreams
    bl_to_disperse = my_clade.branch_length
    
    #Recover from downstream terminals
    current_ds_weights = np.sum(new_array[len(finished):len(finished)+ds_count])
    if current_ds_weights > 0:
        to_subtract = new_array[len(finished):len(finished)+ds_count]/current_ds_weights*-1*bl_to_disperse
    else:
        to_subtract = np.zeros_like(new_array[len(finished):len(finished)+ds_count])

    #Disperse to upstream terminals
    current_us_weights = np.sum(new_array[:len(finished)]) + np.sum(new_array[len(finished)+ds_count:]) 
    if current_us_weights > 0:
        to_add_a = new_array[:len(finished)] / current_us_weights * bl_to_disperse  
        to_add_b = new_array[len(finished)+ds_count:] / current_us_weights * bl_to_disperse
    else:
        to_add_a = np.zeros_like(new_array[:len(finished)])  
        to_add_b = np.zeros_like(new_array[len(finished)+ds_count:])
        
    assert  np.isclose(np.sum(to_add_a) + np.sum(to_add_b) + np.sum(to_subtract), 0.)
    #et voila
    new_array = new_array + np.concatenate((to_add_a, to_subtract, to_add_b))
    weights_dict[my_clade] = new_array   

#def update_GSC_weights_dict(my_clade, parent_clade, weights_dict, finished):
#    """    
#    This is pretty convoluted and could perhaps be simplified greatly with more thought. A lot of steps
#    but should be linear in O(t). The goal/purpose is to not have to re-calculate GSC weights for each 
#    possible root location. Rather, calculate these values once at the starting root node and then 
#    apply this function when recursively crawling the tree.
#    
#    Input/s:
#    my_clade - just a Bio.Phylo clade object
#    parent_clade - the parent of the relevant clade
#    weights_dict - the existing dictionary of clade(key):weights matrix(value) pairs
#    finished - a list of the terminals that have been completed (used for rapidly accessing
#                the downstream and upstream terminals)
#                
#    Output/s:
#    weights_dict - the updated weights_dict object with a new key:val pair added to it
#    
#    """
#    #Get number of downstream terminals
#    ds_count = len(my_clade.get_terminals())
#    #Copy matrix from parent
#    new_array = np.array(weights_dict[parent_clade])
#    #This is the total "weight" to reclaim from the downstream terms and distribute to the upstreams
#    bl_to_disperse = my_clade.branch_length
#    assert np.isclose(bl_to_disperse, np.sum(new_array[len(finished):len(finished)+ds_count, -1])-\
#                                    np.sum(new_array[len(finished):len(finished)+ds_count, -2]))
#    
#    #Get the total current weight of all upstream terms
#    to_divide = np.sum(new_array[:,-1]) - np.sum(new_array[len(finished):len(finished)+ds_count, -1])
#    #Array of values to add to the first and second set of upstream terms
#    to_add_a = new_array[:len(finished),-1]/to_divide * bl_to_disperse + new_array[:len(finished),-1]
#    to_add_b = new_array[len(finished)+ds_count:,-1]/to_divide * bl_to_disperse + new_array[len(finished)+ds_count:,-1]
#    
#    #Subtract the values from the downstream terms by rolling the values over
#    new_array[len(finished):len(finished)+ds_count] =\
#                np.roll(new_array[len(finished):len(finished)+ds_count], 1, axis=1)
#    #And setting the first column to be zeros
#    new_array[len(finished):len(finished)+ds_count, 0] = 0
#    #Finally, append now column of zeros
#    new_array = np.append(new_array, np.zeros([len(new_array),1]), axis=1)
#    #Roll the downstream terms again
#    new_array[len(finished):len(finished)+ds_count] =\
#                np.roll(new_array[len(finished):len(finished)+ds_count], 1, axis=1)
#    #Append the new vals for both upstream term setes
#    new_array[:len(finished),-1] = to_add_a
#    new_array[len(finished)+ds_count:,-1] = to_add_b
#    #et voila
#    weights_dict[my_clade] = new_array   

def no_updating(*args):
    pass

def update_depth_array_dict(my_clade, parent_clade, depths_dict, finished):
    """
    This function updates the depths of each terminal in a pretty straightforward manner.
    
    Input/s:
    my_clade - Bio.Phylo clade object
    parent_clade - parent of my_clade
    depths_dict - the exsting dictionary of depths where: clade(key):array of depths (value)
    finished - a list of all the terminals that have been completed in the depth first search
    
    Output/s:
    depths_dict - the updated depths_dict
    ds_count - the number of terminals downstream of this particular clade
    
    """
    #First grab who is downstream of this new clade
    ds_count = len(my_clade.get_terminals())
    #Instantiate new array with values from parent
    new_array = np.array(depths_dict[parent_clade])
    #Subtract the branch length from all the downstream clades
    new_array[len(finished):len(finished)+ds_count] -= my_clade.branch_length
    #Add the branch length to all the upstream clades (two sets)
    new_array[:len(finished)] += my_clade.branch_length
    new_array[len(finished)+ds_count:] += my_clade.branch_length
    #Update the dictionary
    depths_dict[my_clade] = new_array
    return depths_dict, ds_count

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
def mad_root_adhock(tree, normalize_weights=False):
    ###Let's question this assumption that zero bls screw things up and comment it for now
    for node in tree.get_terminals() + tree.get_nonterminals():
        if node == tree.root:
            continue
        if node.branch_length == 0.:
            node.branch_length = 10e-16
    ###Get the starting LCA matrix where each entry i,j is the length of i's 
    ###distance to it's last common ancestor with j
    
    #############NOTE: Big debate/parameter here! Toggle on/off the ..._normalized line to alter the weights
    #################  pretty substantially. No clue which is better/makes more ideological sense.
    weights_matrix, weights_matrix_normalized, weights_matrix_raw, terminal_list = pairwise_weighting.get_weight_matrices(tree)
    if normalize_weights:
        weights_matrix = weights_matrix_normalized
    #Toggle this on/off to ensure that weight vals of one equal the regular MAD implementation
    #weights_matrix = np.ones((len(tree.get_terminals()), len(tree.get_terminals())))
    lca_matrix, initial_order = get_lca_matrix(tree)
    tempy_dict = {}
    tempy_dict[tree.root] = lca_matrix
    ###Recursively compute MAD for all possible rootings of the tree
    explored, function_optima, lca_matrix_dict = recursive_crawl_mad(tree.root, None, [], [], tree, tempy_dict, weights_matrix)
    ###And using the optimal root position, re-root the tree
    function_optima = sorted(function_optima, key=lambda x: x[1][1])
    function_optima_success = (obj for obj in function_optima if np.isnan(obj[1][0]) == False)
    winner = next(function_optima_success)
    runner_up = next(function_optima_success)
    #Re-root at this point
    tree.root_with_outgroup(winner[0], outgroup_branch_length=0.)
    assert tree.root.clades[1].branch_length == 0.
    assert tree.root.clades[1] == winner[0]
    #And adjust the branch lengths around this root
    tree.root.clades[0].branch_length -= winner[1][0]
    tree.root.clades[1].branch_length += winner[1][0]
    if winner != function_optima[0]:
        print('May want to investigate function optima, appears to be unsuccessful search result at the top')
    RAI = winner[1][1] / runner_up[1][1]
    return tree, RAI, function_optima

def get_lca_matrix(tree):
    """
    The LCA matrix here is subtle. I'm actually calculating the distance to the last common ancestor
    for all pairs of terminal leaves for an initial hypothetical bifurcating root. This is done pretty
    straightforwardly by first getting the variance-covariance matrix and then subtracting each terminals
    depth from this matrix.

    """
    assert tree.is_bifurcating()
    initial_order = tree.get_terminals()
    initial_matrix = np.zeros((len(initial_order),len(initial_order)))
    #Call recursive function
    vcv_matrix, finished_list = recursive_vcv_matrix(tree.root, initial_matrix, finished=[])
    ###This makes the matrix asymmetrical and gives us what we ultimately want
    final_matrix = vcv_matrix - vcv_matrix.diagonal()
    return final_matrix, initial_order

def recursive_vcv_matrix(node, vcv_matrix, finished=[]):
    """
    This computes the variance-covariance matrix for a given root where each diagonal entry
    is the depth of that terminal to the root and each off diagonal entry is the amount of variance
    shared between terminals i and j (i.e. the distance of their last common ancestor to the root)
    """
    if node.branch_length:
        ###Keep track of the number of downstream terminal nodes and how many are finished
        n_terminals = len(node.get_terminals())
        n_finished = len(finished)
        ###Add the branch length of the node to all downstream terminals (and only the downstream terminals)
        vcv_matrix[n_finished:n_finished+n_terminals, n_finished:n_finished+n_terminals] += node.branch_length
    ###Recurse
    if len(node.clades) == 2:
        l_clade, r_clade = node.clades
        vcv_matrix, finished = recursive_vcv_matrix(l_clade, vcv_matrix, finished)
        vcv_matrix, finished = recursive_vcv_matrix(r_clade, vcv_matrix, finished)
    elif len(node.clades) == 0:
        finished.append(node)
    else:
        print("ERROR: APPEARS TO BE A NON-BINARY TREE. MATRIX GENERATION WILL PROBABLY FAIL")
    return vcv_matrix, finished


def recursive_crawl_mad(node, parent_node, explored, function_optima, tree, lca_matrix_dict, weights_matrix):
    """
    Another recursive function. This calculates the MAD value across the entire tree. It does so by constantly updating
    the lca_matrix so that this need not be re-computed and using these values to calculate the deviations between same side
    and different side nodes (side being relative to the proposed root node). 
    """
    if parent_node:###Don't bother calculating on the current root node since it is redundant
        ###Update the LCA matrix and track the number of downstream terminals from this node
        lca_matrix_dict, ds_count = update_lca_matrix_dict(lca_matrix_dict, node, parent_node, tree, explored)
        ###And optimize the MAD score using this lca matrix
        res = mad_from_matrix(node, ds_count, lca_matrix_dict[node], explored, weights_matrix)
        function_optima.append((node, res))
    ###Recurse
    if len(node.clades) == 2:
        l_clade, r_clade = node.clades        
        explored, function_optima, lca_matrix_dict = recursive_crawl_mad(l_clade, node, explored, function_optima, tree, lca_matrix_dict, weights_matrix)
        explored, function_optima, lca_matrix_dict = recursive_crawl_mad(r_clade, node, explored, function_optima, tree, lca_matrix_dict, weights_matrix)
    elif len(node.clades) == 0: ###base case
        explored.append(node)
        return explored, function_optima, lca_matrix_dict
    else:
        print('non binary tree...?')
    return explored, function_optima, lca_matrix_dict 

def update_lca_matrix_dict(lca_matrix_dict, my_clade, parent, my_tree, finished):
    """
    This does some heavy lifting in the calculations later
    """
    ###Get the branch length in question
    bl = my_clade.branch_length
    ###And the number of terminals downstream of this node
    ds_count = len(my_clade.get_terminals())
    ###Copy over the matrix of this node's parent
    new_matrix = np.array(lca_matrix_dict[parent])
    ###Since I moved down a node, subtract the current branch length
    ###from all downstream nodes
    new_matrix[len(finished):len(finished)+ds_count, :] -= bl
    ###And add the branch length to the upstream nodes (note that the values will 
    ###cancel out for the square of downstream terminals)
    new_matrix[:, len(finished):len(finished)+ds_count] += bl
    ###Add to the dictionary
    lca_matrix_dict[my_clade] = new_matrix
    return lca_matrix_dict, ds_count

def mad_from_matrix(my_clade, ds_count, lca_matrix, finished, weights_matrix):
    """
    And this pretty much does the whole MAD calculation given the up-to-date lca_matrix.

    In theory it's just measuring the deviations from the upper right triangle and the lower left.
    I feel like that fact could be leveraged to perform these calculations in a far less complicated way
    """
    #my_matrix = lca_matrix[len(finished):len(finished)+ds_count, len(finished):len(finished)+ds_count]
    #other_matrix = np.delete(lca_matrix, np.s_[len(finished):len(finished)+ds_count], 0)
    #other_matrix = np.delete(other_matrix, np.s_[len(finished):len(finished)+ds_count],1)
    #my_matrix_trans = my_matrix.T
    #other_matrix_trans = other_matrix.T
    #ss_a_dists = np.abs(np.concatenate((my_matrix[np.triu_indices(ds_count, k=1)],\
    #        other_matrix[np.triu_indices(other_matrix.shape[0], k=1)])))
    #ss_b_dists = np.abs(np.concatenate((my_matrix_trans[np.triu_indices(ds_count, k=1)],\
    #        other_matrix_trans[np.triu_indices(other_matrix_trans.shape[0], k=1)])))
    #ss_total_dists = ss_a_dists + ss_b_dists
    #ss_devs = np.abs(((2*ss_a_dists)/ss_total_dists)-1) 
    #
    #ds_a_dists = lca_matrix[len(finished):len(finished)+ds_count,np.r_[:len(finished),len(finished)+ds_count:lca_matrix.shape[0]]].flatten(order='C')
    #ds_b_dists = lca_matrix[np.r_[:len(finished),len(finished)+ds_count:lca_matrix.shape[0]], len(finished):len(finished)+ds_count].flatten(order='F')
    #ds_total_dists = ds_a_dists + ds_b_dists

    ####Using the analytical solution to "rho" parameter as outlined in the MAD paper
    #total_bl = my_clade.branch_length
    #if total_bl > 0.:
    #    rho = np.sum((ds_total_dists-(2*ds_a_dists))*ds_total_dists**-2)/(2*total_bl*np.sum(ds_total_dists**-2))
    #    modifier = total_bl*rho
    #    modifier = min(max(0, modifier), total_bl) 
    #else:
    #    modifier = 0.
 
    ####Rescale the distances with the optimized modifier
    #ds_a_dists = ds_a_dists + modifier
    #ds_b_dists = ds_b_dists - modifier
    #ds_total_dists = ds_a_dists + ds_b_dists
    ####Calculate their deviations
    #ds_devs = np.abs(((2*ds_a_dists)/ds_total_dists)-1)
    ####Concatenate them with the pre-computed same side deviations (ss_devs)
    #all_devs = np.concatenate((ss_devs, ds_devs))
    ####And compute final MAD score
    #all_devs = all_devs**2
    #dev_score = np.mean(all_devs)
    #dev_score = dev_score**0.5

    
    #An alternative that looks cleaner, but runs slower. Ho hum. Keeping it here incase it comes in handy for weighting
    #n_terms = lca_matrix.shape[0]
    #ds_a_dists = lca_matrix[len(finished):len(finished)+ds_count,np.r_[:len(finished),len(finished)+ds_count:n_terms]].flatten(order='C')
    #ds_b_dists = lca_matrix[np.r_[:len(finished),len(finished)+ds_count:n_terms], len(finished):len(finished)+ds_count].flatten(order='F')
    #ds_total_dists = ds_a_dists + ds_b_dists
    ####Using the analytical solution to "rho" parameter as outlined in the MAD paper
    #total_bl = my_clade.branch_length
    #if total_bl > 0.:
    #    rho = np.sum((ds_total_dists-(2*ds_a_dists))*ds_total_dists**-2)/(2*total_bl*np.sum(ds_total_dists**-2))
    #    modifier = total_bl*rho
    #    modifier = min(max(0, modifier), total_bl) 
    #else:
    #    modifier = 0.
    #
    #
    #
    #
    #new_matrix[len(finished):len(finished)+ds_count, np.r_[:len(finished), len(finished)+ds_count:n_terms]] += modifier
    #new_matrix[np.r_[:len(finished), len(finished)+ds_count:n_terms], len(finished):len(finished)+ds_count] -= modifier
    #all_mat = np.abs(new_matrix + new_matrix.T)[np.triu_indices(n_terms, k=1)]
    #one_side = np.abs(new_matrix)[np.triu_indices(n_terms, k=1)]
    #devs = np.abs(((2*one_side)/all_mat)-1)
    #dev_score = np.mean(devs**2)**0.5
    
    n_finished = len(finished)
    n_terms = lca_matrix.shape[0]
    bl_bounds = np.array([[0, my_clade.branch_length]]) 
    res = minimize(optimize_fxn_mad, np.array(0.),\
                           args=(lca_matrix, weights_matrix, n_finished, n_terms, ds_count),\
                           bounds=bl_bounds, method='L-BFGS-B')
    modifier = res.x[0]
    dev_score = res.fun
    return (modifier, dev_score)

def optimize_fxn_mad(modifier, lca_matrix, weights_matrix, n_finished, n_terms, ds_count):
    new_matrix = np.array(lca_matrix)
    new_matrix[n_finished:n_finished+ds_count, np.r_[:n_finished, n_finished+ds_count:n_terms]] += modifier
    new_matrix[np.r_[:n_finished, n_finished+ds_count:n_terms], n_finished:n_finished+ds_count] -= modifier
    all_mat = np.abs(new_matrix + new_matrix.T)[np.triu_indices(n_terms, k=1)]
    one_side = np.abs(new_matrix)[np.triu_indices(n_terms, k=1)]
    devs = np.abs(((2*one_side)/all_mat)-1)
    dev_score = np.average(devs**2, weights=weights_matrix[np.triu_indices(n_terms, k=1)])**0.5
    return dev_score
