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

from StringIO import StringIO

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
def update_GSC_weights_dict(my_clade, parent_clade, weights_dict, finished):
    """    
    This is pretty convoluted and could perhaps be simplified greatly with more thought. A lot of steps
    but should be linear in O(t). The goal/purpose is to not have to re-calculate GSC weights for each 
    possible root location. Rather, calculate these values once at the starting root node and then 
    apply this function when recursively crawling the tree.
    
    Input/s:
    my_clade - just a Bio.Phylo clade object
    parent_clade - the parent of the relevant clade
    weights_dict - the existing dictionary of clade(key):weights matrix(value) pairs
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
    bl_to_disperse = 1. 
    assert np.isclose(bl_to_disperse, np.sum(new_array[len(finished):len(finished)+ds_count, -1])-\
                                    np.sum(new_array[len(finished):len(finished)+ds_count, -2]))
    
    #Get the total current weight of all upstream terms
    to_divide = np.sum(new_array[:,-1]) - np.sum(new_array[len(finished):len(finished)+ds_count, -1])
    #Array of values to add to the first and second set of upstream terms
    to_add_a = new_array[:len(finished),-1]/to_divide * bl_to_disperse + new_array[:len(finished),-1]
    to_add_b = new_array[len(finished)+ds_count:,-1]/to_divide * bl_to_disperse + new_array[len(finished)+ds_count:,-1]
    
    #Subtract the values from the downstream terms by rolling the values over
    new_array[len(finished):len(finished)+ds_count] =\
                np.roll(new_array[len(finished):len(finished)+ds_count], 1, axis=1)
    #And setting the first column to be zeros
    new_array[len(finished):len(finished)+ds_count, 0] = 0
    #Finally, append now column of zeros
    new_array = np.append(new_array, np.zeros([len(new_array),1]), axis=1)
    #Roll the downstream terms again
    new_array[len(finished):len(finished)+ds_count] =\
                np.roll(new_array[len(finished):len(finished)+ds_count], 1, axis=1)
    #Append the new vals for both upstream term setes
    new_array[:len(finished),-1] = to_add_a
    new_array[len(finished)+ds_count:,-1] = to_add_b
    #et voila
    weights_dict[my_clade] = new_array   

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

def MinVar_root_adhock_general(tree, weights_type=None, **kwargs):
    ''' 
    This implements a rooting scheme to minimize the weighted variance of the root-to-tip
    distances for all leaves in the tree. It is unclear whether minimizing the variance of a non-normal
    distribution would be more appropriate but this could easily be accommodated within this framework.
    
    Specifically, the weighting scheme here is the Gerstein-Sonnhammer-Chothia (GSC) weights and is
    complicated by the fact that those weights are root-dependent such that for each putative root
    both the depths and the weights need to be updated. Additionally I introduce the concept of 
    normalized_GSC weights which divide each weight by its depth. This is currently a flag that can 
    be toggled but depending on development may be either removed or set as a default.

    Input/s:
    tree - a Bio.Phylo tree object (in practice I like to root the tree with my basic mid-point 
                algorithm first just to ensure that the tree structure is normal-ish, i.e. bifurcating)
    normalize_GSC - boolean of whether or not to re-scale the GSC weights in the final step
                
    Output/s:
    tree - the now rooted tree object
    function_optima - a list of all the function optimization output for each putative root tested
    depths_dict - a dictionary of the depths for each terminal for each putative root where: 
                clade(key): array of weights for all terminals (value)
    weights_dict - a dictionary of the weights for each terminal for each putative root where:
                clade(key): matrix of weights for all terminals (value), final column are 
                the relevant weights but the entire matrix is necessary for rapid updating

    '''
    unit_tree = Phylo.read(StringIO(tree.format('newick')), 'newick')
    for node in unit_tree.get_terminals() + unit_tree.get_nonterminals():
        if node.branch_length:
            node.branch_length = 1.

    initial_depths = tree.root.depths()
    #Instantiate a depths dictionary with the root node
    depths_dict = {}
    depths_dict[tree.root] = np.array([initial_depths[i] for i in tree.get_terminals()])
    ###
    if weights_type in ['GSC', 'GSCn']:
        initial_weights = weighting_methods.GSC_adhock(unit_tree)
        #Instantiate a weights dictionary with the root node
        weights_array_dict = {}
        weights_array_dict[tree.root] = np.array([initial_weights[i] for i in unit_tree.get_terminals()])
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
        print('Problem interpreting your weights string. Returning')
        return
    #Do a lot of work! Recursively crawl the tree visiting each possible root branch
    function_optima, depths_dict, weights_dict, finished =\
            recursive_crawl_MinVar_general(tree.root, None, [], depths_dict, weights_array_dict, [], weights_update_fxn, weights_type) 
    #Perform the rooting with the results
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
    This is a meaty recursive function that performs the depth first search / tree crawling.
    
    Input/s:
    my_clade - Bio.Phylo clade object
    parent_clade - the parent of my_clade
    function_optima - list of all the function_optima that I'm calculating
    depths_dict - described in detail elsewhere
    weights_dict - described in detail elsewhere
    finished - list of all terminals that the depth first search has completed
    normalize_GSC - boolean of whether or not to re-scale the GSC values

    Output/s (updated versions of each of):
    function_optima
    depths_dict
    weights_dict
    finished
    
    """
    #If the node has a parent, do some calculations (this is here just to skip calculations on the root
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
    For a given branch, ths will take the depths and weights and optimize the exact location
    of the root for that particular branch.
    
    Input/s:
    my_clade - Bio.Phylo clade object
    depths_array - 1D numpy array of all the depths for each terminal
    weights_array - 1D numpy array of all the weights for each terminal
    ds_count - number of downstream terminals emanating from this clade
    finished - list of all completed terminals during the depth first search
    
    Output/s:
    res - the function optima (scipy.optimize object)
    
    """
    #Root-to-tip distances for all downstream terminals
    downstream_dists = np.array(depths_array[len(finished):len(finished)+ds_count])
    #And all upstream terminals
    upstream_dists = np.concatenate((depths_array[:len(finished)],\
                                     depths_array[len(finished)+ds_count:]))
    
    #Weights for all downstream terminals
    downstream_weights = np.array(weights_array[len(finished):len(finished)+ds_count])
    #And all upstream terminals
    upstream_weights = np.concatenate((weights_array[:len(finished)],\
                                       weights_array[len(finished)+ds_count:]))
    all_weights = np.concatenate((downstream_weights, upstream_weights))
    #Set the bounds for the optimization
    bl_bounds = np.array([[0., my_clade.branch_length]])
    #Valid options for method are L-BFGS-B, SLSQP and TNC
    res = minimize(branch_scan_MinVar_general, np.array(np.mean(bl_bounds)),\
                          args=(downstream_dists, upstream_dists,\
                                all_weights),\
                          bounds=bl_bounds, method='L-BFGS-B')
    return res  


    
def branch_scan_MinVar_general(modifier, ds_dists, us_dists, all_weights):
    """
    Should really try to make this a bit quicker/simpler if possible. All array options but a bit too
    many I suspect.
    
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
    For a given branch, ths will take the depths and weights and optimize the exact location
    of the root for that particular branch.
    
    Input/s:
    my_clade - Bio.Phylo clade object
    depths_array - 1D numpy array of all the depths for each terminal
    weights_array - 2D numpy array of all the weights for each terminal (last column counts)
    ds_count - number of downstream terminals emanating from this clade
    finished - list of all completed terminals during the depth first search
    
    Output/s:
    res - the function optima (scipy.optimize object)
    
    """
    #Root-to-tip distances for all downstream terminals
    downstream_dists = np.array(depths_array[len(finished):len(finished)+ds_count])
    #And all upstream terminals
    upstream_dists = np.concatenate((depths_array[:len(finished)],\
                                     depths_array[len(finished)+ds_count:]))
    
    #Weights for all downstream terminals
    downstream_weights = np.array(weights_array[len(finished):len(finished)+ds_count, -1])
    #And all upstream terminals
    upstream_weights = np.concatenate((weights_array[:len(finished), -1],\
                                       weights_array[len(finished)+ds_count:, -1]))
    #Also will need to know the old weights for upstream folks which should be the second to last column
    old_upstream_weights = np.concatenate((weights_array[:len(finished), -2],\
                                           weights_array[len(finished)+ds_count:, -2]))
    #Set the bounds for the optimization
    bl_bounds = np.array([[0., my_clade.branch_length]])
    #Valid options for method are L-BFGS-B, SLSQP and TNC
    res = minimize(branch_scan_MinVar_GSC, np.array(np.mean(bl_bounds)),\
                          args=(downstream_dists, upstream_dists,\
                                downstream_weights, upstream_weights, old_upstream_weights, weights_type),\
                          bounds=bl_bounds, method='SLSQP')
    return res 


    
def branch_scan_MinVar_GSC(modifier, ds_dists, us_dists, ds_weights, us_weights, old_us_weights, weights_type):
    """ 
    Should really try to make this a bit quicker/simpler if possible. All array options but a bit too
    many I suspect.
    
    Input/s:
    modifier - This is the parameter to be optimized! Essentially a float of how much to shift the
                root left or right so as to minimize the root-to-tip variance
    ds_dists - array of downstream root-to-tip distances
    us_dists - array of upstream root-to-tip distances
    ds_weights - array of downstream terminal weights
    us_weights - array of upstream terminal weights
    old_us_weights - array of upstream terminal weights at the last step
    
    Output/s:
    dsw.var - weighted variance
    
    """
    #Adjust the downstream and upstream root-to-tip distances with the modifier
    temp_ds_dists = ds_dists + modifier
    temp_us_dists = us_dists - modifier
    all_dists = np.concatenate((temp_ds_dists, temp_us_dists))
    #Get the total down stream weights
    total_ds = np.sum(ds_weights)
    #Divide up the added branch length (modifier) across the downstream weights
    if total_ds != 0:
        temp_ds_weights = ds_weights + (ds_weights/total_ds*modifier)
    #Special case if nothing is downstream (for terminal branches)
    else:
        temp_ds_weights = ds_weights + modifier
    #Get the total old upstream weights
    total_us = np.sum(old_us_weights)
    #Reclaim the branch length (modifier) from all the upstream weights
    if total_us != 0:
        temp_us_weights = us_weights - (old_us_weights/total_us*modifier)
    #Special case for terminal branches
    else:
        temp_us_weights = us_weights - modifier
    #Put all the weights together
    all_weights = np.concatenate((temp_ds_weights, temp_us_weights))
    #Finally putting that boolean I've been passing around to use. Basically this is a re-scaling
    #of the GSC weights that I came up with that expresses each GSC weight for a given terminal
    #as a fraction of its total possible weight (its depth). In practice, it is a less dramatic 
    #weighting scheme than the non-normalized counterpart.
    if weights_type=='GSCn':
        all_weights = np.divide(all_weights, all_dists, out=np.zeros_like(all_weights), where=all_dists!=0)
        #all_weights = all_weights/all_dists
        #all_weights = np.nan_to_num(all_weights)
    #Calculate weighted variance and return
    dsw = DescrStatsW(all_dists, all_weights)
    return dsw.var




###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
def mad_root_adhock(tree):
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
    weights_matrix_raw, weights_matrix, weights_matrix_normalized = pairwise_weighting.get_weight_matrices(tree)
    #weights_matrix = weights_matrix_normalized
    #weights_matrix = 1./weights_matrix_normalized
    #weights_matrix = 1./weights_matrix
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
    '''
    The LCA matrix here is subtle. I'm actually calculating the distance to the last common ancestor
    for all pairs of terminal leaves for an initial hypothetical bifurcating root. This is done pretty
    straightforwardly by first getting the variance-covariance matrix and then subtracting each terminals
    depth from this matrix.

    '''
    assert tree.is_bifurcating()
    initial_order = tree.get_terminals()
    initial_matrix = np.zeros((len(initial_order),len(initial_order)))
    #Call recursive function
    vcv_matrix, finished_list = recursive_vcv_matrix(tree.root, initial_matrix, finished=[])
    ###This makes the matrix asymmetrical and gives us what we ultimately want
    final_matrix = vcv_matrix - vcv_matrix.diagonal()
    return final_matrix, initial_order

def recursive_vcv_matrix(node, vcv_matrix, finished=[]):
    '''
    This computes the variance-covariance matrix for a given root where each diagonal entry
    is the depth of that terminal to the root and each off diagonal entry is the amount of variance
    shared between terminals i and j (i.e. the distance of their last common ancestor to the root)
    '''
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
    '''
    Another recursive function. This calculates the MAD value across the entire tree. It does so by constantly updating
    the lca_matrix so that this need not be re-computed and using these values to calculate the deviations between same side
    and different side nodes (side being relative to the proposed root node). 
    '''
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
    '''
    This does some heavy lifting in the calculations later
    '''
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
    '''
    And this pretty much does the whole MAD calculation given the up-to-date lca_matrix.

    In theory it's just measuring the deviations from the upper right triangle and the lower left.
    I feel like that fact could be leveraged to perform these calculations in a far less complicated way
    '''
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


#def mad_from_matrix(my_clade, ds_count, lca_matrix, finished):
#    '''
#    And this pretty much does the whole MAD calculation given the up-to-date lca_matrix.
#
#    In theory it's just measuring the deviations from the upper right triangle and the lower left.
#    I feel like that fact could be leveraged to perform these calculations in a far less complicated way
#    '''
#    my_matrix = lca_matrix[len(finished):len(finished)+ds_count, len(finished):len(finished)+ds_count]
#    other_matrix = np.delete(lca_matrix, np.s_[len(finished):len(finished)+ds_count], 0)
#    other_matrix = np.delete(other_matrix, np.s_[len(finished):len(finished)+ds_count],1)
#    my_matrix_trans = my_matrix.T
#    other_matrix_trans = other_matrix.T
#    ss_a_dists = np.abs(np.concatenate((my_matrix[np.triu_indices(ds_count, k=1)],\
#            other_matrix[np.triu_indices(other_matrix.shape[0], k=1)])))
#    ss_b_dists = np.abs(np.concatenate((my_matrix_trans[np.triu_indices(ds_count, k=1)],\
#            other_matrix_trans[np.triu_indices(other_matrix_trans.shape[0], k=1)])))
#    ss_total_dists = ss_a_dists + ss_b_dists
#    ss_devs = np.abs(((2*ss_a_dists)/ss_total_dists)-1) 
#    
#    ds_a_dists = lca_matrix[len(finished):len(finished)+ds_count,np.r_[:len(finished),len(finished)+ds_count:lca_matrix.shape[0]]].flatten(order='C')
#    ds_b_dists = lca_matrix[np.r_[:len(finished),len(finished)+ds_count:lca_matrix.shape[0]], len(finished):len(finished)+ds_count].flatten(order='F')
#    ds_total_dists = ds_a_dists + ds_b_dists
#
#    ###Using the analytical solution to "rho" parameter as outlined in the MAD paper
#    total_bl = my_clade.branch_length
#    if total_bl > 0.:
#        rho = np.sum((ds_total_dists-(2*ds_a_dists))*ds_total_dists**-2)/(2*total_bl*np.sum(ds_total_dists**-2))
#        modifier = total_bl*rho
#        modifier = min(max(0, modifier), total_bl) 
#    else:
#        modifier = 0.
# 
#    ###Rescale the distances with the optimized modifier
#    ds_a_dists = ds_a_dists + modifier
#    ds_b_dists = ds_b_dists - modifier
#    ds_total_dists = ds_a_dists + ds_b_dists
#    ###Calculate their deviations
#    ds_devs = np.abs(((2*ds_a_dists)/ds_total_dists)-1)
#    ###Concatenate them with the pre-computed same side deviations (ss_devs)
#    all_devs = np.concatenate((ss_devs, ds_devs))
#    ###And compute final MAD score
#    all_devs = all_devs**2
#    dev_score = np.mean(all_devs)
#    dev_score = dev_score**0.5
#    return (modifier, dev_score)
#
#
