import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats


###########################################################################################
#Mid-point rooting, orders of magnitude faster than the current biopython implementation###
###########################################################################################
def mp_root_adhock(tree):
    '''
    The only really strange behavior that I'm aware of right now are the obvious "ties" in terms of 
    the farthest away branches. I'm not sure if it will eventually be best to try them all and make sure
    that there are differences in the chosen root node / location. But for now I'm just choosing
    one at random, where random is whoever pops up first in the list (so maybe not fully random).


    In any event, this is so much faster than the biopython implementation.
    '''

    initial_bl = tree.total_branch_length()
    initial_term_names = [i.name for i in tree.get_terminals()]
    ###Root randomly with an outgroup at a terminal
    tree.root_with_outgroup(tree.get_terminals()[0], outgroup_branch_length=0.0)
    
    ###Through some type of bug in Bio.Phylo, new terminals can pop into existence and
    ###this is a hack to remove them. I think they come from non-zero root branch lengths
    ###based on visual inspection of some of the trees that cause the issue
    pruned_bls = []
    for terminal in tree.get_terminals():
        if terminal.name not in initial_term_names:
            pruned_bls.append(terminal.branch_length)
            tree.prune(terminal)
            print('Pruned a strange terminal that popped into existence during midpoint rooting')
    
    ###I'm not entirely sure how this algorithm works on non-bifurcating trees. Thus, the initial
    ###assertion statement. It might be fine, but I'd have to think about it.
    assert tree.is_bifurcating()
    
    ###Find which terminal/s is farthest away from my randomly selected root terminal
    depths = tree.depths()
    max_depth = max(list(depths.values()))
    clade1 = [i for i,j in depths.items() if j == max_depth]
    ###Idealy this would be a list of length 1. If not, there are multiple terminals
    ###that are maximally distant (could be due to polytomies or I suppose random chance)
    if len(clade1) > 1:
        print('Potential for multiple midpoints. Choosing one at random')

    ###Re-root at that farthest terminal (NOTE: just choosing the first clade in the list
    ###that may be longer than length 1)
    tree.root_with_outgroup(clade1[0], outgroup_branch_length=0.0)
    
    ###And finally find which terminal is farthest from that one to identify the farthest pair
    depths = tree.depths()
    max_depth = max(list(depths.values()))
    clade2 = [i for i,j in depths.items() if j == max_depth]
    ###Same constraint/caveat applies as above with regard to "ties"
    if len(clade2) > 1:
        print('Potential for multiple midpoints. Choosing one at random')
    
    ###Given the clade pairs, re-root the tree
    rooted_tree = final_root(clade1[0], clade2[0], depths[clade2[0]], tree)
    
    ###Ensuring that I've fully conserved branch length after all these manipulations
    ###because I've had problems with gaining / losing owing to what I think are issues
    ###in Bio.Phylo that I think I've fully figured out.
    assert np.isclose(initial_bl-np.sum(pruned_bls), rooted_tree.total_branch_length())
    
    return rooted_tree

def final_root(clade1, clade2, max_distance, tree):
    '''
    Given the target clades, this actually re-roots the tree.

    Support function for `mp_root_adhock`.
    '''
    # Depth to go from the ingroup tip toward the outgroup tip 
    root_remainder = 0.5 * (max_distance - (tree.root.branch_length or 0)) 
    assert root_remainder >= 0 
    for node in tree.get_path(clade2): 
        root_remainder -= node.branch_length 
        if root_remainder < 0: 
            outgroup_node = node 
            outgroup_branch_length = -root_remainder
            break 
    else: 
        raise ValueError("Somehow, failed to find the midpoint!")
    ###Specifying the outgroup_branch_length directly with this flag lead to some
    ###error-prone behavior so I'm doing it in two steps. Must be a bug and/or mis-understanding
    ###in Bio.Phylo
    tree.root_with_outgroup(outgroup_node, outgroup_branch_length=0.0)
    assert outgroup_node == tree.root.clades[1]
    tree.root.clades[0].branch_length = tree.root.clades[0].branch_length + root_remainder
    tree.root.clades[1].branch_length = tree.root.clades[1].branch_length - root_remainder
    return tree


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
def MinVar_root_adhock(tree):
    ''' 
    This implements a rooting scheme to maximize the likelihood of the root-to-tip distances coming from a 
    normal distribution (which I believe is also equivalent to minimizing the standard deviation). 
    This code could be easily modified to accomodate other distributions, of course, and some
    of these should probably be tested / ideologically considered. 

    Additional thought to consider is whether to include an outlier detection so that single branches do
    not skew results too strongly.

    Finally, in scratch phase is an algorithm to consider weighted averages.
    '''
    initial_depths = tree.root.depths()
    depths_dict = {}
    depths_dict[tree.root] = np.array([initial_depths[i] for i in tree.get_terminals()])
    function_optima, depths_dict, finished = recursive_crawl_MinVar(tree.root, None, [], depths_dict, [])
    function_optima = sorted(function_optima, key=lambda x: x[1].fun)
    #Re-root at this point 
    winner = next(obj for obj in function_optima if obj[1].success==True)
    tree.root_with_outgroup(winner[0], outgroup_branch_length=0.)
    assert tree.root.clades[1].branch_length == 0.
    assert tree.root.clades[1] == winner[0]
    #And adjust the branch lengths around this root
    tree.root.clades[0].branch_length -= winner[1].x[0]
    tree.root.clades[1].branch_length += winner[1].x[0]
    return tree, function_optima, depths_dict

def recursive_crawl_MinVar(node, parent_node, function_optima, depths_dict, finished):
    if parent_node:
        depths_dict, ds_count = update_depth_array_dict(node, parent_node, depths_dict, finished)
        res = optimize_root_loc_on_branch_MinVar(node, depths_dict[node], ds_count, finished)
        function_optima.append((node, res))
    ###Recurse
    if len(node.clades) == 2:
        l_clade, r_clade = node.clades
        function_optima, depths_dict, finished = recursive_crawl_MinVar(l_clade, node, function_optima, depths_dict, finished)
        function_optima, depths_dict, finished = recursive_crawl_MinVar(r_clade, node, function_optima, depths_dict, finished)
    elif len(node.clades) == 0:
        finished.append(node)
        return function_optima, depths_dict, finished
    else:
        print('Some big error here with the number of clades stemming from this root')
    return function_optima, depths_dict, finished

def update_depth_array_dict(my_clade, parent_clade, depths_dict, finished):
    '''
    This function updates the depths of each terminal in a pretty straightforward manner.
    Using pandas here rather than numpy because speed doesn't seem to be much of a factor 
    but it could definitely be sped up by ridding our selves from the dataframe calls
    '''
    ###First grab who is downstream of this new clade
    ds_count = len(my_clade.get_terminals())
    new_array = np.array(depths_dict[parent_clade])
    new_array[len(finished):len(finished)+ds_count] -= my_clade.branch_length
    new_array[:len(finished)] += my_clade.branch_length
    new_array[len(finished)+ds_count:] += my_clade.branch_length
    depths_dict[my_clade] = new_array
    return depths_dict, ds_count 

def optimize_root_loc_on_branch_MinVar(my_clade, depths_array, ds_count, finished):
    ''' 
    While the update_depth_df_dict function gets us started on this new terminal, this function
    will optimize the location on that terminal
    '''
    downstream_dists = np.array(depths_array[len(finished):len(finished)+ds_count])
    upstream_dists = np.concatenate((depths_array[:len(finished)], depths_array[len(finished)+ds_count:]))
    bl_bounds = np.array([[0., my_clade.branch_length]])
    ###Optimize!
    ###Valid options for method are L-BFGS-B, SLSQP and TNC
    res = minimize(branch_scan_MinVar, np.array(np.mean(bl_bounds)),\
                          args=(downstream_dists, upstream_dists),\
                          bounds=bl_bounds, method='L-BFGS-B') 
    return res

def branch_scan_MinVar(modifier, ds_dists, us_dists):
    temp_ds_dists = ds_dists + modifier
    temp_us_dists = us_dists - modifier
    all_dists = np.concatenate((temp_ds_dists, temp_us_dists))
    mean, std = stats.norm.fit(all_dists)
    return np.std(all_dists)#Minimize the Standard Deviation
    #return np.std(all_dists)/np.abs(np.mean(all_dists))#Minimize the coefficient of variation
    #return -np.sum(stats.norm.logpdf(all_dists, loc=mean, scale=std))#Maximize the likelihood



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
    lca_matrix, initial_order = get_lca_matrix(tree)
    tempy_dict = {}
    tempy_dict[tree.root] = lca_matrix
    ###Recursively compute MAD for all possible rootings of the tree
    explored, function_optima, lca_matrix_dict = recursive_crawl_mad(tree.root, None, [], [], tree, tempy_dict)
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


def recursive_crawl_mad(node, parent_node, explored, function_optima, tree, lca_matrix_dict):
    '''
    Another recursive function. This calculates the MAD value across the entire tree. It does so by constantly updating
    the lca_matrix so that this need not be re-computed and using these values to calculate the deviations between same side
    and different side nodes (side being relative to the proposed root node). 
    '''
    if parent_node:###Don't bother calculating on the current root node since it is redundant
        ###Update the LCA matrix and track the number of downstream terminals from this node
        lca_matrix_dict, ds_count = update_lca_matrix_dict(lca_matrix_dict, node, parent_node, tree, explored)
        ###And optimize the MAD score using this lca matrix
        res = mad_from_matrix(node, ds_count, lca_matrix_dict[node], explored)
        function_optima.append((node, res))
    ###Recurse
    if len(node.clades) == 2:
        l_clade, r_clade = node.clades        
        explored, function_optima, lca_matrix_dict = recursive_crawl_mad(l_clade, node, explored, function_optima, tree, lca_matrix_dict)
        explored, function_optima, lca_matrix_dict = recursive_crawl_mad(r_clade, node, explored, function_optima, tree, lca_matrix_dict)
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

def mad_from_matrix(my_clade, ds_count, lca_matrix, finished):
    '''
    And this pretty much does the whole MAD calculation given the up-to-date lca_matrix.

    In theory it's just measuring the deviations from the upper right triangle and the lower left.
    I feel like that fact could be leveraged to perform these calculations in a far less complicated way
    '''
    my_matrix = lca_matrix[len(finished):len(finished)+ds_count, len(finished):len(finished)+ds_count]
    other_matrix = np.delete(lca_matrix, np.s_[len(finished):len(finished)+ds_count], 0)
    other_matrix = np.delete(other_matrix, np.s_[len(finished):len(finished)+ds_count],1)
    my_matrix_trans = my_matrix.T
    other_matrix_trans = other_matrix.T
    ss_a_dists = np.abs(np.concatenate((my_matrix[np.triu_indices(ds_count, k=1)],\
            other_matrix[np.triu_indices(other_matrix.shape[0], k=1)])))
    ss_b_dists = np.abs(np.concatenate((my_matrix_trans[np.triu_indices(ds_count, k=1)],\
            other_matrix_trans[np.triu_indices(other_matrix_trans.shape[0], k=1)])))
    ss_total_dists = ss_a_dists + ss_b_dists
    ss_devs = np.abs(((2*ss_a_dists)/ss_total_dists)-1) 
    
    ds_a_dists = lca_matrix[len(finished):len(finished)+ds_count,np.r_[:len(finished),len(finished)+ds_count:lca_matrix.shape[0]]].flatten(order='C')
    ds_b_dists = lca_matrix[np.r_[:len(finished),len(finished)+ds_count:lca_matrix.shape[0]], len(finished):len(finished)+ds_count].flatten(order='F')
    ds_total_dists = ds_a_dists + ds_b_dists

    ###Using the analytical solution to "rho" parameter as outlined in the MAD paper
    total_bl = my_clade.branch_length
    if total_bl > 0.:
        rho = np.sum((ds_total_dists-(2*ds_a_dists))*ds_total_dists**-2)/(2*total_bl*np.sum(ds_total_dists**-2))
        modifier = total_bl*rho
        modifier = min(max(0, modifier), total_bl) 
    else:
        modifier = 0.
 
    ###Rescale the distances with the optimized modifier
    ds_a_dists = ds_a_dists + modifier
    ds_b_dists = ds_b_dists - modifier
    ds_total_dists = ds_a_dists + ds_b_dists
    ###Calculate their deviations
    ds_devs = np.abs(((2*ds_a_dists)/ds_total_dists)-1)
    ###Concatenate them with the pre-computed same side deviations (ss_devs)
    all_devs = np.concatenate((ss_devs, ds_devs))
    ###And compute final MAD score
    all_devs = all_devs**2
    dev_score = np.mean(all_devs)
    dev_score = dev_score**0.5

    
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
    #new_matrix = np.array(lca_matrix)
    #new_matrix[len(finished):len(finished)+ds_count, np.r_[:len(finished), len(finished)+ds_count:n_terms]] += modifier
    #new_matrix[np.r_[:len(finished), len(finished)+ds_count:n_terms], len(finished):len(finished)+ds_count] -= modifier
    #all_mat = np.abs(new_matrix + new_matrix.T)[np.triu_indices(n_terms, k=1)]
    #one_side = np.abs(new_matrix)[np.triu_indices(n_terms, k=1)]
    #devs = np.abs(((2*one_side)/all_mat)-1)
    #new_dev_score = np.mean(devs**2)**0.5
    return (modifier, dev_score)

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
