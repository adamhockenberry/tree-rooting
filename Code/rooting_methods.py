import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

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
    ###error-prone behavior so I'm doing it in two steps. Must be a bug in Bio.Phylo
    tree.root_with_outgroup(outgroup_node, outgroup_branch_length=0.0)
    assert outgroup_node == tree.root.clades[1]
    tree.root.clades[0].branch_length = tree.root.clades[0].branch_length + root_remainder
    tree.root.clades[1].branch_length = tree.root.clades[1].branch_length - root_remainder
    return tree

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



###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
def branch_scan_ml(modifier, ds_dists, us_dists):
    temp_ds_dists = ds_dists + modifier
    temp_us_dists = us_dists - modifier
    all_dists = np.concatenate((temp_ds_dists, temp_us_dists))
    mean, std = stats.norm.fit(all_dists)
    #return np.std(all_dists)
    #return np.std(all_dists)/np.abs(np.mean(all_dists))
    return -np.sum(stats.norm.logpdf(all_dists, loc=mean, scale=std))

def update_depth_df_dict(my_clade, parent_clade, depths_dict):
    downstream_terms = [i.name for i in my_clade.get_terminals()]
    upstream_terms = list(set(list(depths_dict[parent_clade].index)) - set(downstream_terms))
    depths_dict[my_clade] = depths_dict[parent_clade].copy(deep=True)
    depths_dict[my_clade].loc[downstream_terms, 'depth'] -= my_clade.branch_length
    depths_dict[my_clade].loc[upstream_terms, 'depth'] += my_clade.branch_length
    return depths_dict, downstream_terms, upstream_terms

def recursive_crawl_ml(hypothetical_root, explored, function_optima, depths_dict, tree):
    if len(hypothetical_root.clades) == 2:
        l_clade, r_clade = hypothetical_root.clades 
        l_bl = l_clade.branch_length
        r_bl = r_clade.branch_length
        #L clade first
        if l_bl > 0:
            depths_dict, downstream_terms, upstream_terms = update_depth_df_dict(l_clade, hypothetical_root, depths_dict)
            res = optimize_root_loc_on_branch(l_clade, depths_dict[l_clade], downstream_terms, upstream_terms)
            function_optima.append((l_clade, res))
            explored, function_optima, depths_dict = recursive_crawl_ml(l_clade, explored, function_optima, depths_dict, tree)
        #R clade second
        if r_bl > 0:
            depths_dict, downstream_terms, upstream_terms = update_depth_df_dict(r_clade, hypothetical_root, depths_dict)
            res = optimize_root_loc_on_branch(r_clade, depths_dict[r_clade], downstream_terms, upstream_terms)
            function_optima.append((r_clade, res))
            explored, function_optima, depths_dict = recursive_crawl_ml(r_clade, explored, function_optima, depths_dict, tree)
        
    elif len(hypothetical_root.clades) == 1:
        l_clade = hypothetical_root.clades[0]
        l_bl = l_clade.branch_length
        if l_bl > 0:
            depths_dict, downstream_terms, upstream_terms = update_depth_df_dict(l_clade, hypothetical_root, depths_dict)
            res = optimize_root_loc_on_branch(l_clade, depths_dict[l_clade], downstream_terms, upstream_terms)
            function_optima.append((l_clade, res))
            explored, function_optima, depths_dict = recursive_crawl_ml(l_clade, explored, function_optima, depths_dict, tree)
    
    elif len(hypothetical_root.clades) == 0:
        explored.append(hypothetical_root)
        return explored, function_optima, depths_dict
    
    else:
        print('Some big error here with the number of clades stemming from this root')
    explored.append(hypothetical_root)
    return explored, function_optima, depths_dict

def optimize_root_loc_on_branch(my_clade, depths_df, downstream_terms, upstream_terms):
    '''
    '''
    downstream_dists = np.array(depths_df.loc[downstream_terms, 'depth'])
    upstream_dists = np.array(depths_df.loc[upstream_terms, 'depth'])
    bl_bounds = np.array([[0., my_clade.branch_length]])
    ###Valid options for method are L-BFGS-B, SLSQP and TNC
    res = minimize(branch_scan_ml, np.array(np.mean(bl_bounds)),\
                          args=(downstream_dists, upstream_dists),\
                          bounds=bl_bounds, method='L-BFGS-B') 
    return res

def ml_root_adhock(tree):
    '''
    This implements a rooting scheme to maximize the likelihood of the root-to-tip distances coming from a 
    normal distribution. Could be easily modified to accomodate other distributions, of course, and some
    of these should probably be tested / ideologically considered. 

    Additional thought to consider is whether to include an outlier detection so that single branches do
    not skew results too strongly.

    Finally, in scratch phase is an algorithm to consider weighted averages.
    '''
    initial_depths = tree.root.depths()
    terminal_depths_df = pd.DataFrame()
    terminal_depths_df['depth'] = np.nan
    for term in tree.get_terminals():
        terminal_depths_df.set_value(term.name, 'depth', initial_depths[term])
    depths_dict = {}
    depths_dict[tree.root] = terminal_depths_df
    explored, function_optima, depths_dict = recursive_crawl_ml(tree.root, [], [], depths_dict, tree)
    function_optima = sorted(function_optima, key=lambda x: x[1].fun)
    tree.root_with_outgroup(function_optima[0][0], outgroup_branch_length=0.)
    assert tree.root.clades[1].branch_length == 0.
    assert tree.root.clades[1] == function_optima[0][0]
    tree.root.clades[0].branch_length -= function_optima[0][1].x[0]
    tree.root.clades[1].branch_length += function_optima[0][1].x[0]
    return tree, function_optima, depths_dict


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
def get_lca_dist_df(tree):
    '''
    Where distance matrix here is subtle. I'm actually calculating the distance to LCA for an initial 
    hypothetical bifurcating root.
    '''
    assert tree.is_bifurcating()
    initial = np.zeros((len(tree.get_terminals()),len(tree.get_terminals())))
    #Call recursive function
    recurse, finished_list = recursive_clade(initial, tree.root, finished=[])
    final = recurse - recurse.diagonal()
    term_names = [i.name for i in tree.get_terminals()]
    final_df = pd.DataFrame(final, index=term_names, columns=term_names)
    return final_df

def recursive_clade(vcv_matrix, initial_clade, finished=[]):
    '''
    This is kind of complicated looking but it should scale linearly with tree size
    '''
    if len(initial_clade) == 2:
        #Add branch length to relevant cells in matrix and move down the left side
        if not set(initial_clade[0].get_terminals()).issubset(set(finished)):
            clade = initial_clade[0]
            clade_term_n = len(clade.get_terminals())
            finished_n = len(finished)
            vcv_matrix[finished_n:finished_n+clade_term_n, finished_n:finished_n+clade_term_n] += clade.branch_length
            vcv_matrix, finished = recursive_clade(vcv_matrix, clade, finished)
        #Add branch length to relevant cells in matrix and move down the right side
        if not set(initial_clade[1].get_terminals()).issubset(set(finished)):
            clade = initial_clade[1]
            clade_term_n = len(clade.get_terminals())
            finished_n = len(finished)
            vcv_matrix[finished_n:finished_n+clade_term_n, finished_n:finished_n+clade_term_n] += clade.branch_length
            vcv_matrix, finished = recursive_clade(vcv_matrix, clade, finished)
    elif len(initial_clade) == 0:
        finished.append(initial_clade)
    else:
        print("ERROR: APPEARS TO BE A NON-BINARY TREE. MATRIX GENERATION WILL PROBABLY FAIL")
    return vcv_matrix, finished

def update_lca_dist_df_dict(lca_dist_df_dict, my_clade, parent, my_tree):
    bl = my_clade.branch_length
    downstream_terms = [i.name for i in my_clade.get_terminals()]
    upstream_terms = list(set([i.name for i in my_tree.get_terminals()]) - set(downstream_terms))
    lca_dist_df = lca_dist_df_dict[parent].copy(deep=True)
    lca_dist_df.loc[downstream_terms,upstream_terms] -= bl
    lca_dist_df.loc[upstream_terms,downstream_terms] += bl
    lca_dist_df_dict[my_clade] = lca_dist_df
    return lca_dist_df_dict, downstream_terms, upstream_terms



def mad_from_df(my_clade, my_terms, other_terms, lca_dist_df):
    '''
    Need to document this
    '''
    my_df = lca_dist_df.loc[my_terms, my_terms]
    other_df = lca_dist_df.loc[other_terms, other_terms]
    my_df_trans = my_df.T
    other_df_trans = other_df.T
    #Dealing with same side pairs
    ss_a_dists = np.abs(np.concatenate((my_df.values[np.triu_indices(len(my_terms), k = 1)],\
                                other_df.values[np.triu_indices(len(other_terms), k = 1)])))
    ss_b_dists = np.abs(np.concatenate((my_df_trans.values[np.triu_indices(len(my_terms), k = 1)],\
                                other_df_trans.values[np.triu_indices(len(other_terms), k = 1)])))
    ss_total_dists = ss_a_dists + ss_b_dists
    ss_devs = np.abs(((2*ss_a_dists)/ss_total_dists)-1)  
    #Dealing with different side pairs
    ds_a_dists = lca_dist_df.loc[my_terms, other_terms].values.flatten(order='C')
    ds_b_dists = lca_dist_df.loc[other_terms, my_terms].values.flatten(order='F')
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
    return (modifier, dev_score)


def recursive_crawl_mad(hypothetical_root, explored, function_optima, tree, lca_dist_df_dict):
    if len(hypothetical_root.clades) == 2:
        l_clade, r_clade = hypothetical_root.clades        
        ###Recurse on l clade
        lca_dist_df_dict, my_terms, other_terms = update_lca_dist_df_dict(lca_dist_df_dict, l_clade, hypothetical_root, tree)
        res = mad_from_df(l_clade, my_terms, other_terms, lca_dist_df_dict[l_clade])
        function_optima.append((l_clade, res))
        explored, function_optima, lca_dist_df_dict = recursive_crawl_mad(l_clade, explored, function_optima, tree, lca_dist_df_dict)
        ###Recurse on r clade
        lca_dist_df_dict, my_terms, other_terms = update_lca_dist_df_dict(lca_dist_df_dict, r_clade, hypothetical_root, tree)
        res = mad_from_df(r_clade, my_terms, other_terms, lca_dist_df_dict[r_clade])
        function_optima.append((r_clade, res))
        explored, function_optima, lca_dist_df_dict = recursive_crawl_mad(r_clade, explored, function_optima, tree, lca_dist_df_dict)
    elif len(hypothetical_root.clades) == 0:
        explored.append(hypothetical_root)
        return explored, function_optima, lca_dist_df_dict
    else:
        print('non binary tree...?')
    explored.append(hypothetical_root)
    return explored, function_optima, lca_dist_df_dict

def mad_root_adhock(tree):
    for node in tree.get_terminals() + tree.get_nonterminals():
        if node == tree.root:
            continue
        if node.branch_length == 0.:
            node.branch_length = 10e-16
    dist_df = get_lca_dist_df(tree)
    tempy_dict = {}
    tempy_dict[tree.root] = dist_df
    explored, function_optima, lca_dist_df_dict = recursive_crawl_mad(tree.root, [], [], tree, tempy_dict)
    function_optima = sorted(function_optima, key=lambda x: x[1][1])
    tree.root_with_outgroup(function_optima[0][0], outgroup_branch_length=0.)
    tree.root.clades[0].branch_length -= function_optima[0][1][0]
    tree.root.clades[1].branch_length += function_optima[0][1][0]
    RAI = function_optima[0][1][1] / function_optima[1][1][1]
    return tree, RAI, function_optima






